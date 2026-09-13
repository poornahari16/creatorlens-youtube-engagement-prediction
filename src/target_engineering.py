import os
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

RAW_DATA_PATH = (
    "data/processed/youtube_dataset_processed.csv"
)

FEATURE_DATA_PATH = (
    "data/features/youtube_model_features.csv"
)

TARGET_DATA_DIR = "data/features"

TARGET_DATA_PATH = os.path.join(
    TARGET_DATA_DIR,
    "youtube_target_dataset.csv"
)


# Minimum number of earlier videos required for a
# reliable group baseline.
MIN_GROUP_SIZE = 10

# Minimum number of earlier videos from the same channel.
MIN_CHANNEL_HISTORY = 3


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    raw_df = pd.read_csv(
        RAW_DATA_PATH
    )

    feature_df = pd.read_csv(
        FEATURE_DATA_PATH
    )

    print("=" * 70)
    print("DATA LOADED")
    print("=" * 70)

    print(
        f"Processed rows : {len(raw_df):,}"
    )

    print(
        f"Feature rows   : {len(feature_df):,}"
    )

    return raw_df, feature_df


# ============================================================
# PREPARE RAW DATA
# ============================================================

def prepare_raw_data(df):

    df = df.copy()

    numeric_columns = [
        "view_count",
        "subscriber_count",
        "video_count",
        "category_id"
    ]

    for column in numeric_columns:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    # Convert publication time to datetime.
    # Missing/invalid values remain NaT.
    df["published_at"] = df["published_at"].apply(
        lambda x: pd.to_datetime(
            x,
            errors="coerce",
            utc=True
        )
    )

    return df


# ============================================================
# CREATE CHANNEL SIZE GROUP
# ============================================================

def create_channel_size_group(df):

    df = df.copy()

    bins = [
        -1,
        1_000,
        10_000,
        100_000,
        1_000_000,
        np.inf
    ]

    labels = [
        "0-1K",
        "1K-10K",
        "10K-100K",
        "100K-1M",
        "1M+"
    ]

    df["channel_size_group"] = pd.cut(
        df["subscriber_count"],
        bins=bins,
        labels=labels
    )

    return df


# ============================================================
# CREATE TOPIC GROUP
# ============================================================

def create_topic_group(df):

    df = df.copy()

    df["search_category"] = (
        df["search_category"]
        .fillna("Unknown")
        .astype(str)
    )

    df["search_keyword"] = (
        df["search_keyword"]
        .fillna("Unknown")
        .astype(str)
    )

    df["topic_group"] = (
        df["search_category"]
        + " | "
        + df["search_keyword"]
    )

    return df


# ============================================================
# TIME-AWARE BASELINE
# ============================================================

def calculate_prior_baseline(
    df,
    group_columns,
    minimum_history,
    statistic="median"
):
    """
    Calculate the baseline for each video using ONLY videos
    published before that video.

    The current video's own views are never included.

    Videos without a valid published_at timestamp cannot
    receive a time-aware baseline.
    """

    baseline = pd.Series(
        np.nan,
        index=df.index,
        dtype=float
    )

    # Only videos with valid publication time and view count
    # can be used to create historical baselines.
    dated_df = df[
        df["published_at"].notna()
        & df["view_count"].notna()
        & (df["view_count"] >= 0)
    ].copy()

    if dated_df.empty:
        return baseline

    # Sort chronologically.
    dated_df = dated_df.sort_values(
        ["published_at", "video_id"]
    )

    grouped = dated_df.groupby(
        group_columns,
        observed=True,
        sort=False
    )

    for _, group in grouped:

        values = group["view_count"].to_numpy(
            dtype=float
        )

        indices = group.index.to_list()

        dates = group["published_at"].to_numpy()

        # Calculate each video's baseline independently.
        for position, index in enumerate(indices):

            current_date = dates[position]

            # IMPORTANT:
            # Only strictly earlier videos are allowed.
            prior_values = values[
                dates < current_date
            ]

            prior_values = prior_values[
                ~np.isnan(prior_values)
            ]

            if len(prior_values) >= minimum_history:

                if statistic == "mean":

                    baseline.loc[index] = (
                        np.mean(prior_values)
                    )

                else:

                    baseline.loc[index] = (
                        np.median(prior_values)
                    )

    return baseline


# ============================================================
# CHANNEL BASELINE
# ============================================================

def calculate_channel_baseline(df):

    df = df.copy()

    df["channel_baseline_views"] = (
        calculate_prior_baseline(
            df=df,
            group_columns=["channel_id"],
            minimum_history=MIN_CHANNEL_HISTORY,
            statistic="mean"
        )
    )

    return df


# ============================================================
# TOPIC + CHANNEL SIZE BASELINE
# ============================================================

def calculate_topic_baseline(df):

    df = df.copy()

    df["topic_size_baseline_views"] = (
        calculate_prior_baseline(
            df=df,
            group_columns=[
                "topic_group",
                "channel_size_group"
            ],
            minimum_history=MIN_GROUP_SIZE,
            statistic="median"
        )
    )

    return df


# ============================================================
# CATEGORY + CHANNEL SIZE BASELINE
# ============================================================

def calculate_category_baseline(df):

    df = df.copy()

    df["category_size_baseline_views"] = (
        calculate_prior_baseline(
            df=df,
            group_columns=[
                "search_category",
                "channel_size_group"
            ],
            minimum_history=MIN_GROUP_SIZE,
            statistic="median"
        )
    )

    return df


# ============================================================
# CHANNEL SIZE BASELINE
# ============================================================

def calculate_size_baseline(df):

    df = df.copy()

    df["size_baseline_views"] = (
        calculate_prior_baseline(
            df=df,
            group_columns=["channel_size_group"],
            minimum_history=MIN_GROUP_SIZE,
            statistic="median"
        )
    )

    return df


# ============================================================
# SELECT BEST BASELINE
# ============================================================

def select_best_baseline(df):

    df = df.copy()

    df["baseline_type"] = "Unavailable"

    df["baseline_views"] = np.nan

    # --------------------------------------------------------
    # 1. CHANNEL
    # --------------------------------------------------------

    channel_available = (
        df["channel_baseline_views"].notna()
        & (df["channel_baseline_views"] > 0)
    )

    df.loc[
        channel_available,
        "baseline_views"
    ] = df.loc[
        channel_available,
        "channel_baseline_views"
    ]

    df.loc[
        channel_available,
        "baseline_type"
    ] = "Channel"

    # --------------------------------------------------------
    # 2. TOPIC + CHANNEL SIZE
    # --------------------------------------------------------

    topic_available = (
        df["topic_size_baseline_views"].notna()
        & (df["topic_size_baseline_views"] > 0)
    )

    topic_fallback = (
        ~channel_available
        & topic_available
    )

    df.loc[
        topic_fallback,
        "baseline_views"
    ] = df.loc[
        topic_fallback,
        "topic_size_baseline_views"
    ]

    df.loc[
        topic_fallback,
        "baseline_type"
    ] = "Topic + Channel Size"

    # --------------------------------------------------------
    # 3. CATEGORY + CHANNEL SIZE
    # --------------------------------------------------------

    category_available = (
        df["category_size_baseline_views"].notna()
        & (df["category_size_baseline_views"] > 0)
    )

    category_fallback = (
        ~channel_available
        & ~topic_fallback
        & category_available
    )

    df.loc[
        category_fallback,
        "baseline_views"
    ] = df.loc[
        category_fallback,
        "category_size_baseline_views"
    ]

    df.loc[
        category_fallback,
        "baseline_type"
    ] = "Category + Channel Size"

    # --------------------------------------------------------
    # 4. CHANNEL SIZE
    # --------------------------------------------------------

    size_available = (
        df["size_baseline_views"].notna()
        & (df["size_baseline_views"] > 0)
    )

    size_fallback = (
        ~channel_available
        & ~topic_fallback
        & ~category_fallback
        & size_available
    )

    df.loc[
        size_fallback,
        "baseline_views"
    ] = df.loc[
        size_fallback,
        "size_baseline_views"
    ]

    df.loc[
        size_fallback,
        "baseline_type"
    ] = "Channel Size"

    return df


# ============================================================
# CREATE TARGET
# ============================================================

def create_relative_target(df):

    df = df.copy()

    valid = (
        df["view_count"].notna()
        & df["baseline_views"].notna()
        & (df["view_count"] >= 0)
        & (df["baseline_views"] > 0)
    )

    df["relative_performance"] = np.nan

    df.loc[
        valid,
        "relative_performance"
    ] = (
        np.log1p(
            df.loc[
                valid,
                "view_count"
            ]
        )
        -
        np.log1p(
            df.loc[
                valid,
                "baseline_views"
            ]
        )
    )

    return df


# ============================================================
# BUILD FINAL TARGET DATASET
# ============================================================

def build_target_dataset(
    raw_df,
    feature_df
):

    # Only bring the outcome into the feature dataset.
    target_info = raw_df[
        [
            "video_id",
            "view_count"
        ]
    ].copy()

    final_df = feature_df.merge(
        target_info,
        on="video_id",
        how="left"
    )

    # Add baseline and target information.
    target_columns = [
        "video_id",
        "baseline_views",
        "baseline_type",
        "relative_performance"
    ]

    final_df = final_df.merge(
        raw_df[target_columns],
        on="video_id",
        how="left"
    )

    return final_df


# ============================================================
# PRINT STATISTICS
# ============================================================

def print_statistics(df):

    print("\n" + "=" * 70)
    print("BASELINE COVERAGE")
    print("=" * 70)

    total = len(df)

    counts = (
        df["baseline_type"]
        .value_counts()
    )

    for baseline_type in [
        "Channel",
        "Topic + Channel Size",
        "Category + Channel Size",
        "Channel Size",
        "Unavailable"
    ]:

        count = counts.get(
            baseline_type,
            0
        )

        percentage = (
            count / total * 100
        )

        print(
            f"{baseline_type:28s}: "
            f"{count:,} "
            f"({percentage:.2f}%)"
        )

    usable = (
        df["relative_performance"]
        .notna()
        .sum()
    )

    print(
        f"\nUsable target rows        : "
        f"{usable:,} "
        f"({usable / total * 100:.2f}%)"
    )

    print("\n" + "=" * 70)
    print("RELATIVE PERFORMANCE TARGET")
    print("=" * 70)

    target = (
        df["relative_performance"]
        .dropna()
    )

    print(
        f"\nValid target rows : "
        f"{len(target):,}"
    )

    if len(target) == 0:
        return

    print(
        f"Mean              : "
        f"{target.mean():.4f}"
    )

    print(
        f"Median            : "
        f"{target.median():.4f}"
    )

    print(
        f"Std deviation     : "
        f"{target.std():.4f}"
    )

    print(
        f"Minimum           : "
        f"{target.min():.4f}"
    )

    print(
        f"Maximum           : "
        f"{target.max():.4f}"
    )

    print("\nTarget percentiles:")

    for percentile in [
        0.10,
        0.25,
        0.50,
        0.75,
        0.90
    ]:

        print(
            f"{int(percentile * 100):>2}th percentile   : "
            f"{target.quantile(percentile):.4f}"
        )


# ============================================================
# EXAMPLES
# ============================================================

def print_examples(df):

    print("\n" + "=" * 70)
    print("EXAMPLE TARGET CALCULATIONS")
    print("=" * 70)

    columns = [
        "title",
        "channel_id",
        "subscriber_count",
        "published_at",
        "view_count",
        "baseline_views",
        "baseline_type",
        "relative_performance"
    ]

    valid_df = df[
        df["relative_performance"].notna()
    ]

    if valid_df.empty:
        print("No valid target examples available.")
        return

    print("\nTop relative performers:")

    top = (
        valid_df[
            columns
        ]
        .sort_values(
            "relative_performance",
            ascending=False
        )
        .head(5)
    )

    print(
        top.to_string(
            index=False
        )
    )

    print("\nLowest relative performers:")

    bottom = (
        valid_df[
            columns
        ]
        .sort_values(
            "relative_performance",
            ascending=True
        )
        .head(5)
    )

    print(
        bottom.to_string(
            index=False
        )
    )


# ============================================================
# SAVE
# ============================================================

def save_dataset(df):

    os.makedirs(
        TARGET_DATA_DIR,
        exist_ok=True
    )

    df.to_csv(
        TARGET_DATA_PATH,
        index=False
    )

    print("\n" + "=" * 70)
    print("TARGET DATASET SAVED")
    print("=" * 70)

    print(
        f"Path    : {TARGET_DATA_PATH}"
    )

    print(
        f"Rows    : {len(df):,}"
    )

    print(
        f"Columns : {df.shape[1]:,}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    # 1. Load processed data and model features
    raw_df, feature_df = load_data()

    # 2. Prepare raw data
    raw_df = prepare_raw_data(
        raw_df
    )

    # 3. Create grouping information
    raw_df = create_channel_size_group(
        raw_df
    )

    raw_df = create_topic_group(
        raw_df
    )

    # 4. Calculate TIME-AWARE baselines
    raw_df = calculate_channel_baseline(
        raw_df
    )

    raw_df = calculate_topic_baseline(
        raw_df
    )

    raw_df = calculate_category_baseline(
        raw_df
    )

    raw_df = calculate_size_baseline(
        raw_df
    )

    # 5. Select best baseline
    raw_df = select_best_baseline(
        raw_df
    )

    # 6. Create target
    raw_df = create_relative_target(
        raw_df
    )

    # 7. Merge target with model features
    final_df = build_target_dataset(
        raw_df,
        feature_df
    )

    # 8. Print statistics
    print_statistics(
        raw_df
    )

    print_examples(
        raw_df
    )

    # 9. Save
    save_dataset(
        final_df
    )

    print("\n" + "=" * 70)
    print("TARGET ENGINEERING COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()