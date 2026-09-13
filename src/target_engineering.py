import os
import pandas as pd
import numpy as np


FEATURE_DATA_PATH = "data/features/youtube_features.csv"
TARGET_DATA_DIR = "data/features"
TARGET_DATA_PATH = os.path.join(
    TARGET_DATA_DIR,
    "youtube_target_dataset.csv"
)


def load_data(path=FEATURE_DATA_PATH):
    df = pd.read_csv(path)

    print("=" * 70)
    print("FEATURE DATA LOADED")
    print("=" * 70)
    print(f"Rows    : {len(df):,}")
    print(f"Columns : {df.shape[1]:,}")

    return df


def prepare_data(df):
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

    return df


def create_channel_size_group(df):
    """
    Group channels by their current subscriber count.

    These groups are used only as a broad comparison context.
    """

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


def create_topic_group(df):
    """
    Use the existing search category and keyword information
    as a broad topic/context grouping.
    """

    df["topic_group"] = (
        df["search_category"].fillna("Unknown").astype(str)
        + " | "
        + df["search_keyword"].fillna("Unknown").astype(str)
    )

    return df


def calculate_channel_baseline(df):
    """
    Calculate leave-one-out channel median views.

    A video's own views are excluded from its channel baseline.

    This baseline is only considered reliable when a channel
    has enough historical videos.
    """

    channel_stats = (
        df.groupby("channel_id")["view_count"]
        .agg(
            channel_video_count="count",
            channel_view_sum="sum"
        )
    )

    df = df.join(
        channel_stats,
        on="channel_id"
    )

    df["channel_baseline_views"] = np.nan

    enough_history = df["channel_video_count"] >= 3

    df.loc[enough_history, "channel_baseline_views"] = (
        (
            df.loc[enough_history, "channel_view_sum"]
            - df.loc[enough_history, "view_count"]
        )
        /
        (
            df.loc[enough_history, "channel_video_count"]
            - 1
        )
    )

    return df


def calculate_topic_baseline(df):
    """
    Calculate topic-level median views.

    This provides a fallback when channel history is insufficient.
    """

    topic_baseline = (
        df.groupby(
            ["topic_group", "channel_size_group"],
            observed=True
        )["view_count"]
        .transform("median")
    )

    df["topic_size_baseline_views"] = topic_baseline

    return df


def calculate_category_baseline(df):
    """
    Calculate a broader category + channel-size baseline.
    """

    category_baseline = (
        df.groupby(
            ["search_category", "channel_size_group"],
            observed=True
        )["view_count"]
        .transform("median")
    )

    df["category_size_baseline_views"] = category_baseline

    return df


def select_best_baseline(df):
    """
    Select the strongest available baseline.

    Priority:
    1. Channel baseline when enough channel history exists
    2. Topic + channel-size baseline
    3. Category + channel-size baseline
    """

    df["baseline_type"] = "Unavailable"
    df["baseline_views"] = np.nan

    channel_available = (
        df["channel_baseline_views"].notna()
        & (df["channel_baseline_views"] > 0)
    )

    topic_available = (
        df["topic_size_baseline_views"].notna()
        & (df["topic_size_baseline_views"] > 0)
    )

    category_available = (
        df["category_size_baseline_views"].notna()
        & (df["category_size_baseline_views"] > 0)
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

    return df


def create_relative_target(df):
    """
    Create the continuous relative-performance target.

    Positive values:
        Better than baseline

    Around zero:
        Around baseline

    Negative values:
        Worse than baseline
    """

    valid = (
        df["view_count"].notna()
        & df["baseline_views"].notna()
        & (df["view_count"] >= 0)
        & (df["baseline_views"] > 0)
    )

    df["relative_performance"] = np.nan

    df.loc[valid, "relative_performance"] = (
        np.log1p(df.loc[valid, "view_count"])
        -
        np.log1p(df.loc[valid, "baseline_views"])
    )

    return df


def print_baseline_statistics(df):
    print("\n" + "=" * 70)
    print("BASELINE COVERAGE")
    print("=" * 70)

    total = len(df)

    channel_count = (
        df["baseline_type"] == "Channel"
    ).sum()

    topic_count = (
        df["baseline_type"] == "Topic + Channel Size"
    ).sum()

    category_count = (
        df["baseline_type"] == "Category + Channel Size"
    ).sum()

    unavailable_count = (
        df["baseline_type"] == "Unavailable"
    ).sum()

    print(f"\nTotal videos              : {total:,}")
    print(f"Channel baseline          : {channel_count:,}")
    print(f"Topic + channel size      : {topic_count:,}")
    print(f"Category + channel size   : {category_count:,}")
    print(f"Unavailable                : {unavailable_count:,}")

    usable = total - unavailable_count

    print(
        f"\nUsable target rows        : "
        f"{usable:,} ({usable / total * 100:.2f}%)"
    )


def print_target_statistics(df):
    print("\n" + "=" * 70)
    print("RELATIVE PERFORMANCE TARGET")
    print("=" * 70)

    target = df["relative_performance"].dropna()

    print(f"\nValid target rows : {len(target):,}")

    if len(target) == 0:
        print("No valid target values were created.")
        return

    print(f"Mean              : {target.mean():.4f}")
    print(f"Median            : {target.median():.4f}")
    print(f"Std deviation     : {target.std():.4f}")
    print(f"Minimum           : {target.min():.4f}")
    print(f"Maximum           : {target.max():.4f}")

    print("\nTarget percentiles:")
    print(f"10th percentile   : {target.quantile(0.10):.4f}")
    print(f"25th percentile   : {target.quantile(0.25):.4f}")
    print(f"50th percentile   : {target.quantile(0.50):.4f}")
    print(f"75th percentile   : {target.quantile(0.75):.4f}")
    print(f"90th percentile   : {target.quantile(0.90):.4f}")


def print_examples(df):
    print("\n" + "=" * 70)
    print("EXAMPLE TARGET CALCULATIONS")
    print("=" * 70)

    columns = [
        "title",
        "channel_id",
        "subscriber_count",
        "view_count",
        "baseline_views",
        "baseline_type",
        "relative_performance"
    ]

    examples = (
        df[
            df["relative_performance"].notna()
        ][columns]
        .sort_values(
            "relative_performance",
            ascending=False
        )
        .head(5)
    )

    print("\nTop performing relative examples:")
    print(
        examples.to_string(
            index=False
        )
    )

    print("\nLowest performing relative examples:")

    examples_low = (
        df[
            df["relative_performance"].notna()
        ][columns]
        .sort_values(
            "relative_performance",
            ascending=True
        )
        .head(5)
    )

    print(
        examples_low.to_string(
            index=False
        )
    )


def save_target_dataset(df):
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
    print(TARGET_DATA_PATH)


def main():
    df = load_data()

    df = prepare_data(df)
    df = create_channel_size_group(df)
    df = create_topic_group(df)
    df = calculate_channel_baseline(df)
    df = calculate_topic_baseline(df)
    df = calculate_category_baseline(df)
    df = select_best_baseline(df)
    df = create_relative_target(df)

    print_baseline_statistics(df)
    print_target_statistics(df)
    print_examples(df)

    save_target_dataset(df)

    print("\n" + "=" * 70)
    print("TARGET ENGINEERING COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()