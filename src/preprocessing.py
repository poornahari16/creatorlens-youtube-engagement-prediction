import os
import pandas as pd


# ============================================================
# PATHS
# ============================================================

RAW_DATA_PATH = "data/raw/youtube_dataset.csv"

PROCESSED_DATA_DIR = "data/processed"

PROCESSED_DATA_PATH = os.path.join(
    PROCESSED_DATA_DIR,
    "youtube_dataset_processed.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

def load_data(path=RAW_DATA_PATH):

    df = pd.read_csv(path)

    print("=" * 60)
    print("DATA LOADED")
    print("=" * 60)

    print(f"Rows    : {len(df):,}")
    print(f"Columns : {df.shape[1]:,}")

    return df


# ============================================================
# CLEAN DATA TYPES
# ============================================================

def clean_data_types(df):

    df = df.copy()

    # --------------------------------------------------------
    # Numerical columns
    # --------------------------------------------------------

    numeric_columns = [
        "category_id",
        "view_count",
        "like_count",
        "comment_count",
        "subscriber_count",
        "video_count"
    ]

    for column in numeric_columns:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    # --------------------------------------------------------
    # Publication timestamp
    # --------------------------------------------------------

    if "published_at" in df.columns:

        df["published_at"] = df["published_at"].apply(
            lambda x: pd.to_datetime(
                x,
                errors="coerce",
                utc=True
            )
        )
    # --------------------------------------------------------
    # Collection timestamp
    # --------------------------------------------------------

    if "collected_at" in df.columns:

        df["collected_at"] = df["collected_at"].apply(
            lambda x: pd.to_datetime(
                x,
                errors="coerce",
                utc=True
            )
        )
    return df

# ============================================================
# HANDLE EMPTY STRINGS
# ============================================================

def handle_empty_strings(df):

    df = df.copy()

    # Only process actual text columns.
    #
    # Datetime columns are deliberately excluded so that
    # published_at and collected_at remain datetime values.

    text_columns = df.select_dtypes(
        include=["object", "string"]
    ).columns.tolist()

    datetime_columns = df.select_dtypes(
        include=["datetime64[ns]", "datetime64[ns, UTC]"]
    ).columns.tolist()

    text_columns = [
        column
        for column in text_columns
        if column not in datetime_columns
    ]

    for column in text_columns:

        df[column] = df[column].replace(
            r"^\s*$",
            pd.NA,
            regex=True
        )

    return df


# ============================================================
# HANDLE MISSING VALUES
# ============================================================

def handle_missing_values(df):

    df = df.copy()

    # --------------------------------------------------------
    # Text fields
    # --------------------------------------------------------

    text_columns = [
        "description",
        "tags",
        "channel_description"
    ]

    for column in text_columns:

        if column in df.columns:

            df[column] = df[column].fillna("")


    # --------------------------------------------------------
    # Categorical fields
    # --------------------------------------------------------

    categorical_columns = [
        "country"
    ]

    for column in categorical_columns:

        if column in df.columns:

            df[column] = df[column].fillna(
                "Unknown"
            )


    # --------------------------------------------------------
    # Engagement metrics
    # --------------------------------------------------------

    numeric_columns = [
        "like_count",
        "comment_count"
    ]

    for column in numeric_columns:

        if column in df.columns:

            df[column] = df[column].fillna(0)


    return df


# ============================================================
# REMOVE INVALID RECORDS
# ============================================================

def remove_invalid_records(df):

    df = df.copy()

    initial_rows = len(df)

    # --------------------------------------------------------
    # Video ID is required
    # --------------------------------------------------------

    if "video_id" in df.columns:

        df = df.dropna(
            subset=["video_id"]
        )


    # --------------------------------------------------------
    # View count is required
    # --------------------------------------------------------

    if "view_count" in df.columns:

        df = df.dropna(
            subset=["view_count"]
        )


    # --------------------------------------------------------
    # Numerical values cannot be negative
    # --------------------------------------------------------

    non_negative_columns = [
        "view_count",
        "like_count",
        "comment_count",
        "subscriber_count",
        "video_count"
    ]

    for column in non_negative_columns:

        if column in df.columns:

            df = df[
                df[column] >= 0
            ]


    # --------------------------------------------------------
    # Likes cannot exceed views
    # --------------------------------------------------------

    if (
        "like_count" in df.columns
        and "view_count" in df.columns
    ):

        df = df[
            df["like_count"]
            <= df["view_count"]
        ]


    # --------------------------------------------------------
    # Comments cannot exceed views
    # --------------------------------------------------------

    if (
        "comment_count" in df.columns
        and "view_count" in df.columns
    ):

        df = df[
            df["comment_count"]
            <= df["view_count"]
        ]


    # --------------------------------------------------------
    # One row per video
    # --------------------------------------------------------

    if "video_id" in df.columns:

        df = df.drop_duplicates(
            subset=["video_id"],
            keep="first"
        )


    removed_rows = (
        initial_rows - len(df)
    )

    print("\nINVALID RECORD CLEANING")
    print("-" * 60)

    print(
        f"Rows before cleaning : "
        f"{initial_rows:,}"
    )

    print(
        f"Rows removed         : "
        f"{removed_rows:,}"
    )

    print(
        f"Rows after cleaning  : "
        f"{len(df):,}"
    )

    return df


# ============================================================
# PREPROCESS DATA
# ============================================================

def preprocess_data(df):

    df = clean_data_types(df)

    df = handle_empty_strings(df)

    df = handle_missing_values(df)

    df = remove_invalid_records(df)

    return df


# ============================================================
# SAVE PROCESSED DATA
# ============================================================

def save_processed_data(
    df,
    path=PROCESSED_DATA_PATH
):

    os.makedirs(
        PROCESSED_DATA_DIR,
        exist_ok=True
    )

    df.to_csv(
        path,
        index=False
    )

    print("\nPROCESSED DATA SAVED")
    print("-" * 60)

    print(path)


# ============================================================
# VALIDATE PROCESSED DATA
# ============================================================

def validate_processed_data(df):

    print("\n" + "=" * 60)
    print("PROCESSED DATA VALIDATION")
    print("=" * 60)

    print(
        f"\nRows    : {len(df):,}"
    )

    print(
        f"Columns : {df.shape[1]:,}"
    )


    # --------------------------------------------------------
    # Duplicate video IDs
    # --------------------------------------------------------

    print("\nDuplicate video IDs:")

    if "video_id" in df.columns:

        print(
            df["video_id"]
            .duplicated()
            .sum()
        )


    # --------------------------------------------------------
    # Negative numerical values
    # --------------------------------------------------------

    print("\nNegative numerical values:")

    numeric_columns = [
        "view_count",
        "like_count",
        "comment_count",
        "subscriber_count",
        "video_count"
    ]

    for column in numeric_columns:

        if column in df.columns:

            negative_count = (
                df[column] < 0
            ).sum()

            print(
                f"{column:<20}: "
                f"{negative_count}"
            )


    # --------------------------------------------------------
    # Publication timestamp validation
    # --------------------------------------------------------

    print("\nPublication timestamp:")

    if "published_at" in df.columns:

        missing_published = (
            df["published_at"].isna()
        )

        print(
            f"Missing published_at : "
            f"{missing_published.sum():,}"
        )

        print(
            f"Valid published_at   : "
            f"{df['published_at'].notna().sum():,}"
        )


    # --------------------------------------------------------
    # Collection timestamp validation
    # --------------------------------------------------------

    print("\nCollection timestamp:")

    if "collected_at" in df.columns:

        missing_collected = (
            df["collected_at"].isna()
        )

        print(
            f"Missing collected_at : "
            f"{missing_collected.sum():,}"
        )

        print(
            f"Valid collected_at   : "
            f"{df['collected_at'].notna().sum():,}"
        )


    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    print("\nMissing values:")

    missing_values = (
        df.isnull().sum()
    )

    for column, count in missing_values.items():

        if count > 0:

            print(
                f"{column:<25}: "
                f"{count:,}"
            )


    print("\nValidation completed.")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    df = load_data()

    df_processed = preprocess_data(
        df
    )

    save_processed_data(
        df_processed
    )

    validate_processed_data(
        df_processed
    )

    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETED")
    print("=" * 60)