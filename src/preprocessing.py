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

    if "published_at" in df.columns:
        df["published_at"] = pd.to_datetime(
            df["published_at"],
            errors="coerce",
            utc=True
        )

    if "collected_at" in df.columns:
        df["collected_at"] = pd.to_datetime(
            df["collected_at"],
            errors="coerce",
            utc=True
        )

    return df


# ============================================================
# HANDLE EMPTY STRINGS
# ============================================================

def handle_empty_strings(df):

    string_columns = df.select_dtypes(
        include=["object", "string"]
    ).columns

    for column in string_columns:
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

    # Text fields
    text_columns = [
        "description",
        "tags",
        "channel_description"
    ]

    for column in text_columns:
        if column in df.columns:
            df[column] = df[column].fillna("")

    # Categorical fields
    categorical_columns = [
        "country"
    ]

    for column in categorical_columns:
        if column in df.columns:
            df[column] = df[column].fillna("Unknown")

    # Engagement metrics
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

    initial_rows = len(df)

    # Video ID is required
    if "video_id" in df.columns:
        df = df.dropna(subset=["video_id"])

    # View count is required for engagement calculations
    if "view_count" in df.columns:
        df = df.dropna(subset=["view_count"])

    # Numerical values cannot be negative
    non_negative_columns = [
        "view_count",
        "like_count",
        "comment_count",
        "subscriber_count",
        "video_count"
    ]

    for column in non_negative_columns:
        if column in df.columns:
            df = df[df[column] >= 0]

    # Likes cannot exceed views
    if "like_count" in df.columns and "view_count" in df.columns:
        df = df[
            df["like_count"] <= df["view_count"]
        ]

    # Comments cannot exceed views
    if "comment_count" in df.columns and "view_count" in df.columns:
        df = df[
            df["comment_count"] <= df["view_count"]
        ]

    # One row per video
    if "video_id" in df.columns:
        df = df.drop_duplicates(
            subset=["video_id"],
            keep="first"
        )

    removed_rows = initial_rows - len(df)

    print("\nINVALID RECORD CLEANING")
    print("-" * 60)
    print(f"Rows before cleaning : {initial_rows:,}")
    print(f"Rows removed         : {removed_rows:,}")
    print(f"Rows after cleaning  : {len(df):,}")

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

    print(f"\nRows    : {len(df):,}")
    print(f"Columns : {df.shape[1]:,}")

    print("\nDuplicate video IDs:")
    if "video_id" in df.columns:
        print(df["video_id"].duplicated().sum())

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
            negative_count = (df[column] < 0).sum()
            print(f"{column:<20}: {negative_count}")

    print("\nMissing values:")

    missing_values = df.isnull().sum()

    for column, count in missing_values.items():
        if count > 0:
            print(f"{column:<25}: {count:,}")

    print("\nValidation completed.")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    df = load_data()

    df_processed = preprocess_data(df)

    save_processed_data(df_processed)

    validate_processed_data(df_processed)

    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETED")
    print("=" * 60)