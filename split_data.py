import os
import pandas as pd


# ============================================================
# PATHS
# ============================================================

INPUT_PATH = "data/features/youtube_target_dataset.csv"

OUTPUT_DIR = "data/splits"


# ============================================================
# SETTINGS
# ============================================================

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    df = pd.read_csv(INPUT_PATH)

    print("=" * 70)
    print("DATA LOADED")
    print("=" * 70)

    print(f"Total rows : {len(df):,}")

    return df


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_data(df):

    df = df.copy()

    # Parse each timestamp individually.
    # The dataset contains mixed ISO timestamp formats,
    # including fractional seconds.
    df["published_at"] = df["published_at"].apply(
        lambda x: pd.to_datetime(
            x,
            errors="coerce",
            utc=True
        )
    )

    # Only rows with a valid target are eligible
    # for the ML dataset.
    df = df[
        df["relative_performance"].notna()
    ].copy()

    # A time-based split requires a valid publication date.
    df = df[
        df["published_at"].notna()
    ].copy()

    # Sort oldest → newest.
    df = df.sort_values(
        ["published_at", "video_id"]
    ).reset_index(drop=True)

    return df


# ============================================================
# CREATE CHRONOLOGICAL SPLIT
# ============================================================

def create_split(df):

    total = len(df)

    train_end = int(
        total * TRAIN_RATIO
    )

    validation_end = int(
        total * (
            TRAIN_RATIO +
            VALIDATION_RATIO
        )
    )

    train_df = df.iloc[
        :train_end
    ].copy()

    validation_df = df.iloc[
        train_end:validation_end
    ].copy()

    test_df = df.iloc[
        validation_end:
    ].copy()

    return (
        train_df,
        validation_df,
        test_df
    )


# ============================================================
# PRINT SPLIT INFORMATION
# ============================================================

def print_split_info(
    name,
    df
):

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    print(
        f"Rows : {len(df):,}"
    )

    if len(df) > 0:

        print(
            f"Start : {df['published_at'].min()}"
        )

        print(
            f"End   : {df['published_at'].max()}"
        )

    print(
        f"Target mean   : "
        f"{df['relative_performance'].mean():.4f}"
    )

    print(
        f"Target median : "
        f"{df['relative_performance'].median():.4f}"
    )


# ============================================================
# VALIDATE SPLIT
# ============================================================

def validate_split(
    train_df,
    validation_df,
    test_df
):

    print("\n" + "=" * 70)
    print("SPLIT VALIDATION")
    print("=" * 70)

    # Check chronological ordering.
    train_before_validation = (
        train_df["published_at"].max()
        <
        validation_df["published_at"].min()
    )

    validation_before_test = (
        validation_df["published_at"].max()
        <
        test_df["published_at"].min()
    )

    print(
        "Train ends before validation starts: "
        f"{train_before_validation}"
    )

    print(
        "Validation ends before test starts: "
        f"{validation_before_test}"
    )

    # Check duplicate video IDs between sets.
    train_ids = set(train_df["video_id"])
    validation_ids = set(validation_df["video_id"])
    test_ids = set(test_df["video_id"])

    train_validation_overlap = (
        len(train_ids & validation_ids)
    )

    validation_test_overlap = (
        len(validation_ids & test_ids)
    )

    train_test_overlap = (
        len(train_ids & test_ids)
    )

    print(
        f"Train / Validation ID overlap: "
        f"{train_validation_overlap}"
    )

    print(
        f"Validation / Test ID overlap: "
        f"{validation_test_overlap}"
    )

    print(
        f"Train / Test ID overlap: "
        f"{train_test_overlap}"
    )

    # Overall validation result.
    passed = (
        train_before_validation
        and validation_before_test
        and train_validation_overlap == 0
        and validation_test_overlap == 0
        and train_test_overlap == 0
    )

    print(
        f"Overall split validation: "
        f"{'PASS' if passed else 'FAIL'}"
    )

    if not passed:
        raise ValueError(
            "Chronological split validation failed."
        )


# ============================================================
# SAVE
# ============================================================

def save_split(
    df,
    filename
):

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    path = os.path.join(
        OUTPUT_DIR,
        filename
    )

    df.to_csv(
        path,
        index=False
    )

    print(
        f"Saved: {path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    # 1. Load
    df = load_data()

    # 2. Prepare
    df = prepare_data(df)

    print("\n" + "=" * 70)
    print("USABLE DATA FOR TIME-BASED SPLIT")
    print("=" * 70)

    print(
        f"Rows : {len(df):,}"
    )

    # 3. Split chronologically
    (
        train_df,
        validation_df,
        test_df
    ) = create_split(df)

    # 4. Print information
    print_split_info(
        "TRAIN SET",
        train_df
    )

    print_split_info(
        "VALIDATION SET",
        validation_df
    )

    print_split_info(
        "TEST SET",
        test_df
    )

    # 5. Validate split
    validate_split(
        train_df,
        validation_df,
        test_df
    )

    # 6. Save
    save_split(
        train_df,
        "train.csv"
    )

    save_split(
        validation_df,
        "validation.csv"
    )

    save_split(
        test_df,
        "test.csv"
    )

    print("\n" + "=" * 70)
    print("DATA SPLIT COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()