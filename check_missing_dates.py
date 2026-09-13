import pandas as pd


# ============================================================
# PATHS
# ============================================================

RAW_PATH = "data/raw/youtube_dataset.csv"
PROCESSED_PATH = "data/processed/youtube_dataset_processed.csv"


# ============================================================
# LOAD
# ============================================================

raw = pd.read_csv(RAW_PATH)

processed = pd.read_csv(PROCESSED_PATH)


# ============================================================
# BASIC INFORMATION
# ============================================================

print("=" * 70)
print("MISSING PUBLICATION DATE INVESTIGATION")
print("=" * 70)

print(f"Raw rows       : {len(raw):,}")
print(f"Processed rows : {len(processed):,}")


# ============================================================
# RAW DATA
# ============================================================

print("\n" + "=" * 70)
print("RAW DATA")
print("=" * 70)

if "published_at" in raw.columns:

    raw_missing = raw["published_at"].isna()

    print(
        f"published_at column exists : True"
    )

    print(
        f"Missing published_at       : "
        f"{raw_missing.sum():,}"
    )

    print(
        f"Valid published_at         : "
        f"{raw['published_at'].notna().sum():,}"
    )

else:

    print(
        "published_at column exists : False"
    )


# ============================================================
# PROCESSED DATA
# ============================================================

print("\n" + "=" * 70)
print("PROCESSED DATA")
print("=" * 70)

if "published_at" in processed.columns:

    processed_missing = processed["published_at"].isna()

    print(
        f"published_at column exists : True"
    )

    print(
        f"Missing published_at       : "
        f"{processed_missing.sum():,}"
    )

    print(
        f"Valid published_at         : "
        f"{processed['published_at'].notna().sum():,}"
    )

else:

    print(
        "published_at column exists : False"
    )


# ============================================================
# CHECK WHETHER PREPROCESSING LOST DATES
# ============================================================

print("\n" + "=" * 70)
print("RAW → PROCESSED COMPARISON")
print("=" * 70)

if (
    "video_id" in raw.columns
    and "video_id" in processed.columns
    and "published_at" in raw.columns
    and "published_at" in processed.columns
):

    comparison = raw[
        [
            "video_id",
            "published_at"
        ]
    ].merge(
        processed[
            [
                "video_id",
                "published_at"
            ]
        ],
        on="video_id",
        how="inner",
        suffixes=("_raw", "_processed")
    )

    raw_has_date = (
        comparison["published_at_raw"].notna()
    )

    processed_missing_date = (
        comparison["published_at_processed"].isna()
    )

    lost_dates = comparison[
        raw_has_date
        & processed_missing_date
    ]

    print(
        f"Videos where raw has date but "
        f"processed is missing : {len(lost_dates):,}"
    )

    if len(lost_dates) > 0:

        print("\nExamples of potentially lost dates:")

        print(
            lost_dates.head(20).to_string(
                index=False
            )
        )

    else:

        print(
            "No publication dates appear to have "
            "been lost during preprocessing."
        )


# ============================================================
# CHECK RAW MISSING ROWS
# ============================================================

print("\n" + "=" * 70)
print("RAW MISSING-DATE EXAMPLES")
print("=" * 70)

if "published_at" in raw.columns:

    missing_rows = raw[
        raw["published_at"].isna()
    ].copy()

    print(
        f"Rows with missing date : "
        f"{len(missing_rows):,}"
    )

    columns_to_show = [
        column
        for column in [
            "video_id",
            "title",
            "channel_id",
            "channel_title",
            "view_count",
            "collected_at"
        ]
        if column in missing_rows.columns
    ]

    print(
        missing_rows[
            columns_to_show
        ]
        .head(20)
        .to_string(index=False)
    )


# ============================================================
# CHECK WHETHER ANOTHER DATE COLUMN EXISTS
# ============================================================

print("\n" + "=" * 70)
print("DATE-LIKE COLUMNS")
print("=" * 70)

date_keywords = [
    "date",
    "time",
    "published",
    "created",
    "updated"
]

date_like_columns = [
    column
    for column in raw.columns
    if any(
        keyword in column.lower()
        for keyword in date_keywords
    )
]

print(
    "Possible date/time columns:"
)

for column in date_like_columns:

    print(
        f" - {column}"
    )


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("INVESTIGATION COMPLETED")
print("=" * 70)