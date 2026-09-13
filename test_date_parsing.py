import pandas as pd


RAW_DATA_PATH = "data/raw/youtube_dataset.csv"


print("=" * 60)
print("DATE PARSING DIAGNOSTIC")
print("=" * 60)


# Load raw CSV WITHOUT any preprocessing
df = pd.read_csv(RAW_DATA_PATH)

print("\nRAW DATA")
print("-" * 60)

print(f"Rows: {len(df):,}")
print(f"published_at dtype: {df['published_at'].dtype}")
print(f"Missing raw dates: {df['published_at'].isna().sum():,}")


# Show examples
print("\nRAW DATE EXAMPLES")
print("-" * 60)

print(
    df["published_at"]
    .dropna()
    .head(10)
    .to_string(index=False)
)


# ------------------------------------------------------------
# TEST 1: Current parsing method
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("TEST 1: pd.to_datetime(..., utc=True)")
print("=" * 60)

parsed_1 = pd.to_datetime(
    df["published_at"],
    errors="coerce",
    utc=True
)

print(f"Valid: {parsed_1.notna().sum():,}")
print(f"Missing: {parsed_1.isna().sum():,}")


# ------------------------------------------------------------
# TEST 2: Parse strings individually
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("TEST 2: INDIVIDUAL DATE PARSING")
print("=" * 60)

parsed_individual = df["published_at"].apply(
    lambda x: pd.to_datetime(
        x,
        errors="coerce",
        utc=True
    )
)

print(f"Valid: {parsed_individual.notna().sum():,}")
print(f"Missing: {parsed_individual.isna().sum():,}")


# ------------------------------------------------------------
# Find dates that fail
# ------------------------------------------------------------

failed = df.loc[
    parsed_1.isna()
    & df["published_at"].notna(),
    "published_at"
]

print("\n" + "=" * 60)
print("FAILED DATE EXAMPLES")
print("=" * 60)

print(f"Number of failed dates: {len(failed):,}")

if len(failed) > 0:

    print("\nFirst 20 failed raw values:")

    print(
        failed.head(20)
        .to_string(index=False)
    )


# ------------------------------------------------------------
# Compare successful and failed formats
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("DATE STRING CHARACTERISTICS")
print("=" * 60)

if len(failed) > 0:

    example = failed.iloc[0]

    print("\nFirst failed value:")
    print(repr(example))

    print("\nLength:")
    print(len(str(example)))

    print("\nCharacters:")
    print(
        [
            (character, ord(character))
            for character in str(example)
        ]
    )


print("\n" + "=" * 60)
print("DIAGNOSTIC COMPLETED")
print("=" * 60)