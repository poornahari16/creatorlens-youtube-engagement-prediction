import os
import re
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

INPUT_PATH = "data/processed/youtube_dataset_processed.csv"
OUTPUT_DIR = "data/features"
OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "youtube_model_features.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

def load_data(path=INPUT_PATH):
    """
    Load the cleaned/processed YouTube dataset.
    """
    df = pd.read_csv(path)

    print("=" * 70)
    print("DATA LOADED")
    print("=" * 70)
    print(f"Rows    : {len(df):,}")
    print(f"Columns : {df.shape[1]:,}")

    return df


# ============================================================
# BASIC TYPE CONVERSION
# ============================================================

def prepare_data(df):
    """
    Convert important columns into appropriate data types.
    """
    df = df.copy()

    numeric_columns = [
        "view_count",
        "like_count",
        "comment_count",
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

    if "published_at" in df.columns:
        df["published_at"] = df["published_at"].apply(
            lambda x: pd.to_datetime(
                x,
                errors="coerce",
                utc=True
            )
        )
        return df


# ============================================================
# TEXT CLEANING HELPER
# ============================================================

def safe_text(series):
    """
    Convert text columns to clean strings.
    Missing values become empty strings.
    """
    return series.fillna("").astype(str).str.strip()


# ============================================================
# CHANNEL FEATURES
# ============================================================

def create_channel_features(df):
    """
    Create features describing the size and basic characteristics
    of the channel.
    """

    df = df.copy()

    df["subscriber_count"] = df["subscriber_count"].fillna(0)
    df["video_count"] = df["video_count"].fillna(0)

    # Log transforms reduce the impact of extremely large channels.
    df["log_subscriber_count"] = np.log1p(
        df["subscriber_count"].clip(lower=0)
    )

    df["log_video_count"] = np.log1p(
        df["video_count"].clip(lower=0)
    )

    # Channel size groups help the model compare channels
    # at roughly similar scales.
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

    # Channel description length
    df["channel_description"] = safe_text(
        df["channel_description"]
    )

    df["channel_description_length"] = (
        df["channel_description"].str.len()
    )

    df["channel_description_word_count"] = (
        df["channel_description"]
        .str.split()
        .str.len()
    )

    return df


# ============================================================
# TITLE FEATURES
# ============================================================

def create_title_features(df):
    """
    Create structural features from video titles.
    """

    df = df.copy()

    df["title"] = safe_text(df["title"])

    # Basic length features
    df["title_length"] = df["title"].str.len()

    df["title_word_count"] = (
        df["title"]
        .str.split()
        .str.len()
    )

    # Punctuation
    df["title_exclamation_count"] = (
        df["title"].str.count("!")
    )

    df["title_question_count"] = (
        df["title"].str.count(r"\?")
    )

    # Numbers
    df["title_digit_count"] = (
        df["title"].str.count(r"\d")
    )

    df["title_has_number"] = (
        df["title_digit_count"] > 0
    ).astype(int)

    df["title_has_question"] = (
        df["title_question_count"] > 0
    ).astype(int)

    df["title_has_exclamation"] = (
        df["title_exclamation_count"] > 0
    ).astype(int)

    # Uppercase ratio
    def uppercase_ratio(text):
        letters = [
            character
            for character in text
            if character.isalpha()
        ]

        if not letters:
            return 0.0

        uppercase_letters = sum(
            character.isupper()
            for character in letters
        )

        return uppercase_letters / len(letters)

    df["title_uppercase_ratio"] = (
        df["title"].apply(uppercase_ratio)
    )

    return df


# ============================================================
# DESCRIPTION FEATURES
# ============================================================

def create_description_features(df):
    """
    Create structural features from video descriptions.
    """

    df = df.copy()

    df["description"] = safe_text(df["description"])

    df["description_length"] = (
        df["description"].str.len()
    )

    df["description_word_count"] = (
        df["description"]
        .str.split()
        .str.len()
    )

    # Hashtags
    df["description_hashtag_count"] = (
        df["description"].str.count(r"#\w+")
    )

    df["description_hashtag_present"] = (
        df["description_hashtag_count"] > 0
    ).astype(int)

    # URLs
    url_pattern = r"https?://\S+|www\.\S+"

    df["description_url_count"] = (
        df["description"].str.count(url_pattern)
    )

    df["description_url_present"] = (
        df["description_url_count"] > 0
    ).astype(int)

    return df


# ============================================================
# TAG FEATURES
# ============================================================

def create_tag_features(df):
    """
    Create simple structural features from video tags.
    """

    df = df.copy()

    df["tags"] = safe_text(df["tags"])

    def parse_tags(tag_string):
        if not tag_string:
            return []

        # Tags in the collected dataset may use "|" as a separator.
        # If "|" is not present, treat the whole string as one tag.
        if "|" in tag_string:
            tags = tag_string.split("|")
        else:
            tags = [tag_string]

        return [
            tag.strip()
            for tag in tags
            if tag.strip()
        ]

    parsed_tags = df["tags"].apply(parse_tags)

    df["tag_count"] = parsed_tags.apply(len)

    df["tag_text_length"] = (
        parsed_tags.apply(
            lambda tags: sum(len(tag) for tag in tags)
        )
    )

    df["tag_present"] = (
        df["tag_count"] > 0
    ).astype(int)

    return df


# ============================================================
# TOPIC FEATURES
# ============================================================

def create_topic_features(df):
    """
    Create topic/context features.
    """

    df = df.copy()

    df["search_category"] = safe_text(
        df["search_category"]
    )

    df["search_keyword"] = safe_text(
        df["search_keyword"]
    )

    df["topic_group"] = (
        df["search_category"]
        + " | "
        + df["search_keyword"]
    )

    return df


# ============================================================
# DURATION FEATURES
# ============================================================

def parse_duration(duration):
    """
    Convert ISO 8601 YouTube duration into seconds.

    Examples:
    PT5M30S       -> 330
    PT1H2M10S     -> 3730
    PT45S         -> 45
    P1DT1H30M     -> 91800
    P2DT14H55M8S  -> 226508

    P0D represents zero duration and is returned as 0.
    """

    if pd.isna(duration):
        return np.nan

    duration = str(duration).strip()

    if not duration:
        return np.nan

    # Handle zero-duration values such as P0D.
    if duration == "P0D":
        return 0

    pattern = (
        r"P"
        r"(?:(\d+)D)?"
        r"(?:T"
        r"(?:(\d+)H)?"
        r"(?:(\d+)M)?"
        r"(?:(\d+)S)?"
        r")?"
    )

    match = re.fullmatch(
        pattern,
        duration
    )

    if not match:
        return np.nan

    days = int(match.group(1) or 0)
    hours = int(match.group(2) or 0)
    minutes = int(match.group(3) or 0)
    seconds = int(match.group(4) or 0)

    return (
        days * 24 * 60 * 60
        + hours * 60 * 60
        + minutes * 60
        + seconds
    )

def create_duration_features(df):
    """
    Convert video duration into numerical features.
    """

    df = df.copy()

    df["duration_seconds"] = (
        df["duration"].apply(parse_duration)
    )

    df["duration_minutes"] = (
        df["duration_seconds"] / 60
    )

    return df


# ============================================================
# PUBLISHING TIME FEATURES
# ============================================================

def create_time_features(df):
    """
    Extract calendar/time features from published_at.

    Missing dates are preserved instead of being fabricated.
    """

    df = df.copy()

    published = df["published_at"]

    df["is_missing_publish_time"] = (
        published.isna()
    ).astype(int)

    df["upload_year"] = published.dt.year
    df["upload_month"] = published.dt.month
    df["upload_day"] = published.dt.day
    df["upload_hour"] = published.dt.hour
    df["upload_weekday"] = published.dt.weekday

    df["is_weekend"] = (
        df["upload_weekday"].isin([5, 6])
    ).astype(int)

    # Restore missing values for rows where published_at
    # is unavailable.
    missing_mask = published.isna()

    time_columns = [
        "upload_year",
        "upload_month",
        "upload_day",
        "upload_hour",
        "upload_weekday"
    ]

    for column in time_columns:
        df.loc[missing_mask, column] = np.nan

    return df


# ============================================================
# FINAL MODEL FEATURE SELECTION
# ============================================================

def select_model_features(df):
    """
    Keep only information that can reasonably be available
    before a new video is published.

    Post-publication outcome variables such as views,
    likes and comments are intentionally excluded.
    """

    feature_columns = [
        # Identifiers/context
        "video_id",
        "channel_id",
        "channel_title",

        # Text
        "title",
        "description",
        "tags",
        "channel_description",

        # Topic
        "search_category",
        "search_keyword",
        "topic_group",
        "category_id",

        # Channel
        "subscriber_count",
        "video_count",
        "country",
        "channel_size_group",
        "log_subscriber_count",
        "log_video_count",
        "channel_description_length",
        "channel_description_word_count",

        # Title
        "title_length",
        "title_word_count",
        "title_exclamation_count",
        "title_question_count",
        "title_digit_count",
        "title_has_number",
        "title_has_question",
        "title_has_exclamation",
        "title_uppercase_ratio",

        # Description
        "description_length",
        "description_word_count",
        "description_hashtag_count",
        "description_hashtag_present",
        "description_url_count",
        "description_url_present",

        # Tags
        "tag_count",
        "tag_text_length",
        "tag_present",

        # Video
        "duration_seconds",
        "duration_minutes",

        # Timing
        "upload_year",
        "upload_month",
        "upload_day",
        "upload_hour",
        "upload_weekday",
        "is_weekend",
        "is_missing_publish_time",

        # Original metadata
        "published_at",
        "collected_at"
    ]

    existing_columns = [
        column
        for column in feature_columns
        if column in df.columns
    ]

    return df[existing_columns].copy()


# ============================================================
# VALIDATION
# ============================================================

def validate_features(df):
    """
    Check that the generated model features look correct.
    """

    print("\n" + "=" * 70)
    print("FEATURE VALIDATION")
    print("=" * 70)

    print(f"\nRows    : {len(df):,}")
    print(f"Columns : {df.shape[1]:,}")

    print("\nFeature columns:")
    for column in df.columns:
        print(f" - {column}")

    print("\nMissing values in important features:")

    important_columns = [
        "title",
        "description",
        "tags",
        "subscriber_count",
        "video_count",
        "duration_seconds",
        "upload_year",
        "upload_hour",
        "channel_size_group"
    ]

    for column in important_columns:
        if column in df.columns:
            missing = df[column].isna().sum()
            print(
                f" {column:35s}: "
                f"{missing:,}"
            )

    # Make sure outcome variables are NOT present.
    forbidden_columns = [
        "view_count",
        "like_count",
        "comment_count",
        "like_rate",
        "comment_rate",
        "engagement_rate",
        "subscriber_to_view_ratio",
        "video_age_days",
        "log_view_count",
        "log_like_count",
        "log_comment_count",
        "relative_performance",
        "baseline_views"
    ]

    forbidden_found = [
        column
        for column in forbidden_columns
        if column in df.columns
    ]

    print("\nLeakage check:")

    if forbidden_found:
        print("WARNING - forbidden columns found:")
        for column in forbidden_found:
            print(f" - {column}")
    else:
        print(
            "PASS - no post-publication outcome "
            "features found."
        )


# ============================================================
# SAVE
# ============================================================

def save_features(df, path=OUTPUT_PATH):
    """
    Save the final model feature dataset.
    """

    os.makedirs(
        os.path.dirname(path),
        exist_ok=True
    )

    df.to_csv(
        path,
        index=False
    )

    print("\n" + "=" * 70)
    print("FEATURE DATASET SAVED")
    print("=" * 70)

    print(f"Path    : {path}")
    print(f"Rows    : {len(df):,}")
    print(f"Columns : {df.shape[1]:,}")


# ============================================================
# MAIN
# ============================================================

def main():

    # 1. Load
    df = load_data()

    # 2. Prepare types
    df = prepare_data(df)

    # 3. Channel features
    df = create_channel_features(df)

    # 4. Title features
    df = create_title_features(df)

    # 5. Description features
    df = create_description_features(df)

    # 6. Tag features
    df = create_tag_features(df)

    # 7. Topic features
    df = create_topic_features(df)

    # 8. Duration features
    df = create_duration_features(df)

    # 9. Time features
    df = create_time_features(df)

    # 10. Select final model features
    model_features = select_model_features(df)

    # 11. Validate
    validate_features(model_features)

    # 12. Save
    save_features(model_features)

    print("\n" + "=" * 70)
    print("FEATURE ENGINEERING COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()