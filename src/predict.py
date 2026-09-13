import os
import re
import joblib
import numpy as np
import pandas as pd

from scipy.sparse import hstack


# ============================================================
# MODEL FILES
# ============================================================

MODEL_PATH = "models/text_structured_ridge.joblib"

PREPROCESSOR_PATH = (
    "models/text_structured_preprocessor.joblib"
)

VECTORIZERS_PATH = (
    "models/text_structured_vectorizers.joblib"
)


# ============================================================
# LOAD FROZEN MODEL ARTIFACTS
# ============================================================

print("Loading CreatorLens model...")

model = joblib.load(MODEL_PATH)

preprocessor = joblib.load(
    PREPROCESSOR_PATH
)

vectorizers = joblib.load(
    VECTORIZERS_PATH
)

print("Model loaded successfully.")


# ============================================================
# GET EXACT TRAINING FEATURE SCHEMA
# ============================================================

def get_preprocessor_columns():

    numeric_columns = []
    categorical_columns = []

    for name, transformer, columns in (
        preprocessor.transformers_
    ):

        if name == "numeric":
            numeric_columns = list(columns)

        elif name == "categorical":
            categorical_columns = list(columns)

    return numeric_columns, categorical_columns


NUMERIC_COLUMNS, CATEGORICAL_COLUMNS = (
    get_preprocessor_columns()
)


STRUCTURED_COLUMNS = (
    NUMERIC_COLUMNS
    + CATEGORICAL_COLUMNS
)


# ============================================================
# TEXT HELPERS
# ============================================================

def safe_text(value):

    if pd.isna(value):
        return ""

    return str(value)


def safe_ratio(a, b):

    if b == 0:
        return 0.0

    return a / b


# ============================================================
# TITLE FEATURES
# EXACTLY MATCH TRAINING FEATURE ENGINEERING
# ============================================================

def create_title_features(title):

    title = safe_text(title)

    # Training uses whitespace-based word splitting.
    words = title.split()

    alphabetic_characters = sum(
        char.isalpha()
        for char in title
    )

    uppercase_characters = sum(
        char.isupper()
        for char in title
    )

    return {

        "title_length":
            len(title),

        "title_word_count":
            len(words),

        "title_exclamation_count":
            title.count("!"),

        "title_question_count":
            title.count("?"),

        "title_digit_count":
            sum(
                char.isdigit()
                for char in title
            ),

        "title_has_number":
            int(
                any(
                    char.isdigit()
                    for char in title
                )
            ),

        "title_has_question":
            int("?" in title),

        "title_has_exclamation":
            int("!" in title),

        "title_uppercase_ratio":
            safe_ratio(
                uppercase_characters,
                alphabetic_characters
            )
    }


# ============================================================
# DESCRIPTION FEATURES
# EXACTLY MATCH TRAINING FEATURE ENGINEERING
# ============================================================

def create_description_features(description):

    description = safe_text(description)

    words = description.split()

    hashtags = re.findall(
        r"#\w+",
        description
    )

    urls = re.findall(
        r"https?://\S+|www\.\S+",
        description
    )

    return {

        "description_length":
            len(description),

        "description_word_count":
            len(words),

        "description_hashtag_count":
            len(hashtags),

        "description_hashtag_present":
            int(len(hashtags) > 0),

        "description_url_count":
            len(urls),

        "description_url_present":
            int(len(urls) > 0)
    }


# ============================================================
# TAG FEATURES
# EXACTLY MATCH TRAINING FEATURE ENGINEERING
# ============================================================

def create_tag_features(tags):

    tags = safe_text(tags)

    # Training data uses "|" as the separator when
    # the field contains multiple tags.
    if "|" in tags:

        tag_items = [
            tag.strip()
            for tag in tags.split("|")
            if tag.strip()
        ]

    else:

        # Match the original feature engineering:
        # a non-empty tag field without "|" is treated
        # as one tag.
        tag_items = (
            [tags.strip()]
            if tags.strip()
            else []
        )

    return {

        "tag_count":
            len(tag_items),

        "tag_text_length":
            len(tags),

        "tag_present":
            int(bool(tags.strip()))
    }


# ============================================================
# CREATE ALL STRUCTURED FEATURES
# EXACTLY MATCH TRAINING FEATURE ENGINEERING
# ============================================================

def create_structured_features(
    title,
    description,
    tags,
    channel_description="",
    category_id=0,
    subscriber_count=0,
    video_count=0,
    country="",
    channel_size_group="0-1K",
    topic_group="",
    duration_seconds=0,
    upload_year=2026,
    upload_month=1,
    upload_day=1,
    upload_hour=12,
    upload_weekday=0,
    is_weekend=0,
    is_missing_publish_time=0
):

    row = {}

    # --------------------------------------------------------
    # Channel
    # --------------------------------------------------------

    channel_description = safe_text(
        channel_description
    )

    country = safe_text(
        country
    )

    row["subscriber_count"] = (
        subscriber_count
    )

    row["video_count"] = (
        video_count
    )

    row["channel_description"] = (
        channel_description
    )

    row["country"] = (
        country
    )

    row["channel_size_group"] = (
        channel_size_group
    )

    row["log_subscriber_count"] = (
        np.log1p(
            max(float(subscriber_count), 0)
        )
    )

    row["log_video_count"] = (
        np.log1p(
            max(float(video_count), 0)
        )
    )

    row["channel_description_length"] = (
        len(channel_description)
    )

    # IMPORTANT:
    # Training uses whitespace splitting,
    # not regex tokenization.
    row["channel_description_word_count"] = (
        len(channel_description.split())
    )

    # --------------------------------------------------------
    # Topic
    # --------------------------------------------------------

    row["topic_group"] = safe_text(
        topic_group
    )

    row["category_id"] = (
        category_id
    )

    # --------------------------------------------------------
    # Title
    # --------------------------------------------------------

    row.update(
        create_title_features(title)
    )

    # --------------------------------------------------------
    # Description
    # --------------------------------------------------------

    row.update(
        create_description_features(
            description
        )
    )

    # --------------------------------------------------------
    # Tags
    # --------------------------------------------------------

    row.update(
        create_tag_features(tags)
    )

    # --------------------------------------------------------
    # Video
    # --------------------------------------------------------

    duration_seconds = (
        0
        if pd.isna(duration_seconds)
        else float(duration_seconds)
    )

    row["duration_seconds"] = (
        duration_seconds
    )

    row["duration_minutes"] = (
        duration_seconds / 60.0
    )

    # --------------------------------------------------------
    # Timing
    # --------------------------------------------------------

    row["upload_year"] = (
        upload_year
    )

    row["upload_month"] = (
        upload_month
    )

    row["upload_day"] = (
        upload_day
    )

    row["upload_hour"] = (
        upload_hour
    )

    row["upload_weekday"] = (
        upload_weekday
    )

    row["is_weekend"] = (
        is_weekend
    )

    row["is_missing_publish_time"] = (
        is_missing_publish_time
    )

    return row


# ============================================================
# PREPARE STRUCTURED DATAFRAME
# ============================================================

def prepare_structured_dataframe(row):

    df = pd.DataFrame([row])

    # Make sure every column expected by the
    # saved preprocessor exists.

    for column in STRUCTURED_COLUMNS:

        if column not in df.columns:

            if column in NUMERIC_COLUMNS:
                df[column] = np.nan

            else:
                df[column] = ""

    # Keep EXACT training order.

    df = df[
        STRUCTURED_COLUMNS
    ]

    # Convert numeric columns exactly as expected.

    for column in NUMERIC_COLUMNS:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    return df


# ============================================================
# PREDICTION
# ============================================================

def predict_performance(
    title,
    description="",
    tags="",
    channel_description="",
    category_id=0,
    subscriber_count=0,
    video_count=0,
    country="",
    channel_size_group="0-1K",
    topic_group="",
    duration_seconds=0,
    upload_year=2026,
    upload_month=1,
    upload_day=1,
    upload_hour=12,
    upload_weekday=0,
    is_weekend=0,
    is_missing_publish_time=0
):

    # --------------------------------------------------------
    # Normalize text exactly like training
    # --------------------------------------------------------

    title = safe_text(title)

    description = safe_text(
        description
    )

    tags = safe_text(
        tags
    )

    # --------------------------------------------------------
    # Structured features
    # --------------------------------------------------------

    structured_row = (
        create_structured_features(

            title=title,

            description=description,

            tags=tags,

            channel_description=channel_description,

            category_id=category_id,

            subscriber_count=subscriber_count,

            video_count=video_count,

            country=country,

            channel_size_group=channel_size_group,

            topic_group=topic_group,

            duration_seconds=duration_seconds,

            upload_year=upload_year,

            upload_month=upload_month,

            upload_day=upload_day,

            upload_hour=upload_hour,

            upload_weekday=upload_weekday,

            is_weekend=is_weekend,

            is_missing_publish_time=(
                is_missing_publish_time
            )
        )
    )

    structured_df = (
        prepare_structured_dataframe(
            structured_row
        )
    )

    structured_features = (
        preprocessor.transform(
            structured_df
        )
    )

    # --------------------------------------------------------
    # Title TF-IDF
    # EXACT SAME VECTORIZER AS TRAINING
    # --------------------------------------------------------

    title_series = pd.Series(
        [title]
    )

    title_word_features = (
        vectorizers["title_word"]
        .transform(
            title_series
        )
    )

    title_char_features = (
        vectorizers["title_char"]
        .transform(
            title_series
        )
    )

    # --------------------------------------------------------
    # Description TF-IDF
    # --------------------------------------------------------

    description_series = pd.Series(
        [description]
    )

    description_features = (
        vectorizers["description"]
        .transform(
            description_series
        )
    )

    # --------------------------------------------------------
    # Tags TF-IDF
    # --------------------------------------------------------

    tags_series = pd.Series(
        [tags]
    )

    tags_features = (
        vectorizers["tags"]
        .transform(
            tags_series
        )
    )

    # --------------------------------------------------------
    # Combine EXACT SAME ORDER AS TRAINING
    # --------------------------------------------------------

    X = hstack([

        title_word_features,

        title_char_features,

        description_features,

        tags_features,

        structured_features

    ]).tocsr()

    # --------------------------------------------------------
    # Sanity check
    # --------------------------------------------------------

    expected_features = (
        model.n_features_in_
    )

    actual_features = (
        X.shape[1]
    )

    if actual_features != expected_features:

        raise ValueError(
            f"Feature mismatch: "
            f"model expects "
            f"{expected_features} features, "
            f"but prediction pipeline created "
            f"{actual_features}."
        )

    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    prediction = (
        model.predict(X)[0]
    )

    return float(
        prediction
    )


# ============================================================
# INTERPRETATION
# ============================================================

def interpret_prediction(
    relative_performance
):

    if relative_performance < -0.5:

        category = "Below usual"

    elif relative_performance > 0.5:

        category = "Above usual"

    else:

        category = "Around usual"

    # Presentation score only.
    # It is NOT a probability.

    score = (
        50
        + (
            relative_performance
            * 20
        )
    )

    score = max(
        0,
        min(
            100,
            score
        )
    )

    return {

        "category":
            category,

        "score":
            round(
                score,
                1
            )
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 70)
    print(
        "CREATORLENS PREDICTION PIPELINE TEST"
    )
    print("=" * 70)

    print(
        f"\nSaved model expects "
        f"{model.n_features_in_} features."
    )

    print(
        f"Saved preprocessor expects "
        f"{len(STRUCTURED_COLUMNS)} "
        f"structured columns."
    )

    print(
        f"\nNumeric structured columns : "
        f"{len(NUMERIC_COLUMNS)}"
    )

    print(
        f"Categorical structured columns : "
        f"{len(CATEGORICAL_COLUMNS)}"
    )

    # --------------------------------------------------------
    # Example prediction
    # --------------------------------------------------------

    prediction = predict_performance(

        title=(
            "10 Python Projects That Will "
            "Improve Your Coding Skills"
        ),

        description=(
            "In this video, we explore 10 Python "
            "projects for beginners and intermediate "
            "developers. Learn practical Python "
            "programming through hands-on examples."
        ),

        tags=(
            "python,python projects,"
            "programming,coding,learn python"
        ),

        channel_description=(
            "Programming and technology tutorials"
        ),

        category_id=28,

        subscriber_count=50000,

        video_count=120,

        country="IN",

        channel_size_group="10K-100K",

        topic_group=(
            "Education | python tutorial"
        ),

        duration_seconds=600,

        upload_year=2026,

        upload_month=9,

        upload_day=13,

        upload_hour=18,

        upload_weekday=6,

        is_weekend=1,

        is_missing_publish_time=0
    )

    interpretation = (
        interpret_prediction(
            prediction
        )
    )

    print("\nPrediction:")

    print(
        f"Relative performance: "
        f"{prediction:.4f}"
    )

    print(
        f"Performance category: "
        f"{interpretation['category']}"
    )

    print(
        f"Performance score: "
        f"{interpretation['score']}/100"
    )

    print("\n" + "=" * 70)
    print(
        "PREDICTION PIPELINE TEST COMPLETE"
    )
    print("=" * 70)