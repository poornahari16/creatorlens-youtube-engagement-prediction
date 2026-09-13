import os
import joblib
import numpy as np
import pandas as pd

from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# PATHS
# ============================================================

TRAIN_PATH = "data/splits/train.csv"
VALIDATION_PATH = "data/splits/validation.csv"
TEST_PATH = "data/splits/test.csv"

MODEL_DIR = "models"


# ============================================================
# SETTINGS
# ============================================================

TARGET_COLUMN = "relative_performance"


NUMERIC_COLUMNS = [
    "subscriber_count",
    "video_count",
    "log_subscriber_count",
    "log_video_count",
    "channel_description_length",
    "channel_description_word_count",
    "title_length",
    "title_word_count",
    "title_exclamation_count",
    "title_question_count",
    "title_digit_count",
    "title_has_number",
    "title_has_question",
    "title_has_exclamation",
    "title_uppercase_ratio",
    "description_length",
    "description_word_count",
    "description_hashtag_count",
    "description_hashtag_present",
    "description_url_count",
    "description_url_present",
    "tag_count",
    "tag_text_length",
    "tag_present",
    "duration_seconds",
    "duration_minutes",
    "upload_year",
    "upload_month",
    "upload_day",
    "upload_hour",
    "upload_weekday",
    "is_weekend",
    "is_missing_publish_time"
]

CATEGORICAL_COLUMNS = [
    "channel_size_group",
    "topic_group"
]


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    train_df = pd.read_csv(TRAIN_PATH)
    validation_df = pd.read_csv(VALIDATION_PATH)
    test_df = pd.read_csv(TEST_PATH)

    print("=" * 70)
    print("DATA LOADED")
    print("=" * 70)

    print(f"Train rows      : {len(train_df):,}")
    print(f"Validation rows : {len(validation_df):,}")
    print(f"Test rows       : {len(test_df):,}")

    return (
        train_df,
        validation_df,
        test_df
    )


# ============================================================
# PREPARE TEXT
# ============================================================

def prepare_text(df):

    title = (
        df["title"]
        .fillna("")
        .astype(str)
    )

    description = (
        df["description"]
        .fillna("")
        .astype(str)
    )

    tags = (
        df["tags"]
        .fillna("")
        .astype(str)
    )

    return (
        title,
        description,
        tags
    )


# ============================================================
# BUILD TEXT FEATURES
# ============================================================

def build_text_features(
    train_df,
    validation_df,
    test_df
):

    (
        train_title,
        train_description,
        train_tags
    ) = prepare_text(train_df)

    (
        validation_title,
        validation_description,
        validation_tags
    ) = prepare_text(validation_df)

    (
        test_title,
        test_description,
        test_tags
    ) = prepare_text(test_df)


    # --------------------------------------------------------
    # Title word TF-IDF
    # --------------------------------------------------------

    title_word_vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )

    X_train_title_word = (
        title_word_vectorizer.fit_transform(
            train_title
        )
    )

    X_validation_title_word = (
        title_word_vectorizer.transform(
            validation_title
        )
    )

    X_test_title_word = (
        title_word_vectorizer.transform(
            test_title
        )
    )


    # --------------------------------------------------------
    # Title character TF-IDF
    # --------------------------------------------------------

    title_char_vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        min_df=3,
        max_features=10000,
        sublinear_tf=True
    )

    X_train_title_char = (
        title_char_vectorizer.fit_transform(
            train_title
        )
    )

    X_validation_title_char = (
        title_char_vectorizer.transform(
            validation_title
        )
    )

    X_test_title_char = (
        title_char_vectorizer.transform(
            test_title
        )
    )


    # --------------------------------------------------------
    # Description TF-IDF
    # --------------------------------------------------------

    description_vectorizer = TfidfVectorizer(
        max_features=15000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )

    X_train_description = (
        description_vectorizer.fit_transform(
            train_description
        )
    )

    X_validation_description = (
        description_vectorizer.transform(
            validation_description
        )
    )

    X_test_description = (
        description_vectorizer.transform(
            test_description
        )
    )


    # --------------------------------------------------------
    # Tags TF-IDF
    # --------------------------------------------------------

    tags_vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True
    )

    X_train_tags = (
        tags_vectorizer.fit_transform(
            train_tags
        )
    )

    X_validation_tags = (
        tags_vectorizer.transform(
            validation_tags
        )
    )

    X_test_tags = (
        tags_vectorizer.transform(
            test_tags
        )
    )


    # --------------------------------------------------------
    # Combine text features
    # --------------------------------------------------------

    X_train_text = hstack([
        X_train_title_word,
        X_train_title_char,
        X_train_description,
        X_train_tags
    ]).tocsr()

    X_validation_text = hstack([
        X_validation_title_word,
        X_validation_title_char,
        X_validation_description,
        X_validation_tags
    ]).tocsr()

    X_test_text = hstack([
        X_test_title_word,
        X_test_title_char,
        X_test_description,
        X_test_tags
    ]).tocsr()


    print("\n" + "=" * 70)
    print("TEXT FEATURES")
    print("=" * 70)

    print(
        f"Title word features  : "
        f"{X_train_title_word.shape[1]:,}"
    )

    print(
        f"Title char features  : "
        f"{X_train_title_char.shape[1]:,}"
    )

    print(
        f"Description features : "
        f"{X_train_description.shape[1]:,}"
    )

    print(
        f"Tags features        : "
        f"{X_train_tags.shape[1]:,}"
    )

    print(
        f"Combined text shape  : "
        f"{X_train_text.shape}"
    )

    vectorizers = {
        "title_word": title_word_vectorizer,
        "title_char": title_char_vectorizer,
        "description": description_vectorizer,
        "tags": tags_vectorizer
    }

    return (
        X_train_text,
        X_validation_text,
        X_test_text,
        vectorizers
    )


# ============================================================
# BUILD STRUCTURED FEATURES
# ============================================================

def build_structured_features(
    train_df,
    validation_df,
    test_df
):

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline([
                    (
                        "scaler",
                        StandardScaler()
                    )
                ]),
                NUMERIC_COLUMNS
            ),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                CATEGORICAL_COLUMNS
            )
        ],
        remainder="drop"
    )

    X_train = preprocessor.fit_transform(
        train_df
    )

    X_validation = preprocessor.transform(
        validation_df
    )

    X_test = preprocessor.transform(
        test_df
    )

    print("\n" + "=" * 70)
    print("STRUCTURED FEATURES")
    print("=" * 70)

    print(
        f"Train shape      : "
        f"{X_train.shape}"
    )

    print(
        f"Validation shape : "
        f"{X_validation.shape}"
    )

    print(
        f"Test shape       : "
        f"{X_test.shape}"
    )

    return (
        preprocessor,
        X_train,
        X_validation,
        X_test
    )


# ============================================================
# PREPARE TARGET
# ============================================================

def prepare_target(
    train_df,
    validation_df,
    test_df
):

    return (
        train_df[TARGET_COLUMN].values,
        validation_df[TARGET_COLUMN].values,
        test_df[TARGET_COLUMN].values
    )


# ============================================================
# TRAIN RIDGE
# ============================================================

def train_model(
    X_train,
    y_train
):

    model = Ridge(
        alpha=10.0
    )

    model.fit(
        X_train,
        y_train
    )

    return model


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(
    model,
    X,
    y,
    dataset_name
):

    predictions = model.predict(X)

    mae = mean_absolute_error(
        y,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y,
            predictions
        )
    )

    r2 = r2_score(
        y,
        predictions
    )

    print("\n" + "=" * 70)
    print(
        f"{dataset_name} RESULTS"
    )
    print("=" * 70)

    print(
        f"MAE  : {mae:.4f}"
    )

    print(
        f"RMSE : {rmse:.4f}"
    )

    print(
        f"R²   : {r2:.4f}"
    )

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2
    }


# ============================================================
# SAVE
# ============================================================

def save_artifacts(
    model,
    preprocessor,
    vectorizers
):

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    joblib.dump(
        model,
        os.path.join(
            MODEL_DIR,
            "text_structured_ridge.joblib"
        )
    )

    joblib.dump(
        preprocessor,
        os.path.join(
            MODEL_DIR,
            "text_structured_preprocessor.joblib"
        )
    )

    joblib.dump(
        vectorizers,
        os.path.join(
            MODEL_DIR,
            "text_structured_vectorizers.joblib"
        )
    )

    print("\n" + "=" * 70)
    print("MODEL SAVED")
    print("=" * 70)

    print(
        "Model        : "
        "models\\text_structured_ridge.joblib"
    )

    print(
        "Preprocessor : "
        "models\\text_structured_preprocessor.joblib"
    )

    print(
        "Vectorizers  : "
        "models\\text_structured_vectorizers.joblib"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # 1. Load
    # --------------------------------------------------------

    (
        train_df,
        validation_df,
        test_df
    ) = load_data()


    # --------------------------------------------------------
    # 2. Text features
    # --------------------------------------------------------

    (
        X_train_text,
        X_validation_text,
        X_test_text,
        vectorizers
    ) = build_text_features(
        train_df,
        validation_df,
        test_df
    )


    # --------------------------------------------------------
    # 3. Structured features
    # --------------------------------------------------------

    (
        preprocessor,
        X_train_structured,
        X_validation_structured,
        X_test_structured
    ) = build_structured_features(
        train_df,
        validation_df,
        test_df
    )


    # --------------------------------------------------------
    # 4. Combine
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("COMBINING TEXT + STRUCTURED FEATURES")
    print("=" * 70)

    X_train = hstack([
        X_train_text,
        X_train_structured
    ]).tocsr()

    X_validation = hstack([
        X_validation_text,
        X_validation_structured
    ]).tocsr()

    X_test = hstack([
        X_test_text,
        X_test_structured
    ]).tocsr()

    print(
        f"Combined train shape      : "
        f"{X_train.shape}"
    )

    print(
        f"Combined validation shape : "
        f"{X_validation.shape}"
    )

    print(
        f"Combined test shape       : "
        f"{X_test.shape}"
    )


    # --------------------------------------------------------
    # 5. Target
    # --------------------------------------------------------

    (
        y_train,
        y_validation,
        y_test
    ) = prepare_target(
        train_df,
        validation_df,
        test_df
    )


    # --------------------------------------------------------
    # 6. Train
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TRAINING TEXT + STRUCTURED RIDGE")
    print("=" * 70)

    model = train_model(
        X_train,
        y_train
    )

    print(
        "Training completed."
    )


    # --------------------------------------------------------
    # 7. Evaluate
    # --------------------------------------------------------

    validation_results = evaluate_model(
        model,
        X_validation,
        y_validation,
        "VALIDATION"
    )

    test_results = evaluate_model(
        model,
        X_test,
        y_test,
        "TEST"
    )


    # --------------------------------------------------------
    # 8. Save
    # --------------------------------------------------------

    save_artifacts(
        model,
        preprocessor,
        vectorizers
    )


    # --------------------------------------------------------
    # 9. Summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TEXT + STRUCTURED MODEL COMPLETED")
    print("=" * 70)

    print(
        f"Validation MAE : "
        f"{validation_results['mae']:.4f}"
    )

    print(
        f"Validation RMSE: "
        f"{validation_results['rmse']:.4f}"
    )

    print(
        f"Validation R²  : "
        f"{validation_results['r2']:.4f}"
    )

    print()

    print(
        f"Test MAE       : "
        f"{test_results['mae']:.4f}"
    )

    print(
        f"Test RMSE      : "
        f"{test_results['rmse']:.4f}"
    )

    print(
        f"Test R²        : "
        f"{test_results['r2']:.4f}"
    )


if __name__ == "__main__":
    main()