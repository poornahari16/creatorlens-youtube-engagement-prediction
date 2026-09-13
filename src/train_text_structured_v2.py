import os
import joblib
import numpy as np
import pandas as pd

from scipy.sparse import hstack
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_FILE = "data/splits/train_v2.csv"
VALIDATION_FILE = "data/splits/validation_v2.csv"
TEST_FILE = "data/splits/test_v2.csv"

MODEL_DIR = "models"

TARGET = "relative_performance"


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("TEXT + STRUCTURED RIDGE — IMPROVED FEATURES")
print("=" * 70)

train = pd.read_csv(TRAIN_FILE)
validation = pd.read_csv(VALIDATION_FILE)
test = pd.read_csv(TEST_FILE)

print(f"\nTrain rows:      {len(train):,}")
print(f"Validation rows: {len(validation):,}")
print(f"Test rows:       {len(test):,}")


# ============================================================
# TEXT COLUMNS
# ============================================================

TEXT_COLUMNS = [
    "title",
    "description",
    "tags"
]


# ============================================================
# NEW CONTENT FEATURES
# ============================================================

NEW_CONTENT_FEATURES = [
    "title_description_overlap_count",
    "title_description_overlap_ratio",
    "title_tags_overlap_count",
    "title_tags_overlap_ratio",
    "description_tags_overlap_count",
    "description_tags_overlap_ratio",
    "title_unique_word_ratio",
    "title_avg_word_length",
    "title_long_word_count",
    "description_unique_word_ratio",
    "description_avg_word_length",
    "description_long_word_count",
    "description_to_title_length_ratio",
    "description_to_title_word_ratio",
    "total_content_word_count",
    "total_unique_content_word_count"
]


# ============================================================
# COLUMNS THAT MUST NOT BE MODEL INPUTS
# ============================================================

EXCLUDE_COLUMNS = [
    TARGET,

    # IDs / tracking
    "video_id",
    "channel_id",
    "channel_title",

    # Raw text handled separately by TF-IDF
    "title",
    "description",
    "tags",

    # Dates handled only through engineered date features
    "published_at",
    "collected_at",

    # Collection-query context
    "search_keyword",
    "search_category",

    # Outcome / leakage columns
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
    "baseline_views"
]


# ============================================================
# STRUCTURED COLUMNS
# ============================================================

STRUCTURED_COLUMNS = [
    column
    for column in train.columns
    if column not in EXCLUDE_COLUMNS
]


print(f"\nTotal structured columns: {len(STRUCTURED_COLUMNS)}")
print(f"New content features:     {len(NEW_CONTENT_FEATURES)}")


print("\nNew content features being used:")
for feature in NEW_CONTENT_FEATURES:
    print(f"  - {feature}")


# ============================================================
# IDENTIFY NUMERIC AND CATEGORICAL FEATURES
# ============================================================
#
# IMPORTANT:
# We cannot rely only on pandas dtype here.
# Some columns can contain numeric-looking values mixed with
# text. We therefore check whether the column can actually
# be converted to numeric values.
#
# ============================================================

numeric_columns = []
categorical_columns = []

for column in STRUCTURED_COLUMNS:

    converted = pd.to_numeric(
        train[column],
        errors="coerce"
    )

    # If almost all non-missing values are numeric,
    # treat the feature as numeric.
    original_non_missing = train[column].notna().sum()
    converted_non_missing = converted.notna().sum()

    if (
        original_non_missing == 0
        or converted_non_missing / original_non_missing >= 0.95
    ):
        numeric_columns.append(column)
    else:
        categorical_columns.append(column)


print("\nNumeric structured features:")
for column in numeric_columns:
    print(f"  - {column}")

print("\nCategorical structured features:")
for column in categorical_columns:
    print(f"  - {column}")


# ============================================================
# CONVERT NUMERIC COLUMNS
# ============================================================

for column in numeric_columns:

    train[column] = pd.to_numeric(
        train[column],
        errors="coerce"
    )

    validation[column] = pd.to_numeric(
        validation[column],
        errors="coerce"
    )

    test[column] = pd.to_numeric(
        test[column],
        errors="coerce"
    )


# ============================================================
# PREPARE TEXT
# ============================================================

def combine_text(df):

    return (
        df["title"].fillna("").astype(str)
        + " "
        + df["description"].fillna("").astype(str)
        + " "
        + df["tags"].fillna("").astype(str)
    )


train_text = combine_text(train)
validation_text = combine_text(validation)
test_text = combine_text(test)


# ============================================================
# TARGET
# ============================================================

y_train = train[TARGET].values
y_validation = validation[TARGET].values
y_test = test[TARGET].values


# ============================================================
# TEXT VECTORIZERS
# ============================================================

print("\nFitting TF-IDF vectorizers...")


# Title word TF-IDF
title_word_vectorizer = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 2),
    max_features=10000,
    min_df=2,
    sublinear_tf=True
)


# Title character TF-IDF
title_char_vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5),
    max_features=10000,
    min_df=2,
    sublinear_tf=True
)


# Description TF-IDF
description_vectorizer = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 2),
    max_features=15000,
    min_df=2,
    sublinear_tf=True
)


# Tags TF-IDF
tags_vectorizer = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 2),
    max_features=5000,
    min_df=1,
    sublinear_tf=True
)


# ============================================================
# TITLE VECTORIZATION
# ============================================================

train_title = train["title"].fillna("").astype(str)
validation_title = validation["title"].fillna("").astype(str)
test_title = test["title"].fillna("").astype(str)


title_word_train = title_word_vectorizer.fit_transform(
    train_title
)

title_word_validation = title_word_vectorizer.transform(
    validation_title
)

title_word_test = title_word_vectorizer.transform(
    test_title
)


title_char_train = title_char_vectorizer.fit_transform(
    train_title
)

title_char_validation = title_char_vectorizer.transform(
    validation_title
)

title_char_test = title_char_vectorizer.transform(
    test_title
)


# ============================================================
# DESCRIPTION VECTORIZATION
# ============================================================

train_description = train["description"].fillna("").astype(str)
validation_description = validation["description"].fillna("").astype(str)
test_description = test["description"].fillna("").astype(str)


description_train = description_vectorizer.fit_transform(
    train_description
)

description_validation = description_vectorizer.transform(
    validation_description
)

description_test = description_vectorizer.transform(
    test_description
)


# ============================================================
# TAG VECTORIZATION
# ============================================================

train_tags = train["tags"].fillna("").astype(str)
validation_tags = validation["tags"].fillna("").astype(str)
test_tags = test["tags"].fillna("").astype(str)


tags_train = tags_vectorizer.fit_transform(
    train_tags
)

tags_validation = tags_vectorizer.transform(
    validation_tags
)

tags_test = tags_vectorizer.transform(
    test_tags
)


# ============================================================
# STRUCTURED FEATURE PREPROCESSING
# ============================================================

print("\nPreparing structured features...")


numeric_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    ),
    (
        "scaler",
        StandardScaler()
    )
])


categorical_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="most_frequent")
    ),
    (
        "onehot",
        OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=True
        )
    )
])


transformers = []


if numeric_columns:

    transformers.append(
        (
            "numeric",
            numeric_pipeline,
            numeric_columns
        )
    )


if categorical_columns:

    transformers.append(
        (
            "categorical",
            categorical_pipeline,
            categorical_columns
        )
    )


preprocessor = ColumnTransformer(
    transformers=transformers
)


structured_train = preprocessor.fit_transform(
    train[STRUCTURED_COLUMNS]
)

structured_validation = preprocessor.transform(
    validation[STRUCTURED_COLUMNS]
)

structured_test = preprocessor.transform(
    test[STRUCTURED_COLUMNS]
)


print(
    f"Structured matrix shape: {structured_train.shape}"
)


# ============================================================
# COMBINE TEXT + STRUCTURED FEATURES
# ============================================================

X_train = hstack([
    title_word_train,
    title_char_train,
    description_train,
    tags_train,
    structured_train
]).tocsr()


X_validation = hstack([
    title_word_validation,
    title_char_validation,
    description_validation,
    tags_validation,
    structured_validation
]).tocsr()


X_test = hstack([
    title_word_test,
    title_char_test,
    description_test,
    tags_test,
    structured_test
]).tocsr()


print("\nFinal feature matrix sizes:")
print(f"Train:      {X_train.shape}")
print(f"Validation: {X_validation.shape}")
print(f"Test:       {X_test.shape}")


# ============================================================
# TRAIN RIDGE
# ============================================================

print("\nTraining Ridge model...")


model = Ridge(
    alpha=10.0
)


model.fit(
    X_train,
    y_train
)


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(name, X, y):

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

    print(f"\n{name}")
    print("-" * 40)
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²:   {r2:.4f}")

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }


# ============================================================
# VALIDATION
# ============================================================

validation_results = evaluate_model(
    "Validation Results",
    X_validation,
    y_validation
)


# ============================================================
# TEST
# ============================================================

test_results = evaluate_model(
    "Test Results",
    X_test,
    y_test
)


# ============================================================
# COMPARE WITH CURRENT WINNER
# ============================================================

OLD_TEST_MAE = 0.8053
OLD_TEST_R2 = 0.4339


new_test_mae = test_results["MAE"]
new_test_r2 = test_results["R2"]


mae_change = new_test_mae - OLD_TEST_MAE
r2_change = new_test_r2 - OLD_TEST_R2


print("\n" + "=" * 70)
print("COMPARISON WITH CURRENT WINNER")
print("=" * 70)


print("\nCurrent Text + Structured Ridge:")
print(f"Test MAE: {OLD_TEST_MAE:.4f}")
print(f"Test R²:  {OLD_TEST_R2:.4f}")


print("\nImproved Text + Structured Ridge:")
print(f"Test MAE: {new_test_mae:.4f}")
print(f"Test R²:  {new_test_r2:.4f}")


print("\nChange:")
print(f"MAE change: {mae_change:+.4f}")
print(f"R² change:  {r2_change:+.4f}")


if (
    new_test_mae < OLD_TEST_MAE
    and new_test_r2 > OLD_TEST_R2
):

    print("\nRESULT: IMPROVEMENT")

elif new_test_mae < OLD_TEST_MAE:

    print("\nRESULT: MAE IMPROVED")

elif new_test_r2 > OLD_TEST_R2:

    print("\nRESULT: R² IMPROVED")

else:

    print("\nRESULT: NO IMPROVEMENT")


# ============================================================
# SAVE MODEL
# ============================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


model_path = os.path.join(
    MODEL_DIR,
    "text_structured_ridge_v2.joblib"
)


preprocessor_path = os.path.join(
    MODEL_DIR,
    "text_structured_preprocessor_v2.joblib"
)


vectorizers_path = os.path.join(
    MODEL_DIR,
    "text_structured_vectorizers_v2.joblib"
)


joblib.dump(
    model,
    model_path
)


joblib.dump(
    preprocessor,
    preprocessor_path
)


joblib.dump(
    {
        "title_word": title_word_vectorizer,
        "title_char": title_char_vectorizer,
        "description": description_vectorizer,
        "tags": tags_vectorizer
    },
    vectorizers_path
)


print("\nSaved:")
print(f"  {model_path}")
print(f"  {preprocessor_path}")
print(f"  {vectorizers_path}")


print("\n" + "=" * 70)
print("IMPROVED MODEL TRAINING COMPLETE")
print("=" * 70)