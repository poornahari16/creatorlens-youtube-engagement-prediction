import os
import joblib
import numpy as np
import pandas as pd

from scipy.sparse import hstack
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_FILE = "data/splits/train.csv"
VALIDATION_FILE = "data/splits/validation.csv"
TEST_FILE = "data/splits/test.csv"

MODEL_DIR = "models"

TARGET = "relative_performance"


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("ROBUST TEXT + STRUCTURED REGRESSION")
print("=" * 70)

train = pd.read_csv(TRAIN_FILE)
validation = pd.read_csv(VALIDATION_FILE)
test = pd.read_csv(TEST_FILE)

print(f"\nTrain rows:      {len(train):,}")
print(f"Validation rows: {len(validation):,}")
print(f"Test rows:       {len(test):,}")


# ============================================================
# COLUMNS TO EXCLUDE
# ============================================================

EXCLUDE_COLUMNS = [
    TARGET,

    "video_id",
    "channel_id",
    "channel_title",

    "title",
    "description",
    "tags",

    "published_at",
    "collected_at",

    "search_keyword",
    "search_category",

    # Leakage / outcome features
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


STRUCTURED_COLUMNS = [
    column
    for column in train.columns
    if column not in EXCLUDE_COLUMNS
]


# ============================================================
# IDENTIFY NUMERIC / CATEGORICAL FEATURES
# ============================================================

numeric_columns = []
categorical_columns = []

for column in STRUCTURED_COLUMNS:

    converted = pd.to_numeric(
        train[column],
        errors="coerce"
    )

    original_non_missing = train[column].notna().sum()
    converted_non_missing = converted.notna().sum()

    if (
        original_non_missing == 0
        or converted_non_missing / original_non_missing >= 0.95
    ):
        numeric_columns.append(column)
    else:
        categorical_columns.append(column)


print(f"\nStructured features: {len(STRUCTURED_COLUMNS)}")
print(f"Numeric features:    {len(numeric_columns)}")
print(f"Categorical features:{len(categorical_columns)}")


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
# TARGET
# ============================================================

y_train = train[TARGET].values
y_validation = validation[TARGET].values
y_test = test[TARGET].values


# ============================================================
# TEXT VECTORIZERS
# ============================================================

print("\nFitting TF-IDF vectorizers...")


title_word_vectorizer = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 2),
    max_features=10000,
    min_df=2,
    sublinear_tf=True
)


title_char_vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5),
    max_features=10000,
    min_df=2,
    sublinear_tf=True
)


description_vectorizer = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 2),
    max_features=15000,
    min_df=2,
    sublinear_tf=True
)


tags_vectorizer = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 2),
    max_features=5000,
    min_df=1,
    sublinear_tf=True
)


# ============================================================
# TITLE
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
# DESCRIPTION
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
# TAGS
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
# STRUCTURED PREPROCESSING
# ============================================================

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
    f"\nStructured matrix shape: {structured_train.shape}"
)


# ============================================================
# COMBINE FEATURES
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


print("\nFinal feature matrix:")
print(f"Train:      {X_train.shape}")
print(f"Validation: {X_validation.shape}")
print(f"Test:       {X_test.shape}")


# ============================================================
# ROBUST MODEL
# ============================================================

print("\nTraining robust SGDRegressor...")
print("Loss: Huber")


model = SGDRegressor(
    loss="huber",
    epsilon=0.1,
    penalty="l2",
    alpha=0.0001,
    max_iter=1000,
    tol=1e-4,
    learning_rate="adaptive",
    eta0=0.01,
    random_state=42,
    early_stopping=False
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


print("\n" + "=" * 70)
print("COMPARISON WITH CURRENT WINNER")
print("=" * 70)


print("\nCurrent Text + Structured Ridge:")
print(f"Test MAE: {OLD_TEST_MAE:.4f}")
print(f"Test R²:  {OLD_TEST_R2:.4f}")


print("\nRobust SGDRegressor:")
print(f"Test MAE: {new_test_mae:.4f}")
print(f"Test R²:  {new_test_r2:.4f}")


print("\nChange:")
print(
    f"MAE change: {new_test_mae - OLD_TEST_MAE:+.4f}"
)

print(
    f"R² change:  {new_test_r2 - OLD_TEST_R2:+.4f}"
)


if (
    new_test_mae < OLD_TEST_MAE
    and new_test_r2 > OLD_TEST_R2
):

    print("\nRESULT: CLEAR IMPROVEMENT")

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
    "robust_sgd_regressor.joblib"
)


preprocessor_path = os.path.join(
    MODEL_DIR,
    "robust_sgd_preprocessor.joblib"
)


vectorizers_path = os.path.join(
    MODEL_DIR,
    "robust_sgd_vectorizers.joblib"
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
print("ROBUST MODEL TRAINING COMPLETE")
print("=" * 70)