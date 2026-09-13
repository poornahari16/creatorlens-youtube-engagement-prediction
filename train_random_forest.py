import os
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
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

    train_df = pd.read_csv(
        TRAIN_PATH
    )

    validation_df = pd.read_csv(
        VALIDATION_PATH
    )

    test_df = pd.read_csv(
        TEST_PATH
    )

    print("=" * 70)
    print("DATA LOADED")
    print("=" * 70)

    print(
        f"Train rows      : {len(train_df):,}"
    )

    print(
        f"Validation rows : {len(validation_df):,}"
    )

    print(
        f"Test rows       : {len(test_df):,}"
    )

    return (
        train_df,
        validation_df,
        test_df
    )


# ============================================================
# PREPARE FEATURES
# ============================================================

def prepare_features(
    train_df,
    validation_df,
    test_df
):

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                "passthrough",
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

    y_train = train_df[
        TARGET_COLUMN
    ].values

    y_validation = validation_df[
        TARGET_COLUMN
    ].values

    y_test = test_df[
        TARGET_COLUMN
    ].values

    return (
        y_train,
        y_validation,
        y_test
    )


# ============================================================
# TRAIN RANDOM FOREST
# ============================================================

def train_model(
    X_train,
    y_train
):

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=18,
        min_samples_leaf=5,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1
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
# SAVE MODEL
# ============================================================

def save_model(
    model,
    preprocessor
):

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    model_path = os.path.join(
        MODEL_DIR,
        "random_forest_structured.joblib"
    )

    preprocessor_path = os.path.join(
        MODEL_DIR,
        "random_forest_preprocessor.joblib"
    )

    joblib.dump(
        model,
        model_path
    )

    joblib.dump(
        preprocessor,
        preprocessor_path
    )

    print("\n" + "=" * 70)
    print("MODEL SAVED")
    print("=" * 70)

    print(
        f"Model        : {model_path}"
    )

    print(
        f"Preprocessor : {preprocessor_path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # 1. Load data
    # --------------------------------------------------------

    (
        train_df,
        validation_df,
        test_df
    ) = load_data()


    # --------------------------------------------------------
    # 2. Prepare structured features
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("BUILDING STRUCTURED FEATURES")
    print("=" * 70)

    (
        preprocessor,
        X_train,
        X_validation,
        X_test
    ) = prepare_features(
        train_df,
        validation_df,
        test_df
    )

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


    # --------------------------------------------------------
    # 3. Prepare target
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

    print("\n" + "=" * 70)
    print("TARGET")
    print("=" * 70)

    print(
        f"Train target rows      : "
        f"{len(y_train):,}"
    )

    print(
        f"Validation target rows : "
        f"{len(y_validation):,}"
    )

    print(
        f"Test target rows       : "
        f"{len(y_test):,}"
    )


    # --------------------------------------------------------
    # 4. Train model
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TRAINING RANDOM FOREST")
    print("=" * 70)

    model = train_model(
        X_train,
        y_train
    )

    print(
        "Training completed."
    )


    # --------------------------------------------------------
    # 5. Evaluate
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
    # 6. Save
    # --------------------------------------------------------

    save_model(
        model,
        preprocessor
    )


    # --------------------------------------------------------
    # 7. Final summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("RANDOM FOREST COMPLETED")
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