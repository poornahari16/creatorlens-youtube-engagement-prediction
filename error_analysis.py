import os
import joblib
import numpy as np
import pandas as pd

from scipy.sparse import hstack
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# PATHS
# ============================================================

VALIDATION_PATH = "data/splits/validation.csv"
TEST_PATH = "data/splits/test.csv"

MODEL_PATH = "models/text_structured_ridge.joblib"
PREPROCESSOR_PATH = "models/text_structured_preprocessor.joblib"
VECTORIZERS_PATH = "models/text_structured_vectorizers.joblib"

OUTPUT_DIR = "data/analysis"


# ============================================================
# FEATURE SETTINGS
# ============================================================

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
# LOAD MODEL AND DATA
# ============================================================

def load_resources():

    validation_df = pd.read_csv(
        VALIDATION_PATH
    )

    test_df = pd.read_csv(
        TEST_PATH
    )

    model = joblib.load(
        MODEL_PATH
    )

    preprocessor = joblib.load(
        PREPROCESSOR_PATH
    )

    vectorizers = joblib.load(
        VECTORIZERS_PATH
    )

    return (
        validation_df,
        test_df,
        model,
        preprocessor,
        vectorizers
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
# CREATE TEXT FEATURES
# ============================================================

def create_text_features(
    df,
    vectorizers
):

    (
        title,
        description,
        tags
    ) = prepare_text(df)

    X_title_word = (
        vectorizers["title_word"]
        .transform(title)
    )

    X_title_char = (
        vectorizers["title_char"]
        .transform(title)
    )

    X_description = (
        vectorizers["description"]
        .transform(description)
    )

    X_tags = (
        vectorizers["tags"]
        .transform(tags)
    )

    X_text = hstack([
        X_title_word,
        X_title_char,
        X_description,
        X_tags
    ]).tocsr()

    return X_text


# ============================================================
# CREATE COMPLETE FEATURES
# ============================================================

def create_features(
    df,
    preprocessor,
    vectorizers
):

    X_text = create_text_features(
        df,
        vectorizers
    )

    X_structured = (
        preprocessor.transform(df)
    )

    X = hstack([
        X_text,
        X_structured
    ]).tocsr()

    return X


# ============================================================
# CREATE PREDICTIONS
# ============================================================

def create_predictions(
    df,
    model,
    preprocessor,
    vectorizers
):

    X = create_features(
        df,
        preprocessor,
        vectorizers
    )

    predictions = model.predict(
        X
    )

    result = df.copy()

    result["predicted_relative_performance"] = (
        predictions
    )

    result["prediction_error"] = (
        result["predicted_relative_performance"]
        - result["relative_performance"]
    )

    result["absolute_error"] = (
        result["prediction_error"]
        .abs()
    )

    return result


# ============================================================
# OVERALL METRICS
# ============================================================

def print_metrics(
    df,
    dataset_name
):

    y_true = df[
        "relative_performance"
    ]

    y_pred = df[
        "predicted_relative_performance"
    ]

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred
        )
    )

    r2 = r2_score(
        y_true,
        y_pred
    )

    print("\n" + "=" * 70)
    print(
        f"{dataset_name} METRICS"
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


# ============================================================
# LARGEST ERRORS
# ============================================================

def print_largest_errors(
    df
):

    columns = [
        "video_id",
        "channel_title",
        "title",
        "topic_group",
        "channel_size_group",
        "relative_performance",
        "predicted_relative_performance",
        "prediction_error",
        "absolute_error"
    ]

    available_columns = [
        column
        for column in columns
        if column in df.columns
    ]

    print("\n" + "=" * 70)
    print("TOP 15 LARGEST ERRORS")
    print("=" * 70)

    print(
        df.sort_values(
            "absolute_error",
            ascending=False
        )[available_columns]
        .head(15)
        .to_string(index=False)
    )


# ============================================================
# BIGGEST OVERPREDICTIONS
# ============================================================

def print_overpredictions(
    df
):

    columns = [
        "video_id",
        "channel_title",
        "title",
        "topic_group",
        "relative_performance",
        "predicted_relative_performance",
        "prediction_error"
    ]

    available_columns = [
        column
        for column in columns
        if column in df.columns
    ]

    print("\n" + "=" * 70)
    print("BIGGEST OVERPREDICTIONS")
    print("=" * 70)

    print(
        df.sort_values(
            "prediction_error",
            ascending=False
        )[available_columns]
        .head(10)
        .to_string(index=False)
    )


# ============================================================
# BIGGEST UNDERPREDICTIONS
# ============================================================

def print_underpredictions(
    df
):

    columns = [
        "video_id",
        "channel_title",
        "title",
        "topic_group",
        "relative_performance",
        "predicted_relative_performance",
        "prediction_error"
    ]

    available_columns = [
        column
        for column in columns
        if column in df.columns
    ]

    print("\n" + "=" * 70)
    print("BIGGEST UNDERPREDICTIONS")
    print("=" * 70)

    print(
        df.sort_values(
            "prediction_error",
            ascending=True
        )[available_columns]
        .head(10)
        .to_string(index=False)
    )


# ============================================================
# ERROR BY CHANNEL SIZE
# ============================================================

def analyze_channel_size(
    df
):

    grouped = (
        df.groupby(
            "channel_size_group"
        )
        .agg(
            videos=(
                "relative_performance",
                "size"
            ),
            mae=(
                "absolute_error",
                "mean"
            ),
            median_error=(
                "absolute_error",
                "median"
            ),
            mean_target=(
                "relative_performance",
                "mean"
            )
        )
        .reset_index()
    )

    print("\n" + "=" * 70)
    print("ERROR BY CHANNEL SIZE")
    print("=" * 70)

    print(
        grouped.to_string(
            index=False
        )
    )

    return grouped


# ============================================================
# ERROR BY TOPIC
# ============================================================

def analyze_topic(
    df
):

    grouped = (
        df.groupby(
            "topic_group"
        )
        .agg(
            videos=(
                "relative_performance",
                "size"
            ),
            mae=(
                "absolute_error",
                "mean"
            ),
            mean_target=(
                "relative_performance",
                "mean"
            )
        )
        .sort_values(
            "mae",
            ascending=False
        )
        .reset_index()
    )

    print("\n" + "=" * 70)
    print("TOPIC ERROR ANALYSIS")
    print("=" * 70)

    print(
        grouped.head(20).to_string(
            index=False
        )
    )

    return grouped


# ============================================================
# ERROR BY TARGET RANGE
# ============================================================

def analyze_target_range(
    df
):

    result = df.copy()

    result["target_range"] = pd.cut(
        result["relative_performance"],
        bins=[
            -np.inf,
            -2,
            -1,
            0,
            1,
            2,
            np.inf
        ],
        labels=[
            "< -2",
            "-2 to -1",
            "-1 to 0",
            "0 to 1",
            "1 to 2",
            "> 2"
        ]
    )

    grouped = (
        result.groupby(
            "target_range",
            observed=True
        )
        .agg(
            videos=(
                "relative_performance",
                "size"
            ),
            mae=(
                "absolute_error",
                "mean"
            ),
            mean_prediction=(
                "predicted_relative_performance",
                "mean"
            ),
            mean_actual=(
                "relative_performance",
                "mean"
            )
        )
        .reset_index()
    )

    print("\n" + "=" * 70)
    print("ERROR BY TARGET RANGE")
    print("=" * 70)

    print(
        grouped.to_string(
            index=False
        )
    )

    return grouped


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    validation_df,
    test_df,
    channel_analysis,
    topic_analysis,
    target_analysis
):

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    validation_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "validation_predictions.csv"
        ),
        index=False
    )

    test_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "test_predictions.csv"
        ),
        index=False
    )

    channel_analysis.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "channel_size_error_analysis.csv"
        ),
        index=False
    )

    topic_analysis.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "topic_error_analysis.csv"
        ),
        index=False
    )

    target_analysis.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "target_range_error_analysis.csv"
        ),
        index=False
    )

    print("\n" + "=" * 70)
    print("ANALYSIS FILES SAVED")
    print("=" * 70)

    print(
        "data/analysis/validation_predictions.csv"
    )

    print(
        "data/analysis/test_predictions.csv"
    )

    print(
        "data/analysis/channel_size_error_analysis.csv"
    )

    print(
        "data/analysis/topic_error_analysis.csv"
    )

    print(
        "data/analysis/target_range_error_analysis.csv"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # 1. Load resources
    # --------------------------------------------------------

    (
        validation_df,
        test_df,
        model,
        preprocessor,
        vectorizers
    ) = load_resources()

    print("=" * 70)
    print("ERROR ANALYSIS")
    print("=" * 70)

    print(
        f"Validation rows : {len(validation_df):,}"
    )

    print(
        f"Test rows       : {len(test_df):,}"
    )


    # --------------------------------------------------------
    # 2. Generate predictions
    # --------------------------------------------------------

    print("\nGenerating validation predictions...")

    validation_results = create_predictions(
        validation_df,
        model,
        preprocessor,
        vectorizers
    )

    print(
        "Validation predictions completed."
    )

    print("\nGenerating test predictions...")

    test_results = create_predictions(
        test_df,
        model,
        preprocessor,
        vectorizers
    )

    print(
        "Test predictions completed."
    )


    # --------------------------------------------------------
    # 3. Overall metrics
    # --------------------------------------------------------

    print_metrics(
        validation_results,
        "VALIDATION"
    )

    print_metrics(
        test_results,
        "TEST"
    )


    # --------------------------------------------------------
    # 4. Largest errors
    # --------------------------------------------------------

    print_largest_errors(
        test_results
    )


    # --------------------------------------------------------
    # 5. Overpredictions
    # --------------------------------------------------------

    print_overpredictions(
        test_results
    )


    # --------------------------------------------------------
    # 6. Underpredictions
    # --------------------------------------------------------

    print_underpredictions(
        test_results
    )


    # --------------------------------------------------------
    # 7. Channel-size analysis
    # --------------------------------------------------------

    channel_analysis = analyze_channel_size(
        test_results
    )


    # --------------------------------------------------------
    # 8. Topic analysis
    # --------------------------------------------------------

    topic_analysis = analyze_topic(
        test_results
    )


    # --------------------------------------------------------
    # 9. Target-range analysis
    # --------------------------------------------------------

    target_analysis = analyze_target_range(
        test_results
    )


    # --------------------------------------------------------
    # 10. Save
    # --------------------------------------------------------

    save_results(
        validation_results,
        test_results,
        channel_analysis,
        topic_analysis,
        target_analysis
    )


    print("\n" + "=" * 70)
    print("ERROR ANALYSIS COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()