import sys
import os

import pandas as pd
import numpy as np


# Allow importing predict.py
sys.path.insert(
    0,
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

from predict import (
    predict_performance,
    model
)


# ============================================================
# CONFIGURATION
# ============================================================

TEST_FILE = "data/splits/test.csv"

NUMBER_OF_SAMPLES = 20

TARGET = "relative_performance"


# ============================================================
# LOAD TEST DATA
# ============================================================

print("=" * 70)
print("CREATORLENS EXACT PIPELINE VALIDATION")
print("=" * 70)

test = pd.read_csv(
    TEST_FILE
)

print(
    f"\nTest rows available: "
    f"{len(test):,}"
)


# ============================================================
# SELECT SAMPLE
# ============================================================

sample = (
    test
    .dropna(
        subset=[
            "title",
            TARGET
        ]
    )
    .head(
        NUMBER_OF_SAMPLES
    )
    .copy()
)


print(
    f"Rows selected: "
    f"{len(sample)}"
)


# ============================================================
# GENERATE PREDICTIONS
# ============================================================

results = []


print(
    "\nGenerating predictions..."
)


for index, row in sample.iterrows():

    prediction = predict_performance(

        title=row["title"],

        description=row.get(
            "description",
            ""
        ),

        tags=row.get(
            "tags",
            ""
        ),

        channel_description=row.get(
            "channel_description",
            ""
        ),

        category_id=row.get(
            "category_id",
            0
        ),

        subscriber_count=row.get(
            "subscriber_count",
            0
        ),

        video_count=row.get(
            "video_count",
            0
        ),

        country=row.get(
            "country",
            ""
        ),

        channel_size_group=row.get(
            "channel_size_group",
            "0-1K"
        ),

        topic_group=row.get(
            "topic_group",
            ""
        ),

        duration_seconds=row.get(
            "duration_seconds",
            0
        ),

        upload_year=row.get(
            "upload_year",
            2026
        ),

        upload_month=row.get(
            "upload_month",
            1
        ),

        upload_day=row.get(
            "upload_day",
            1
        ),

        upload_hour=row.get(
            "upload_hour",
            12
        ),

        upload_weekday=row.get(
            "upload_weekday",
            0
        ),

        is_weekend=row.get(
            "is_weekend",
            0
        ),

        is_missing_publish_time=row.get(
            "is_missing_publish_time",
            0
        )
    )


    actual = float(
        row[TARGET]
    )


    results.append({

        "video_id":
            row.get(
                "video_id",
                ""
            ),

        "title":
            row["title"],

        "actual":
            actual,

        "pipeline_prediction":
            prediction,

        "absolute_error":
            abs(
                actual - prediction
            )
    })


# ============================================================
# RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)


mae = (
    results_df["absolute_error"]
    .mean()
)


print("\n" + "=" * 70)
print("VALIDATION RESULTS")
print("=" * 70)


print(
    f"\nPipeline MAE "
    f"({len(results_df)} samples): "
    f"{mae:.4f}"
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

for _, row in results_df.iterrows():

    title = str(
        row["title"]
    )

    if len(title) > 55:

        title = (
            title[:52]
            + "..."
        )


    print(
        f"\n{title}"
    )

    print(
        f"  Actual:     "
        f"{row['actual']:.4f}"
    )

    print(
        f"  Prediction: "
        f"{row['pipeline_prediction']:.4f}"
    )

    print(
        f"  Error:      "
        f"{row['absolute_error']:.4f}"
    )


# ============================================================
# SAVE
# ============================================================

output_dir = (
    "data/analysis"
)

os.makedirs(
    output_dir,
    exist_ok=True
)


output_file = (
    f"{output_dir}/"
    "prediction_pipeline_validation_v2.csv"
)


results_df.to_csv(
    output_file,
    index=False
)


# ============================================================
# FINAL STATUS
# ============================================================

print("\n" + "=" * 70)
print("PIPELINE VALIDATION COMPLETE")
print("=" * 70)

print(
    f"\nSaved:"
    f"\n  {output_file}"
)