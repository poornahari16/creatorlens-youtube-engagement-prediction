# CreatorLens — YouTube Engagement Prediction

CreatorLens is an end-to-end machine learning project that estimates how a YouTube video may perform relative to comparable videos.

It combines video metadata, channel information, text features, topic information, and upload timing to produce a relative performance prediction rather than attempting to predict an exact number of views.

> **Current status:** The core ML pipeline, prediction pipeline, Streamlit application, and YouTube channel integration are working. AI-assisted content optimization and deeper personalization are planned next.

---

## Table of Contents

1. [Overview](#overview)
2. [Current Product Workflow](#current-product-workflow)
3. [Key Features](#key-features)
4. [Machine Learning Approach](#machine-learning-approach)
5. [Dataset](#dataset)
6. [Data Processing Pipeline](#data-processing-pipeline)
7. [Feature Engineering](#feature-engineering)
8. [Target Engineering](#target-engineering)
9. [Train Validation Test Strategy](#train-validation-test-strategy)
10. [Model Experiments](#model-experiments)
11. [Selected Model](#selected-model)
12. [Prediction Pipeline](#prediction-pipeline)
13. [Streamlit Application](#streamlit-application)
14. [YouTube Data API Integration](#youtube-data-api-integration)
15. [Project Structure](#project-structure)
16. [Installation](#installation)
17. [Configuration](#configuration)
18. [Running the Project](#running-the-project)
19. [Validation and Testing](#validation-and-testing)
20. [Error Analysis](#error-analysis)
21. [Limitations](#limitations)
22. [Planned Improvements](#planned-improvements)
23. [Development Roadmap](#development-roadmap)
24. [Tech Stack](#tech-stack)
25. [Repository](#repository)
26. [Author](#author)
27. [Disclaimer](#disclaimer)

---

# Overview

YouTube performance is affected by many factors, and the same number of views can mean very different things for channels of different sizes.

For example, 10,000 views may be excellent for a small channel but ordinary for a large channel.

CreatorLens therefore focuses on **relative performance**.

Instead of asking:

> "How many views will this video get?"

CreatorLens asks:

> "How is this video likely to perform compared with the relevant baseline?"

The current system analyzes:

- Video title
- Video description
- Tags
- Topic
- Channel information
- Channel size
- Video duration
- Planned upload timing

The result is presented as:

- **Below usual**
- **Around usual**
- **Above usual**

along with a presentation score from 0–100.

The score is a presentation metric and is **not a probability**.

---

# Current Product Workflow

```text
                YouTube Channel URL
                         |
                         v
                YouTube Data API
                         |
                         v
              Public Channel Information
                         |
                         |
Creator Input -----------+
     |
     +--> Title
     +--> Description
     +--> Tags
     +--> Topic
     +--> Duration
     +--> Planned Upload Time
                         |
                         v
              Feature Engineering
                         |
                         v
              Saved ML Model Pipeline
                         |
                         v
            Relative Performance Value
                         |
                         v
              Performance Interpretation
                         |
             +-----------+-----------+
             |           |           |
             v           v           v
          Below       Around       Above
           usual        usual       usual
                         |
                         v
                    Score /100
```

---

# Key Features

## Relative Performance Prediction

CreatorLens predicts a relative performance value rather than directly predicting lifetime views.

## YouTube Channel URL

A creator can enter a public YouTube channel URL. The application retrieves public channel information using the YouTube Data API.

## Video Content Analysis

The current model uses:

- Title
- Description
- Tags

as text inputs.

## Channel Context

The model uses:

- Subscriber count
- Video count
- Channel description
- Channel size group
- Topic information

## Upload Timing

The model uses:

- Upload year
- Upload month
- Upload day
- Upload hour
- Upload weekday
- Weekend indicator

## Performance Category

The raw model prediction is converted into:

```text
Relative Performance < -0.5
    -> Below usual

-0.5 <= Relative Performance <= 0.5
    -> Around usual

Relative Performance > 0.5
    -> Above usual
```

## Presentation Score

The current interface converts the model output into a 0–100 presentation score.

The score is intended to make the result easier to understand. It is not a probability of success.

---

# Machine Learning Approach

## Relative Performance Target

The core target is:

```text
Relative Performance =
log(1 + Actual Views) -
log(1 + Baseline Views)
```

Mathematically:

RelativePerformance = log(1 + ActualViews) - log(1 + BaselineViews)

Interpretation:

- Positive value → better than the comparison baseline
- Value near zero → around the baseline
- Negative value → below the baseline

Using logarithms reduces the effect of very large view counts and makes comparisons more manageable.

---

# Dataset

The current dataset contains:

- **8,325 videos**
- **5,364 unique channels**
- **27 categories**
- **270 category-keyword collection combinations**

Data quality checks included:

- Duplicate row checks
- Duplicate video ID checks
- Negative numeric value checks
- Missing publication timestamp checks
- Missing collection timestamp checks
- Category coverage checks

Current checks found:

- 0 duplicate rows
- 0 duplicate video IDs
- 0 negative values
- 8,325 valid publication timestamps
- 8,325 valid collection timestamps

---

# Data Processing Pipeline

```text
Raw YouTube Dataset
        |
        v
Preprocessing
        |
        v
Feature Engineering
        |
        v
Target Engineering
        |
        v
Chronological Split
        |
        +----------+-----------+
        |          |           |
        v          v           v
      Train     Validation    Test
        |
        v
Model Experiments
        |
        v
Selected Model
        |
        v
Saved Prediction Artifacts
        |
        v
Prediction Application
```

---

# Preprocessing

The preprocessing stage handles:

- Data type conversion
- Timestamp parsing
- Duplicate checks
- Invalid numeric values
- Missing values

The `published_at` and `collected_at` timestamps contain mixed ISO-8601 representations.

Element-wise parsing is used to correctly handle the different timestamp formats.

The resulting cleaned dataset contains 8,325 rows.

---

# Feature Engineering

## Channel Features

- `subscriber_count`
- `video_count`
- `log_subscriber_count`
- `log_video_count`
- `channel_description_length`
- `channel_description_word_count`
- `channel_size_group`

## Title Features

- `title_length`
- `title_word_count`
- `title_exclamation_count`
- `title_question_count`
- `title_digit_count`
- `title_has_number`
- `title_has_question`
- `title_has_exclamation`
- `title_uppercase_ratio`

## Description Features

- `description_length`
- `description_word_count`
- `description_hashtag_count`
- `description_hashtag_present`
- `description_url_count`
- `description_url_present`

## Tag Features

- `tag_count`
- `tag_text_length`
- `tag_present`

## Video Features

- `duration_seconds`
- `duration_minutes`

## Timing Features

- `upload_year`
- `upload_month`
- `upload_day`
- `upload_hour`
- `upload_weekday`
- `is_weekend`
- `is_missing_publish_time`

## Topic and Channel Size

Categorical inputs include:

- `topic_group`
- `channel_size_group`

---

# Target Engineering

The target-engineering stage creates the relative performance target.

The baseline is calculated using information from videos that occurred before the current video's publication date.

This prevents future videos from being used to calculate the baseline for an earlier video.

## Baseline Hierarchy

1. **Channel history** when sufficient prior videos are available
2. **Topic + channel size** when sufficient historical observations exist
3. **Category + channel size**
4. **Channel size** as the final fallback

Current minimum requirements:

```text
Channel history: 3 prior videos
Group baseline: 10 prior observations
```

## Target Dataset Results

```text
Total rows:              8,325
Usable target rows:      8,270
Unavailable target rows:    55
```

Baseline usage:

```text
Channel baseline:             0
Topic + size baseline:    1,951
Category + size baseline: 5,056
Size baseline:             1,263
Unavailable:                  55
```

The usable target represents approximately 99.34% of the dataset.

---

# Train Validation Test Strategy

The project uses a **chronological split** instead of a random split.

This better represents the real-world scenario where a model is trained on past videos and evaluated on future videos.

```text
Train:       5,789 videos
Validation:  1,240 videos
Test:        1,241 videos
```

Approximate periods:

```text
Train:
2005-11-05 → 2022-02-01

Validation:
2022-02-02 → 2024-01-11

Test:
2024-01-12 → 2026-08-31
```

The splits were checked for chronological ordering and ID overlap.

There is no overlap between:

- Train and validation
- Validation and test
- Train and test

---

# Model Experiments

## Dummy Median Baseline

Test MAE:

```text
1.1222
```

## Ridge Baseline

Results:

```text
Validation MAE: 0.7500
Validation RMSE: 0.9802
Validation R²: 0.4258

Test MAE: 0.8220
Test RMSE: 1.0464
Test R²: 0.4134
```

## Random Forest

Configuration:

```text
n_estimators = 300
max_depth = 18
min_samples_leaf = 5
max_features = "sqrt"
```

Results:

```text
Validation MAE: 0.8554
Validation RMSE: 1.1033
Validation R²: 0.2725

Test MAE: 0.9298
Test RMSE: 1.1656
Test R²: 0.2722
```

## HistGradientBoosting

Configuration:

```text
max_iter = 300
learning_rate = 0.05
max_leaf_nodes = 31
min_samples_leaf = 10
l2_regularization = 1.0
```

Results:

```text
Validation MAE: 0.7106
Validation RMSE: 0.9233
Validation R²: 0.4905

Test MAE: 0.8195
Test RMSE: 1.0299
Test R²: 0.4318
```

## Text + Structured Ridge

Results:

```text
Validation MAE: 0.7543
Validation RMSE: 0.9872
Validation R²: 0.4176

Test MAE: 0.8053
Test RMSE: 1.0280
Test R²: 0.4339
```

---

# Model Leaderboard

| Model | Test MAE | Test R² |
|---|---:|---:|
| **Text + Structured Ridge** | **0.8053** | **0.4339** |
| HistGradientBoosting | 0.8195 | 0.4318 |
| Ridge Baseline | 0.8220 | 0.4134 |
| Random Forest | 0.9298 | 0.2722 |
| Dummy Median | 1.1222 | — |

The **Text + Structured Ridge** model is currently the selected model because it achieved the best test MAE and best test R² among the evaluated models.

---

# Selected Model

## Text + Structured Ridge

The selected model combines:

### Text features

- Title word TF-IDF
- Title character TF-IDF
- Description TF-IDF
- Tags TF-IDF

### Structured features

- Channel features
- Title statistics
- Description statistics
- Tag statistics
- Duration
- Timing
- Channel size
- Topic

Final input:

```text
38,926 features
```

Model:

```text
Ridge Regression
alpha = 10.0
```

---

# Text Representation

## Title Word TF-IDF

```text
max_features = 10,000
ngram_range = (1, 2)
min_df = 2
max_df = 0.95
sublinear_tf = True
```

## Title Character TF-IDF

```text
analyzer = "char"
ngram_range = (3, 5)
min_df = 3
max_features = 10,000
sublinear_tf = True
```

## Description TF-IDF

```text
max_features = 15,000
ngram_range = (1, 2)
min_df = 2
max_df = 0.95
sublinear_tf = True
```

## Tags TF-IDF

```text
max_features = 5,000
ngram_range = (1, 2)
min_df = 2
sublinear_tf = True
```

---

# Prediction Pipeline

The deployed prediction pipeline uses the same saved artifacts used during training.

```text
models/
├── text_structured_ridge.joblib
├── text_structured_preprocessor.joblib
└── text_structured_vectorizers.joblib
```

Pipeline:

```text
Raw creator inputs
       |
       v
Structured feature creation
       |
       v
Saved structured preprocessor
       |
       v
Saved title word vectorizer
       |
       v
Saved title character vectorizer
       |
       v
Saved description vectorizer
       |
       v
Saved tags vectorizer
       |
       v
Combine features
       |
       v
Saved Ridge model
       |
       v
Prediction
```

Feature order:

```text
Title word TF-IDF
        +
Title character TF-IDF
        +
Description TF-IDF
        +
Tags TF-IDF
        +
Structured features
```

---

# Prediction Pipeline Validation

The deployed prediction pipeline was independently tested using held-out test rows.

Current validation:

```text
Pipeline MAE: 0.7837
```

Full test-set model performance:

```text
Test MAE: 0.8053
Test R²:  0.4339
```

An earlier inference implementation produced a 20-row MAE of 2.1108 because prediction-time feature calculations did not exactly match the training implementation.

The mismatches were corrected, including:

- Word-count calculation
- Hashtag counting
- Tag parsing
- Structured feature construction

The current prediction pipeline now reproduces the trained model behavior correctly.

---

# Streamlit Application

CreatorLens currently has a Streamlit-based interface.

The application allows a user to:

1. Enter a YouTube channel URL
2. Load public channel information
3. Enter a video title
4. Enter a description
5. Enter tags
6. Enter a topic
7. Enter video duration
8. Select planned upload time
9. Generate a prediction

The application displays:

```text
Performance
Below usual / Around usual / Above usual

Performance Score
0–100
```

The technical relative-performance value is available under technical details.

The application also explains that the prediction is not a guarantee of future views or engagement.

---

# YouTube Data API Integration

CreatorLens uses the **YouTube Data API** to retrieve public channel information.

The current integration supports YouTube channel IDs and handles.

Examples:

```text
https://www.youtube.com/@channel
```

```text
https://www.youtube.com/channel/UC...
```

## Retrieved Channel Information

The integration currently retrieves:

- Channel ID
- Channel title
- Channel description
- Country
- Subscriber count
- Video count
- Public channel view count
- Uploads playlist ID

---

# Channel Size Groups

| Subscribers | Channel Size |
|---:|---|
| < 1,000 | 0-1K |
| 1,000–9,999 | 1K-10K |
| 10,000–99,999 | 10K-100K |
| 100,000–999,999 | 100K-1M |
| 1,000,000+ | 1M+ |

---

# Project Structure

```text
CreatorLens/
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── .streamlit/
│   └── secrets.toml
│
├── data/
│   ├── raw/
│   │   └── youtube_dataset.csv
│   │
│   ├── features/
│   │   ├── youtube_model_features.csv
│   │   └── youtube_target_dataset.csv
│   │
│   ├── splits/
│   │   ├── train.csv
│   │   ├── validation.csv
│   │   └── test.csv
│   │
│   └── analysis/
│       ├── validation_predictions.csv
│       ├── test_predictions.csv
│       ├── channel_size_error_analysis.csv
│       ├── topic_error_analysis.csv
│       ├── target_range_error_analysis.csv
│       └── prediction_pipeline_validation_v2.csv
│
├── models/
│   ├── text_structured_ridge.joblib
│   ├── text_structured_preprocessor.joblib
│   ├── text_structured_vectorizers.joblib
│   ├── baseline_ridge.joblib
│   ├── baseline_preprocessor.joblib
│   ├── baseline_tfidf.joblib
│   ├── random_forest_structured.joblib
│   ├── random_forest_preprocessor.joblib
│   ├── gradient_boosting_structured.joblib
│   └── gradient_boosting_preprocessor.joblib
│
└── src/
    ├── preprocessing.py
    ├── feature_engineering.py
    ├── target_engineering.py
    ├── predict.py
    ├── youtube_api.py
    ├── train_baseline.py
    ├── train_random_forest.py
    ├── train_gradient_boosting.py
    ├── train_text_structured.py
    ├── error_analysis.py
    └── validate_prediction_pipeline.py
```

---

# Installation

## Clone

```bash
git clone https://github.com/poornahari16/creatorlens-youtube-engagement-prediction.git
cd creatorlens-youtube-engagement-prediction
```

## Create virtual environment

Windows:

```bash
python -m venv .venv
```

Activate:

```powershell
.venv\Scriptsctivate
```

## Install dependencies

```bash
pip install -r requirements.txt
```

If required:

```bash
pip install streamlit google-api-python-client
```

---

# Configuration

## YouTube API Key

Create:

```text
.streamlit/secrets.toml
```

Add:

```toml
YOUTUBE_API_KEY = "YOUR_API_KEY"
```

Never commit the secrets file.

`.gitignore` should contain:

```text
.streamlit/secrets.toml
```

---

# Running the Project

Start the application:

```powershell
streamlit run app.py
```

The local application normally runs at:

```text
http://localhost:8501
```

---

# Validation and Testing

Run the prediction pipeline validation:

```powershell
python src/validate_prediction_pipeline.py
```

The validation checks that the saved model artifacts and prediction-time feature construction work together correctly.

---

# Error Analysis

The selected Text + Structured Ridge model was analyzed by channel size and target range.

## By Channel Size

| Channel Size | Videos | MAE |
|---|---:|---:|
| 0-1K | 75 | 0.9841 |
| 1K-10K | 216 | 0.8973 |
| 10K-100K | 503 | 0.7833 |
| 100K-1M | 369 | 0.7225 |
| 1M+ | 78 | 0.9131 |

## By Target Range

| Target Range | Videos | MAE |
|---|---:|---:|
| < -2 | 94 | 1.7119 |
| -2 to -1 | 223 | 0.9172 |
| -1 to 0 | 312 | 0.5843 |
| 0 to 1 | 303 | 0.4869 |
| 1 to 2 | 232 | 0.8589 |
| > 2 | 77 | 1.3621 |

The model has more difficulty with extreme outcomes and tends to pull unusually high or low outcomes toward the average.

---

# Limitations

## Limited Historical Channel Data

The dataset contains 8,325 videos but 5,364 unique channels. Many channels have only one or a small number of videos.

Therefore, a channel-specific historical baseline is not always available.

## Current Channel Statistics

Subscriber count and other channel statistics may represent the channel at data-collection time rather than exactly at the video's publication time.

## Current View Counts

The target uses available lifetime view counts. Videos can therefore have different amounts of time to accumulate views.

## Missing YouTube Analytics Metrics

The current public dataset does not contain many metrics that strongly influence YouTube performance, including:

- Impressions
- Click-through rate
- Watch time
- Audience retention
- Viewer satisfaction
- Traffic sources
- Returning viewers
- Audience demographics
- Thumbnail performance

## No Causal Claims

A higher prediction for a title does not prove that changing the title will cause more views.

CreatorLens should use language such as:

> "Higher predicted performance"

rather than:

> "This title will increase views by X%."

---

# Current V2 Experiment

An additional feature-engineering experiment was attempted with features such as:

- Title-description overlap
- Title-tags overlap
- Description-tags overlap
- Unique word ratios
- Average word lengths
- Long-word counts
- Description-to-title ratios
- Total content word counts

The V2 training attempt was not completed successfully because text data was incorrectly treated as numeric structured input.

Therefore, V2 is **not part of the selected model**.

The current production model remains the verified Text + Structured Ridge model.

---

# Planned Improvements

## AI-Assisted Title Improvement

Generate alternative titles and evaluate them using the existing ML model.

```text
Original Title
      |
      v
Current Prediction
      |
      v
AI-generated Alternatives
      |
      v
ML Evaluation
      |
      v
Compare Predictions
```

## AI-Assisted Description Improvement

Potential improvements include:

- Clarity
- Structure
- Keyword usage
- Relevance
- Readability

The resulting description can then be evaluated by the ML model.

## Original vs Improved Comparison

A future interface could show:

```text
Original
59/100

Improved
68/100

Difference
+9 predicted score points
```

This should be presented as a model comparison, not a guaranteed increase in real-world views.

## Recent Upload Analysis

Use the channel's uploads playlist to retrieve recent public videos and build better channel-specific context.

## Personalized Baselines

Use enough recent channel history to estimate:

- Typical performance
- Topic-specific performance
- Content-type performance
- Upload-time patterns

when sufficient historical data exists.

## Connected Channel Mode

A future version may allow creators to connect their own YouTube account through OAuth and use authorized private YouTube Analytics information.

This is separate from the current public-channel workflow.

## Thumbnail Analysis

Thumbnail analysis is planned as a later feature.

Potential signals may include:

- Visual clarity
- Text density
- Composition
- Contrast
- Face presence
- Object prominence

## Confidence Estimation

A future version should provide:

```text
High
Medium
Low
```

based on factors such as:

- Available channel history
- Similarity to training data
- Model uncertainty
- Relevant historical data

## Better Explainability

The application should eventually explain which feature groups contribute most to a prediction using an appropriate model explanation method rather than inventing causal explanations.

---

# Development Roadmap

```text
Phase 1
Data Collection
        ↓
Phase 2
Preprocessing
        ↓
Phase 3
Feature Engineering
        ↓
Phase 4
Target Engineering
        ↓
Phase 5
Chronological Evaluation
        ↓
Phase 6
Model Experiments
        ↓
Phase 7
Prediction Application
        ↓
Phase 8
YouTube API Integration
        ↓
Phase 9
Prediction Result Improvements
        ↓
Phase 10
AI Title/Description Assistance
        ↓
Phase 11
Original vs Improved Comparison
        ↓
Phase 12
Recent Channel Analysis
        ↓
Phase 13
Advanced Personalization
        ↓
Phase 14
Deployment
```

---

# Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- SciPy
- Joblib
- Streamlit
- YouTube Data API v3

---

# Repository

GitHub:

https://github.com/poornahari16/creatorlens-youtube-engagement-prediction

---

# Author

## Poorna Hari

B.Tech Computing & Data Science

GitHub:

https://github.com/poornahari16

LinkedIn:

https://www.linkedin.com/in/poorna-hari/

---

# Disclaimer

CreatorLens is a machine learning project intended for experimentation, research, and creator decision support.

The predictions are estimates generated from the available training data and model features.

A higher predicted performance score does not guarantee:

- More views
- More likes
- More comments
- Higher watch time
- Higher audience retention
- Channel growth

Actual YouTube performance depends on many factors that are not fully represented in the current model.

CreatorLens should therefore be treated as a decision-support tool rather than a guarantee or exact forecasting system.

---

# Current Status

The following components are currently implemented and working:

```text
Data collection                         ✅
Data preprocessing                      ✅
Feature engineering                     ✅
Relative target engineering             ✅
Chronological train/validation/test     ✅
Model experimentation                   ✅
Model selection                         ✅
Error analysis                          ✅
Saved model artifacts                   ✅
Prediction pipeline                     ✅
Prediction pipeline validation          ✅
Streamlit application                   ✅
YouTube channel API integration         ✅

AI content optimization                 Planned
Recent upload personalization           Planned
YouTube Analytics integration           Planned
Thumbnail analysis                      Planned
Deployment                              Planned
```

CreatorLens currently has a functioning end-to-end ML prediction application. The next development focus is making the predictions more useful and personalized for creators.
