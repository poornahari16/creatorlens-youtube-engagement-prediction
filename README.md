# CreatorLens - YouTube Engagement Prediction

**CreatorLens** is an end-to-end machine learning application that estimates how a planned YouTube video may perform compared with the expected performance of similar videos.

Instead of trying to predict an exact number of views, CreatorLens uses a **relative performance approach**. It combines YouTube channel information, video metadata, title, description, tags, topic, duration, and upload timing to estimate whether a video is likely to perform **below, around, or above its usual baseline**.

The project also includes a Streamlit application where creators can enter their YouTube channel and planned video details, receive a performance prediction, and generate AI-assisted title and description suggestions.

---

## 🚀 Live Application

**Streamlit App:**
https://creatorlens-youtube-engagement-prediction.streamlit.app/ 

**GitHub Repository:**
https://github.com/poornahari16/creatorlens-youtube-engagement-prediction

---

## 📌 Project Overview

YouTube videos from different channels cannot always be compared directly using raw views.

For example:

* 10,000 views may be unusually high for a small channel.
* 10,000 views may be unusually low for a large channel.

CreatorLens therefore asks:

> **"How is this video likely to perform compared with its relevant baseline?"**

rather than:

> "How many views will this video get?"

The current system produces:

* Relative performance prediction
* Performance category
* Performance score
* Channel context
* AI-assisted title suggestions
* AI-assisted description suggestions

---

# 🎯 Objective

The main objective of CreatorLens is to build a practical ML-based decision-support system for YouTube creators.

The system is designed to help answer questions such as:

* Is my planned video likely to perform below, around, or above the expected baseline?
* How does my channel size affect the prediction?
* Does the planned title and description resemble patterns found in the training data?
* Can AI suggest alternative titles or descriptions?
* How does an AI-generated version compare with the original version?

The model is **not intended to guarantee views or predict exact future performance**.

---

# 🧠 How It Works

```text
                    YouTube Channel URL
                            │
                            ▼
                    YouTube Data API
                            │
                            ▼
                 Public Channel Information
                            │
                            │
        ┌───────────────────┴───────────────────┐
        │                                       │
        ▼                                       ▼
 Creator Input                            Channel Context
        │                                       │
        ├── Title                               ├── Subscribers
        ├── Description                         ├── Video Count
        ├── Tags                                ├── Channel Description
        ├── Topic                               └── Channel Size
        ├── Duration
        ├── Video Format
        └── Planned Upload Time
                            │
                            ▼
                    Feature Engineering
                            │
                            ▼
                     ML Model Pipeline
                            │
                            ▼
                Relative Performance Value
                            │
                            ▼
                 Performance Interpretation
                            │
             ┌──────────────┼──────────────┐
             ▼              ▼              ▼
        Below Usual     Around Usual    Above Usual
                            │
                            ▼
                     Performance Score
                            │
                            ▼
                 AI Content Suggestions
```

---

# ✨ Key Features

## 1. Relative Performance Prediction

CreatorLens predicts how a video may perform relative to a relevant historical baseline.

The prediction is converted into:

```text
Below Usual
Around Usual
Above Usual
```

---

## 2. YouTube Channel Integration

Users can enter a public YouTube channel URL or handle.

Examples:

```text
https://www.youtube.com/@channel
```

or

```text
https://www.youtube.com/channel/UC...
```

The application uses the YouTube Data API to retrieve available public channel information.

---

## 3. Video Content Analysis

The model considers:

* Video title
* Description
* Tags
* Topic
* Channel information
* Channel size
* Video duration
* Upload timing

---

## 4. AI-Assisted Content Suggestions

CreatorLens can generate alternative:

* Video titles
* Video descriptions

using an AI API through **OpenRouter**.

The AI suggestions are intended to help creators explore alternative content wording.

The ML model can then be used to evaluate the candidate content.

> AI suggestions are not presented as guaranteed improvements. A higher model score only means that the model predicts relatively stronger performance.

---

## 5. Performance Score

The application converts the model's relative prediction into a simple **0–100 performance score** for easier interpretation.

The score is a presentation metric.

It is **not a probability** and does not mean that a video has a specific percentage chance of succeeding.

---

# 📊 Dataset

The current dataset contains:

| Metric                        |     Value |
| ----------------------------- | --------: |
| Videos                        | **8,325** |
| Unique channels               | **5,364** |
| Categories                    |    **27** |
| Category-keyword combinations |   **270** |

The dataset contains information such as:

* Video metadata
* Channel metadata
* Titles
* Descriptions
* Tags
* Publication timestamps
* View counts
* Like counts
* Comment counts
* Search category
* Search keyword

---

# 🧹 Data Quality

The preprocessing pipeline performs checks for:

* Duplicate rows
* Duplicate video IDs
* Missing values
* Invalid numerical values
* Publication timestamps
* Collection timestamps
* Category coverage

Current dataset checks:

```text
Rows                         8,325
Duplicate rows                   0
Duplicate video IDs              0
Negative numerical values        0
Valid publication timestamps 8,325
Valid collection timestamps   8,325
```

---

# ⚙️ Machine Learning Pipeline

```text
Raw Dataset
     │
     ▼
Data Preprocessing
     │
     ▼
Feature Engineering
     │
     ▼
Target Engineering
     │
     ▼
Chronological Split
     │
     ├───────────────┐
     ▼               ▼
   Train        Validation
     │
     ▼
Model Experiments
     │
     ▼
Error Analysis
     │
     ▼
Selected Model
     │
     ▼
Saved Model Artifacts
     │
     ▼
Prediction Pipeline
     │
     ▼
Streamlit Application
```

---

# 🎯 Relative Performance Target

The project does not directly train the model to predict raw views.

Instead, the target is:

```text
Relative Performance =
log(1 + Actual Views)
-
log(1 + Baseline Views)
```

### Interpretation

```text
Positive value
    → Better than baseline

Value near zero
    → Around baseline

Negative value
    → Below baseline
```

Using logarithms reduces the influence of extremely large view counts.

---

# 📐 Baseline Engineering

The baseline is calculated using historical information available **before the video's publication date**.

This prevents future videos from being used to calculate the baseline for earlier videos.

## Baseline hierarchy

CreatorLens uses the following hierarchy:

```text
1. Channel historical baseline
             ↓
2. Topic + Channel Size baseline
             ↓
3. Category + Channel Size baseline
             ↓
4. Channel Size baseline
```

The system requires:

```text
Channel history:
3 or more previous videos

Group baseline:
10 or more previous observations
```

This hierarchy is necessary because many channels in the dataset have limited historical videos.

---

# 📊 Target Engineering Results

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

Therefore:

```text
Usable target coverage ≈ 99.34%
```

---

# 🧩 Feature Engineering

CreatorLens uses both **text features** and **structured features**.

## Channel Features

```text
subscriber_count
video_count
log_subscriber_count
log_video_count
channel_description_length
channel_description_word_count
channel_size_group
```

## Title Features

```text
title_length
title_word_count
title_exclamation_count
title_question_count
title_digit_count
title_has_number
title_has_question
title_has_exclamation
title_uppercase_ratio
```

## Description Features

```text
description_length
description_word_count
description_hashtag_count
description_hashtag_present
description_url_count
description_url_present
```

## Tag Features

```text
tag_count
tag_text_length
tag_present
```

## Video Features

```text
duration_seconds
duration_minutes
```

## Timing Features

```text
upload_year
upload_month
upload_day
upload_hour
upload_weekday
is_weekend
is_missing_publish_time
```

## Categorical Features

```text
topic_group
channel_size_group
```

---

# 📝 Text Representation

The selected model uses TF-IDF representations for textual information.

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
analyzer = char
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

# 📅 Chronological Train / Validation / Test Split

A chronological split is used instead of a random split.

This better represents the real-world situation where a model learns from past videos and predicts future videos.

```text
Train:       5,789 videos
Validation:  1,240 videos
Test:        1,241 videos
```

### Time periods

```text
Train
2005-11-05 → 2022-02-01

Validation
2022-02-02 → 2024-01-11

Test
2024-01-12 → 2026-08-31
```

The split was checked for:

* Correct chronological ordering
* Train/validation ID overlap
* Validation/test ID overlap
* Train/test ID overlap

All ID overlap checks returned zero.

---

# 🤖 Model Experiments

Several models were evaluated.

## Dummy Median

```text
Test MAE: 1.1222
```

This provides a simple baseline for comparison.

---

## Ridge Baseline

```text
Validation MAE:  0.7500
Validation RMSE: 0.9802
Validation R²:   0.4258

Test MAE:        0.8220
Test RMSE:       1.0464
Test R²:         0.4134
```

---

## Random Forest

Configuration:

```text
n_estimators = 300
max_depth = 18
min_samples_leaf = 5
max_features = sqrt
```

Results:

```text
Validation MAE:  0.8554
Validation RMSE: 1.1033
Validation R²:   0.2725

Test MAE:        0.9298
Test RMSE:       1.1656
Test R²:         0.2722
```

---

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
Validation MAE:  0.7106
Validation RMSE: 0.9233
Validation R²:   0.4905

Test MAE:        0.8195
Test RMSE:       1.0299
Test R²:         0.4318
```

---

## Text + Structured Ridge

Results:

```text
Validation MAE:  0.7543
Validation RMSE: 0.9872
Validation R²:   0.4176

Test MAE:        0.8053
Test RMSE:       1.0280
Test R²:         0.4339
```

---

# 🏆 Model Comparison

| Model                       |   Test MAE |    Test R² |
| --------------------------- | ---------: | ---------: |
| **Text + Structured Ridge** | **0.8053** | **0.4339** |
| HistGradientBoosting        |     0.8195 |     0.4318 |
| Ridge Baseline              |     0.8220 |     0.4134 |
| Random Forest               |     0.9298 |     0.2722 |
| Dummy Median                |     1.1222 |          — |

The current selected model is **Text + Structured Ridge**.

It achieved the lowest test MAE among the evaluated models and the highest test R².

---

# 🏗️ Selected Model

The selected model combines:

### Text features

* Title word TF-IDF
* Title character TF-IDF
* Description TF-IDF
* Tags TF-IDF

### Structured features

* Channel information
* Title statistics
* Description statistics
* Tag statistics
* Duration
* Upload timing
* Channel size
* Topic

Model:

```text
Ridge Regression
alpha = 10
```

---

# 🔮 Prediction Pipeline

The application uses saved training artifacts during inference.

```text
Creator Input
     │
     ▼
Feature Construction
     │
     ▼
Saved Preprocessor
     │
     ├── Title Word TF-IDF
     ├── Title Character TF-IDF
     ├── Description TF-IDF
     ├── Tags TF-IDF
     └── Structured Features
             │
             ▼
      Feature Combination
             │
             ▼
       Saved Ridge Model
             │
             ▼
        Prediction
```

Saved artifacts:

```text
models/
├── text_structured_ridge.joblib
├── text_structured_preprocessor.joblib
└── text_structured_vectorizers.joblib
```

---

# ✅ Prediction Pipeline Validation

The prediction pipeline was separately tested against held-out test data.

Current validation result:

```text
Pipeline MAE: 0.7837
```

Full test-set model performance:

```text
Test MAE: 0.8053
Test R²:  0.4339
```

The inference pipeline was corrected to ensure that prediction-time feature construction matches the training pipeline.

This included correcting:

* Word-count calculations
* Hashtag counting
* Tag parsing
* Structured feature construction
* Feature ordering

---

# 📱 Streamlit Application

The project includes a Streamlit application for interacting with the trained model.

## User Flow

### Step 1 — Enter Channel

The user enters a public YouTube channel URL or handle.

### Step 2 — Load Channel

CreatorLens retrieves available public channel information.

### Step 3 — Enter Video Information

The user provides:

* Title
* Description
* Tags
* Category
* Topic
* Duration
* Video format
* Planned upload date
* Planned upload time

### Step 4 — Generate Prediction

The trained ML pipeline generates a relative performance prediction.

### Step 5 — View Results

The application displays:

```text
Performance Category
↓
Below Usual / Around Usual / Above Usual

Performance Score
↓
0–100
```

### Step 6 — AI Suggestions

The application can generate AI-assisted alternatives for the title and description.

---

# 🤖 AI Content Suggestions

CreatorLens uses **OpenRouter** for AI-assisted content generation.

The AI can generate:

* Alternative video titles
* Improved descriptions
* Content suggestions based on the provided video context

The AI layer is separate from the core ML prediction model.

```text
Creator Input
      │
      ├───────────────┐
      ▼               ▼
   ML Model       OpenRouter AI
      │               │
      ▼               ▼
Performance       Content
Prediction       Suggestions
      │               │
      └───────┬───────┘
              ▼
       Creator Decision
```

The AI output is not treated as ground truth.

---

# 📺 YouTube Data API

CreatorLens uses the **YouTube Data API v3** to retrieve public channel information.

The current integration supports:

* Channel IDs
* YouTube handles
* Public channel URLs

## Information Retrieved

```text
Channel ID
Channel title
Channel description
Country
Subscriber count
Video count
Public view count
Uploads playlist ID
Channel size group
```

---

# 📊 Channel Size Groups

|     Subscribers | Channel Size |
| --------------: | ------------ |
|         < 1,000 | 0-1K         |
|     1,000–9,999 | 1K-10K       |
|   10,000–99,999 | 10K-100K     |
| 100,000–999,999 | 100K-1M      |
|      1,000,000+ | 1M+          |

---

# 🔍 Error Analysis

The selected model was analyzed across different channel sizes and target ranges.

## Error by Channel Size

| Channel Size |    MAE |
| ------------ | -----: |
| 0-1K         | 0.9841 |
| 1K-10K       | 0.8973 |
| 10K-100K     | 0.7833 |
| 100K-1M      | 0.7225 |
| 1M+          | 0.9131 |

## Error by Target Range

| Target Range |    MAE |
| ------------ | -----: |
| < -2         | 1.7119 |
| -2 to -1     | 0.9172 |
| -1 to 0      | 0.5843 |
| 0 to 1       | 0.4869 |
| 1 to 2       | 0.8589 |
| > 2          | 1.3621 |

The model performs better around typical outcomes and has more difficulty with unusually high or unusually low-performing videos.

This is expected for a regression model when extreme outcomes are relatively difficult to infer from the available features.

---

# ⚠️ Limitations

## 1. Exact Views Are Not Predicted

CreatorLens predicts relative performance rather than exact future views.

---

## 2. Limited Channel History

The dataset contains:

```text
8,325 videos
5,364 unique channels
```

Many channels therefore have only a small number of historical videos.

A reliable channel-specific baseline is not always available.

---

## 3. Current Channel Statistics

Public channel statistics may represent the channel at collection time rather than exactly at the video's publication date.

---

## 4. Different Video Ages

The dataset contains lifetime view counts.

Older videos may therefore have had more time to accumulate views than newer videos.

---

## 5. Missing YouTube Analytics Data

The current public dataset does not contain several important performance signals, including:

* Impressions
* Click-through rate
* Watch time
* Audience retention
* Traffic sources
* Returning viewers
* Audience demographics
* Viewer satisfaction
* Thumbnail performance

These factors can strongly influence actual YouTube performance.

---

## 6. No Causal Claims

CreatorLens does not establish that changing a title or description will cause a specific increase in views.

For example, the system should not claim:

```text
"This title will increase views by 43%."
```

Instead:

```text
"This version has a higher predicted performance score."
```

---

# 🔮 Future Product Direction

A future version of CreatorLens can evolve from a simple prediction tool into a broader creator decision-support platform.

A possible workflow:

```text
Channel
   ↓
Historical Content
   ↓
Channel Performance Patterns
   ↓
Planned Video
   ↓
Title + Description + Topic
   ↓
ML Prediction
   ↓
AI Alternatives
   ↓
ML Re-evaluation
   ↓
Original vs Alternative Comparison
   ↓
Creator Decision
```

The long-term goal is to provide useful evidence-based suggestions without pretending that YouTube performance can be perfectly predicted.

---

# 📁 Project Structure

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

# 🛠️ Tech Stack

| Technology          | Purpose                    |
| ------------------- | -------------------------- |
| Python              | Core development           |
| Pandas              | Data processing            |
| NumPy               | Numerical operations       |
| Scikit-learn        | Machine learning           |
| SciPy               | Scientific computing       |
| Joblib              | Model serialization        |
| Streamlit           | Web application            |
| YouTube Data API v3 | Public channel information |
| OpenRouter          | AI content suggestions     |
| Git / GitHub        | Version control            |

---

# 💻 Installation

## 1. Clone the Repository

```bash
git clone https://github.com/poornahari16/creatorlens-youtube-engagement-prediction.git
cd creatorlens-youtube-engagement-prediction
```

## 2. Create Virtual Environment

Windows:

```powershell
python -m venv .venv
```

Activate:

```powershell
.venv\Scripts\activate
```

## 3. Install Dependencies

```powershell
pip install -r requirements.txt
```

---

# 🔐 Configuration

Create:

```text
.streamlit/secrets.toml
```

Add:

```toml
YOUTUBE_API_KEY = "YOUR_YOUTUBE_API_KEY"
OPENROUTER_API_KEY = "YOUR_OPENROUTER_API_KEY"
```

Do **not** commit this file to GitHub.

Make sure `.gitignore` contains:

```text
.streamlit/secrets.toml
```

For Streamlit Cloud, add the same secrets through the application's **Secrets** settings instead of committing them to the repository.

---

# ▶️ Run Locally

Start the Streamlit application:

```powershell
streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

---

# 🧪 Validate the Prediction Pipeline

Run:

```powershell
python src/validate_prediction_pipeline.py
```

This checks that:

* Saved model artifacts load correctly
* Feature construction works
* Prediction-time features match the training structure
* The trained model can generate predictions

---

# 📂 Important Generated Files

### Processed features

```text
data/features/youtube_model_features.csv
```

### Target dataset

```text
data/features/youtube_target_dataset.csv
```

### Training split

```text
data/splits/train.csv
```

### Validation split

```text
data/splits/validation.csv
```

### Test split

```text
data/splits/test.csv
```

### Selected model

```text
models/text_structured_ridge.joblib
```

### Prediction preprocessor

```text
models/text_structured_preprocessor.joblib
```

### Text vectorizers

```text
models/text_structured_vectorizers.joblib
```

---

# 🔒 Security

API keys should never be hard-coded into Python files or committed to GitHub.

Use:

```text
.streamlit/secrets.toml
```

locally and Streamlit Cloud Secrets for deployment.

If an API key is accidentally exposed, revoke it and generate a new key.

---

# 👨‍💻 Author

## Poorna Hari

B.Tech — Computing & Data Science

GitHub:
https://github.com/poornahari16

LinkedIn:
https://www.linkedin.com/in/poorna-hari/

---

# 📌 Disclaimer

CreatorLens is a machine learning project developed for experimentation, research, and creator decision support.

The predictions are estimates generated from historical data and available model features.

CreatorLens does **not** guarantee:

* Future views
* Likes
* Comments
* Watch time
* Audience retention
* Subscriber growth
* Revenue
* Viral performance

Actual YouTube performance depends on many factors that are not fully represented in the current dataset or model.

CreatorLens should therefore be treated as a **decision-support system, not an exact forecasting system**.

---

# 📈 Current Project Status

```text
Data Collection                         ✅
Data Preprocessing                     ✅
Feature Engineering                    ✅
Target Engineering                     ✅
Chronological Evaluation               ✅
Model Experiments                      ✅
Model Selection                        ✅
Error Analysis                         ✅
Saved Model Artifacts                  ✅
Prediction Pipeline                    ✅
Prediction Validation                  ✅
Streamlit Application                  ✅
YouTube Data API Integration            ✅
AI Content Suggestions                 ✅
OpenRouter Integration                 ✅

Original vs Improved Comparison        🔄
Advanced Channel Personalization       🔄
YouTube Analytics OAuth                🔜
Thumbnail Analysis                     🔜
Confidence Estimation                  🔜
```

---

## CreatorLens

**From raw YouTube data to an ML-powered creator decision-support application.**
