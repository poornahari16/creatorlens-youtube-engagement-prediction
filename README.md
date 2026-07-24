# CreatorLens - YouTube Engagement Prediction

## About the Project

CreatorLens is an end-to-end machine learning project that predicts the engagement level of YouTube videos using video and channel information collected through the YouTube Data API v3.

The goal of this project is to build a complete machine learning pipeline, starting from collecting raw data to training a prediction model and deploying it as a web application.

This project is being developed as a learning project to gain practical experience in data collection, data preprocessing, feature engineering, machine learning, and deployment.

---

## Problem Statement

Many content creators upload videos without knowing how well they might perform.

This project aims to build a machine learning model that can estimate the engagement of a YouTube video using available metadata such as views, likes, comments, channel statistics, publishing details, and other useful features.

---

## Project Workflow

1. Collect data using the YouTube Data API v3.
2. Store the collected data in CSV format.
3. Clean and preprocess the dataset.
4. Perform exploratory data analysis.
5. Create useful features.
6. Train multiple machine learning models.
7. Evaluate model performance.
8. Deploy the best model using Streamlit.

---

## Project Structure

```
CreatorLens/
│
├── data/
│   ├── external/
│   ├── raw/
│   ├── processed/
│   └── features/
│
├── docs/
│
├── notebooks/
│   ├── checkpoints/
│   ├── 01_API_Integration.ipynb
│   ├── 02_Data_Collector.ipynb
│   └── 03_EDA.ipynb
│
├── src/
│   ├── checkpoint.py
│   ├── collector.py
│   ├── config.py
│   ├── feature_engineering.py
│   ├── features.py
│   ├── keywords.py
│   └── preprocessing.py
│
├── .env
├── .env.example
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Dataset

The dataset is collected using the YouTube Data API v3.

It contains information such as:

- Video title
- Video ID
- Views
- Likes
- Comments
- Duration
- Published date
- Channel name
- Subscriber count
- Total channel views
- Total uploaded videos
- Search keyword
- Search category

The dataset will continue to grow as more videos are collected.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Jupyter Notebook
- YouTube Data API v3
- Streamlit (planned)

---

## Current Progress

- API integration completed
- Data collection pipeline completed
- Duplicate handling implemented
- Checkpoint system implemented
- Initial dataset collected
- Exploratory Data Analysis in progress

---

## Future Improvements

- Collect a larger dataset
- Build additional features
- Compare multiple machine learning models
- Hyperparameter tuning
- Deploy the final model using Streamlit
- Add model explainability

---

## How to Run

Clone the repository.

```bash
git clone https://github.com/your-username/creatorlens-youtube-engagement-prediction.git
```

Move into the project directory.

```bash
cd creatorlens-youtube-engagement-prediction
```

Install the required packages.

```bash
pip install -r requirements.txt
```

Create a `.env` file and add your YouTube API key.

```text
YOUTUBE_API_KEY=YOUR_API_KEY
```

Run the notebooks in order.

1. API Integration
2. Data Collection
3. Exploratory Data Analysis

---

## License

This project is developed for learning and educational purposes.