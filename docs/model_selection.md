# Model Selection

## Final Model

The final model selected for CreatorLens is the **Text + Structured Ridge Regression** model.

## Why this model was selected

The model combines:

- Title word TF-IDF features
- Title character TF-IDF features
- Description TF-IDF features
- Tags TF-IDF features
- Structured video and channel features

The model uses Ridge Regression with:

- `alpha = 10.0`

## Final Test Performance

| Metric | Result |
|---|---:|
| MAE | 0.8053 |
| RMSE | 1.0280 |
| R² | 0.4339 |

## Model Comparison

| Model | Test MAE | Test R² | Decision |
|---|---:|---:|---|
| Text + Structured Ridge | **0.8053** | **0.4339** | **Selected** |
| Text + Structured Ridge + Content Features | 0.8054 | 0.4343 | Not selected |
| HistGradientBoosting | 0.8195 | 0.4318 | Not selected |
| Baseline Ridge | 0.8220 | 0.4134 | Not selected |
| Random Forest | 0.9298 | 0.2722 | Not selected |
| Robust SGD / Huber | 0.8218 | 0.4131 | Not selected |

## Selection Reason

The original Text + Structured Ridge model achieved the lowest test MAE among the tested models.

The V2 content-feature experiment produced only a negligible R² increase while slightly worsening MAE, so it was not considered a meaningful improvement.

The robust Huber-based model performed worse on both MAE and R².

Therefore, the original Text + Structured Ridge model is retained as the final model candidate for the CreatorLens prediction pipeline.

## Important Limitation

The model predicts relative video performance rather than exact future views.

Its output should be interpreted as an estimate of how a video may perform relative to the historical baseline represented in the training data.

The model should not be presented as guaranteeing a specific number of views or a causal increase in performance.