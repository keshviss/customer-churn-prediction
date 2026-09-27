# Customer Churn Prediction & Customer Segmentation System

A machine learning web application that predicts customer churn risk and segments customers into meaningful groups, built as a final-year Computer Engineering ML project using classical algorithms.

## Features

- **Churn Prediction** — predicts whether a customer will churn using 4 classification models (Logistic Regression, Decision Tree, SVM, Random Forest)
- **Model Comparison** — accuracy, precision, recall, F1-score, and ROC-AUC across all models
- **Confusion Matrices** — visual breakdown of prediction errors per model
- **Customer Segmentation** — K-Means clustering (K=4) to identify customer groups
- **PCA Visualization** — 2D projection of customer segments

## Tech Stack

- **ML:** scikit-learn, pandas, numpy
- **Visualization:** matplotlib, seaborn
- **Backend:** Flask
- **Model persistence:** joblib

## Dataset

[Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) — 7,043 customer records, 21 features.

## Project Structure

customer-churn-prediction/
├── data/ # Dataset
├── notebooks/ # EDA, model training, evaluation
├── models/ # Saved trained models (joblib)
├── static/ # CSS and generated plot images
├── templates/ # Flask HTML templates
├── app.py # Flask application
└── requirements.txt


## Setup & Run

```bash
# Clone the repo
git clone <your-repo-url>
cd customer-churn-prediction

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the notebook (trains and saves models) — optional if models/ is already populated
jupyter notebook notebooks/eda_and_model_training.ipynb

# Run the web app
python app.py
```

Then open `http://127.0.0.1:5000` in your browser.

## Model Performance

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 80.55% | 65.72% | 55.88% | 0.6040 | 0.8420 |
| Decision Tree | 78.21% | 67.01% | 35.29% | 0.4623 | 0.8183 |
| SVM | 79.63% | 65.16% | 50.00% | 0.5658 | 0.7941 |
| Random Forest | 80.41% | 67.01% | 51.60% | 0.5831 | 0.8411 |

## Limitations

This is an academic demonstration project. Predictions are statistical estimates based on a specific historical dataset and should not be treated as a validated business decision-making tool. See the "About" page in the app for full details.