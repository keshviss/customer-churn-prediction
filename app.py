from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load dataset once at startup, just for home page stats
df = pd.read_csv('data/customer_churn.csv')

# Load the preprocessing pipeline and all trained models once at startup
# (loading here, not per-request, avoids reloading from disk on every prediction)
preprocessor = joblib.load('models/preprocessing.pkl')

MODELS = {
    'Logistic Regression': joblib.load('models/logistic_regression.pkl'),
    'Decision Tree': joblib.load('models/decision_tree.pkl'),
    'SVM': joblib.load('models/svm.pkl'),
    'Random Forest': joblib.load('models/random_forest.pkl')
}

# Category options for dropdowns — single source of truth, passed to the template
CATEGORY_OPTIONS = {
    'gender': ['Female', 'Male'],
    'SeniorCitizen': ['No', 'Yes'],
    'Partner': ['No', 'Yes'],
    'Dependents': ['No', 'Yes'],
    'PhoneService': ['No', 'Yes'],
    'MultipleLines': ['No', 'No phone service', 'Yes'],
    'InternetService': ['DSL', 'Fiber optic', 'No'],
    'OnlineSecurity': ['No', 'No internet service', 'Yes'],
    'OnlineBackup': ['No', 'No internet service', 'Yes'],
    'DeviceProtection': ['No', 'No internet service', 'Yes'],
    'TechSupport': ['No', 'No internet service', 'Yes'],
    'StreamingTV': ['No', 'No internet service', 'Yes'],
    'StreamingMovies': ['No', 'No internet service', 'Yes'],
    'Contract': ['Month-to-month', 'One year', 'Two year'],
    'PaperlessBilling': ['No', 'Yes'],
    'PaymentMethod': ['Bank transfer (automatic)', 'Credit card (automatic)',
                       'Electronic check', 'Mailed check']
}

# Metrics computed from the notebook (Steps 5, 6, 7, 8) — fixed values from our trained models,
# not recalculated live, since retraining/reevaluating on every page load would be unnecessary overhead.
MODEL_METRICS = [
    {'name': 'Logistic Regression', 'accuracy': 0.8055, 'precision': 0.6572, 'recall': 0.5588, 'f1': 0.6040, 'roc_auc': 0.8420},
    {'name': 'Decision Tree', 'accuracy': 0.7821, 'precision': 0.6701, 'recall': 0.3529, 'f1': 0.4623, 'roc_auc': 0.8183},
    {'name': 'SVM', 'accuracy': 0.7963, 'precision': 0.6516, 'recall': 0.5000, 'f1': 0.5658, 'roc_auc': 0.7941},
    {'name': 'Random Forest', 'accuracy': 0.8041, 'precision': 0.6701, 'recall': 0.5160, 'f1': 0.5831, 'roc_auc': 0.8411},
]

# Cluster profile computed in the notebook (Step 11) — fixed values from our trained K-Means model
CLUSTER_PROFILES = [
    {'id': 0, 'label': 'New, Low-Spend', 'tenure': 10.21, 'monthly_charges': 31.78,
     'total_charges': 302.15, 'count': 1703, 'churn_rate': 24.66},
    {'id': 1, 'label': 'Loyal, High-Spend', 'tenure': 59.53, 'monthly_charges': 93.31,
     'total_charges': 5548.65, 'count': 1904, 'churn_rate': 15.39},
    {'id': 2, 'label': 'New, High-Spend (Highest Risk)', 'tenure': 15.42, 'monthly_charges': 80.78,
     'total_charges': 1251.17, 'count': 2276, 'churn_rate': 48.24},
    {'id': 3, 'label': 'Loyal, Low-Spend (Most Stable)', 'tenure': 53.57, 'monthly_charges': 34.91,
     'total_charges': 1835.62, 'count': 1160, 'churn_rate': 5.00},
]

def get_risk_category(probability):
    """
    Converts a churn probability into a simple risk label.
    Thresholds are chosen for this project's demonstration purposes only —
    not derived from any business cost analysis. This is documented clearly
    in the report/about page so it isn't presented as validated business logic.
    """
    if probability < 0.30:
        return 'LOW'
    elif probability < 0.60:
        return 'MEDIUM'
    else:
        return 'HIGH'


@app.route('/')
def home():
    total_records = len(df)
    available_models = list(MODELS.keys())
    return render_template('index.html', total_records=total_records, available_models=available_models)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        # Show the empty form
        return render_template('predict.html', categories=CATEGORY_OPTIONS,
                                model_names=list(MODELS.keys()))

    # POST: form was submitted
    try:
        # Build a single-row DataFrame matching the exact columns the preprocessor expects
        input_data = {
            'gender': request.form['gender'],
            'SeniorCitizen': 1 if request.form['SeniorCitizen'] == 'Yes' else 0,
            'Partner': request.form['Partner'],
            'Dependents': request.form['Dependents'],
            'tenure': int(request.form['tenure']),
            'PhoneService': request.form['PhoneService'],
            'MultipleLines': request.form['MultipleLines'],
            'InternetService': request.form['InternetService'],
            'OnlineSecurity': request.form['OnlineSecurity'],
            'OnlineBackup': request.form['OnlineBackup'],
            'DeviceProtection': request.form['DeviceProtection'],
            'TechSupport': request.form['TechSupport'],
            'StreamingTV': request.form['StreamingTV'],
            'StreamingMovies': request.form['StreamingMovies'],
            'Contract': request.form['Contract'],
            'PaperlessBilling': request.form['PaperlessBilling'],
            'PaymentMethod': request.form['PaymentMethod'],
            'MonthlyCharges': float(request.form['MonthlyCharges']),
            'TotalCharges': float(request.form['TotalCharges'])
        }

        selected_model_name = request.form['model_choice']
        model = MODELS[selected_model_name]

        # Wrap in a DataFrame — the preprocessor expects the same shape/columns as training
        input_df = pd.DataFrame([input_data])

        # Apply the SAME fitted preprocessing used during training (no re-fitting here)
        input_processed = preprocessor.transform(input_df)

        # Predict class and probability
        prediction = model.predict(input_processed)[0]
        probabilities = model.predict_proba(input_processed)[0]
        no_churn_proba = probabilities[0]
        churn_proba = probabilities[1]

        risk = get_risk_category(churn_proba)

        return render_template('result.html',
                                prediction='Churn' if prediction == 1 else 'No Churn',
                                churn_proba=round(churn_proba * 100, 2),
                                no_churn_proba=round(no_churn_proba * 100, 2),
                                risk=risk,
                                model_used=selected_model_name)

    except (ValueError, KeyError) as e:
        # Basic input validation — redisplay the form with an error message
        return render_template('predict.html', categories=CATEGORY_OPTIONS,
                                model_names=list(MODELS.keys()),
                                error=f"Invalid input: {e}. Please check all fields.")

@app.route('/models')
def models_comparison():
    # Find the best model per metric, to highlight in the template
    best_accuracy = max(MODEL_METRICS, key=lambda m: m['accuracy'])['name']
    best_f1 = max(MODEL_METRICS, key=lambda m: m['f1'])['name']
    best_roc_auc = max(MODEL_METRICS, key=lambda m: m['roc_auc'])['name']

    return render_template('models.html', metrics=MODEL_METRICS,
                            best_accuracy=best_accuracy, best_f1=best_f1, best_roc_auc=best_roc_auc)

@app.route('/confusion-matrices')
def confusion_matrices():
    matrix_images = [
        {'name': 'Logistic Regression', 'filename': 'cm_logistic_regression.png'},
        {'name': 'Decision Tree', 'filename': 'cm_decision_tree.png'},
        {'name': 'SVM', 'filename': 'cm_svm.png'},
        {'name': 'Random Forest', 'filename': 'cm_random_forest.png'}
    ]
    return render_template('confusion_matrices.html', matrices=matrix_images)

@app.route('/clustering')
def clustering():
    return render_template('clustering.html', clusters=CLUSTER_PROFILES)

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)