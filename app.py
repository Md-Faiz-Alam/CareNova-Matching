from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from flask_cors import CORS


app = Flask(__name__)
CORS(app)
# Load saved artifacts
tfidf_vectorizer = joblib.load('E:/CareNova_patient_matching/models/tfidf_vectorizer.pkl')       # fitted TF-IDF vectorizer
best_hgb = joblib.load('E:/CareNova_patient_matching/models/best_hgb_model.pkl')                 # trained model
onehot_cat_columns = joblib.load('E:/CareNova_patient_matching/models/onehot_cat_columns.pkl')   # list of one-hot encoded categorical columns

# Define feature columns (same as used during training)
numeric_cols = [
    'enrollment', 'study_duration_days', 'sex_all', 'sex_female', 'sex_male',
    'has_child', 'has_adult', 'has_older_adult', 'phase1', 'phase2', 'phase3',
    'funder_fed', 'funder_indiv', 'funder_industry', 'funder_network',
    'funder_nih', 'funder_other', 'funder_other_gov', 'funder_unknown',
    'missing_start_date', 'missing_primary_completion_date', 'missing_completion_date'
]

cat_cols = ['sponsor', 'study_type', 'study_design']

text_cols = [
    'study_title', 'brief_summary', 'conditions', 'interventions',
    'primary_outcome_measures', 'secondary_outcome_measures', 'locations'
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.json
        new_df = pd.DataFrame([input_data])

        # Fill missing numeric columns with 0
        new_df[numeric_cols] = new_df.get(numeric_cols, pd.DataFrame()).fillna(0)

        # Handle boolean-like numeric columns
        for col in numeric_cols:
            if col in new_df.columns:
                new_df[col] = new_df[col].apply(
                    lambda x: 1 if str(x).lower() in ['true', '1', 'yes', 'on']
                    else 0 if str(x).lower() in ['false', '0', 'no', 'off'] else x)

        # Fill missing categorical and text
        new_df[cat_cols] = new_df[cat_cols].fillna('Unknown')
        new_df[text_cols] = new_df[text_cols].fillna('')

        # One-hot encoding
        new_cat_encoded = pd.get_dummies(new_df[cat_cols])
        missing_cols = [col for col in onehot_cat_columns if col not in new_cat_encoded.columns]
        if missing_cols:
            missing_df = pd.DataFrame(0, index=new_cat_encoded.index, columns=missing_cols)
            new_cat_encoded = pd.concat([new_cat_encoded, missing_df], axis=1)
        new_cat_encoded = new_cat_encoded[onehot_cat_columns]

        # TF-IDF on combined text
        new_df['combined_text'] = new_df[text_cols].agg(' '.join, axis=1)
        X_text = tfidf_vectorizer.transform(new_df['combined_text'])

        # Numeric + categorical sparse matrices
        X_num = csr_matrix(new_df[numeric_cols].astype(float).values)
        X_cat = csr_matrix(new_cat_encoded.values)

        # Combine all
        X_input_sparse = hstack([X_num, X_cat, X_text])
        X_input = X_input_sparse.toarray()

        # Predict class and probability
        y_pred = best_hgb.predict(X_input)
        y_prob = best_hgb.predict_proba(X_input)[:, 1]

        # Convert prediction to boolean for "match"
        match = bool(y_pred[0])

        return jsonify({
            "match": match,
            "probability": float(y_prob[0])
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
