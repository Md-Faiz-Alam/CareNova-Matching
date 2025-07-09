# CareNova Patient Matching Project 🧠💊

A clinical trial intelligence system powered by machine learning to predict patient-study matches, streamline recruitment, and improve healthcare research outcomes.

## 🧾 Table of Contents

📌 Project Overview

✨ Features

📊 Dataset

🧹 Data Preparation

📌 Feature Descriptions

🧠 Modeling

📈 Evaluation

🚀 API & Deployment

⚙️ Installation

▶️ Usage

🤝 Contributing

📄 License

📬 Contact

---
## 📌 Project Overview

CareNova aims to improve clinical trial recruitment by predicting whether a trial will match a patient or not using machine learning. It works on structured + unstructured data (text, category, number) and predicts the likelihood of trial eligibility, streamlining healthcare research and reducing cost and time.

---

## ✨ Features

✅ Clean preprocessing pipeline for numeric, categorical, and text data

🧠 TF-IDF on clinical summaries for semantic insight

🔍 Feature engineering + missing value imputation

📈 Advanced model using HistGradientBoostingClassifier

⚖️ Handles class imbalance smartly

📊 Model metrics: ROC-AUC, Accuracy, F1, etc

🌐 Flask-based REST API for predictions

---

## 📊 Dataset

**Source**: [ClinicalTrials.gov](https://clinicaltrials.gov/)

This dataset includes real-world clinical trial metadata such as:

- Study titles, summaries, conditions, interventions
- Enrollment numbers, study phases, durations
- Funding sources, study design, demographic inclusion

**Target Variable**: `study_status`
  
- `1`: Study matches patient or inclusion criteria
  
- 0`: Study does not match
  
**Note**: The data is imbalanced with ~80% positive and ~20% negative labels

---

## 🧹 Data Preparation

- Missing Values:

Numeric: median imputation

- Categorical: "Unknown"

Text: empty string

- Text Vectorization:

TF-IDF (max 1000 features, English stop words)

- Categorical:

One-hot encoded

- Final Matrix: All features concatenated and normalized

---

## 🧾 Feature Descriptions

| **Feature Name**                  | **What It Means**                                                                             |
| --------------------------------- | --------------------------------------------------------------------------------------------- |
| `study_title`                     | The name or title of the clinical trial.                                                      |
| `study_status`                    | 📌 **Target column**: Shows if the study matches required criteria (1 = match, 0 = no match). |
| `brief_summary`                   | A short summary of what the study is about.                                                   |
| `conditions`                      | Diseases or health conditions that the study focuses on.                                      |
| `interventions`                   | Treatments, medicines, or actions tested in the study.                                        |
| `primary_outcome_measures`        | The main results that the study wants to measure.                                             |
| `secondary_outcome_measures`      | Other additional results being measured.                                                      |
| `sponsor`                         | The organization or company funding the study.                                                |
| `enrollment`                      | Number of people participating in the study.                                                  |
| `study_type`                      | The kind of study (like Interventional or Observational).                                     |
| `study_design`                    | Details about how the study is set up (e.g., Randomized, Open Label).                         |
| `last_update_posted`              | The last date the study details were updated.                                                 |
| `locations`                       | Places where the study is happening.                                                          |
| `study_duration_days`             | How long the study will run (in days).                                                        |
| `sex_all`                         | True/False: If the study includes all genders.                                                |
| `sex_female`                      | True/False: If it includes females.                                                           |
| `sex_male`                        | True/False: If it includes males.                                                             |
| `has_child`                       | 1/0: Whether children can take part in the study.                                             |
| `has_adult`                       | 1/0: Whether adults can take part.                                                            |
| `has_older_adult`                 | 1/0: Whether older adults (e.g., 60+) can take part.                                          |
| `phase1`                          | 1/0: Is it a Phase 1 clinical trial?                                                          |
| `phase2`                          | 1/0: Is it a Phase 2 trial?                                                                   |
| `phase3`                          | 1/0: Is it a Phase 3 trial?                                                                   |
| `funder_fed`                      | True/False: Funded by the federal government.                                                 |
| `funder_indiv`                    | True/False: Funded by an individual sponsor.                                                  |
| `funder_industry`                 | True/False: Funded by a private company or industry.                                          |
| `funder_network`                  | True/False: Funded by a research group or network.                                            |
| `funder_nih`                      | True/False: Funded by the National Institutes of Health (NIH).                                |
| `funder_other`                    | True/False: Funded by some other source.                                                      |
| `funder_other_gov`                | True/False: Funded by another government body.                                                |
| `funder_unknown`                  | True/False: Funding source is unknown.                                                        |
| `missing_start_date`              | 1/0: If the start date is missing (1 = yes, 0 = no).                                          |
| `missing_primary_completion_date` | 1/0: If the primary completion date is missing.                                               |
| `missing_completion_date`         | 1/0: If the study's final completion date is missing.                                         |

---

## 🧠 Modeling

📌 **Model**: [HistGradientBoostingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html) from scikit-learn

🔁 Cross-validation used to prevent overfitting and ensure generalization

🎯 Tuned with [RandomizedSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html)

🔧 Key Parameters tuned:
  `max_iter`
  `learning_rate`
  `max_depth`
  
⚖️ Optional: Use of [SMOTE](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html) or under-sampling to handle class imbalance

---

## 📈 Evaluation

**📊 Metrics Used:**
• Accuracy  
• ROC-AUC  
• Precision & Recall  
• F1 Score  

**📉 Visualizations:**
• ROC Curve  
• Confusion Matrix  
• Feature Correlation Heatmap

---

## API & Deployment

Flask-based REST API for serving predictions
Input accepts JSON with study features and returns predicted match status and probability
Includes data preprocessing pipeline for incoming requests (imputation, encoding, vectorization)
CORS enabled for cross-domain access
Can be containerized with Docker for scalable deployment (not included here)

---

## ⚙️ Installation

### Clone the repo

git clone https://github.com/Md-Faiz-Alam/carenova-patient-matching.git
cd carenova-patient-matching

### Create virtual environment (optional but recommended)

python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

### Install dependencies

pip install -r requirements.txt

---

## ▶️ Usage

#### 🔬 Run Model Training
python train_model.py

#### 🧪 Evaluate Model

python evaluate_model.py

#### 🌐 Run the API
python app.py

Send a POST request to:
http://localhost:5000/predict

---

## 🤝 Contributing:

Fork the project

Create your branch (git checkout -b feature/xyz)

Commit your changes (git commit -m 'Add feature')

Push and open a Pull Request

---

## 📬 Contact

**👤 Muhammad Faiz Alam**  
📧 Email: [md.faizalam2003@gmail.com](mailto:md.faizalam2003@gmail.com)  
🔗 LinkedIn: [View Profile](https://www.linkedin.com/in/md-faiz-alam)  
💻 GitHub: [Md-Faiz-Alam](https://github.com/Md-Faiz-Alam)

---

## ⭐ If you find this project helpful, consider giving it a star!
