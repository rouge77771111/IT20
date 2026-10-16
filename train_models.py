"""
Machine Learning Training Pipeline
Paper: Predicting Student Academic Performance Using Decision Tree and Random Forest Machine Learning Models
Authors: Centino, Norjeane P. & Saludes, Rouge P. L. (University of Mindanao)
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

def train_and_save():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, 'data', 'StudentPerformanceFactors.csv')
    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    artifacts_path = os.path.join(models_dir, 'model_artifacts.pkl')

    print(f"Loading dataset from: {data_path}")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    df = pd.read_csv(data_path)
    total_records = len(df)
    print(f"Total records: {total_records}, Columns: {df.shape[1]}")

    # Mode imputation for the 3 columns with missing values as detailed in paper section 2.5.1
    missing_cols = ['Teacher_Quality', 'Parental_Education_Level', 'Distance_from_Home']
    imputation_modes = {}
    for col in df.columns:
        if col != 'Exam_Score':
            imputation_modes[col] = df[col].mode()[0]

    for col in missing_cols:
        if col in df.columns:
            df[col] = df[col].fillna(imputation_modes[col])

    # Target variable preparation as defined in Section 3.2 Table 4
    # Low: 55-64, Medium: 65-69, High: 70-101
    def map_performance(score):
        if score <= 64:
            return 'Low'
        elif score <= 69:
            return 'Medium'
        else:
            return 'High'

    df['Performance_Level'] = df['Exam_Score'].apply(map_performance)

    # Class distribution
    class_distribution = df['Performance_Level'].value_counts().to_dict()
    print("Performance Level Distribution:", class_distribution)

    # Separate features and target
    X_raw = df.drop(columns=['Exam_Score', 'Performance_Level'])
    y = df['Performance_Level']

    # Categorical and numerical columns
    categorical_cols = [
        'Parental_Involvement', 'Access_to_Resources', 'Extracurricular_Activities',
        'Motivation_Level', 'Internet_Access', 'Family_Income', 'Teacher_Quality',
        'School_Type', 'Peer_Influence', 'Learning_Disabilities',
        'Parental_Education_Level', 'Distance_from_Home', 'Gender'
    ]
    numerical_cols = [
        'Hours_Studied', 'Attendance', 'Sleep_Hours',
        'Previous_Scores', 'Tutoring_Sessions', 'Physical_Activity'
    ]

    # Categorical choices/options for form UI
    categorical_options = {}
    for col in categorical_cols:
        categorical_options[col] = sorted(df[col].dropna().unique().tolist())

    numerical_stats = {}
    for col in numerical_cols:
        numerical_stats[col] = {
            'min': float(df[col].min()),
            'max': float(df[col].max()),
            'mean': round(float(df[col].mean()), 2),
            'median': float(df[col].median()),
            'default': float(df[col].median())
        }

    # One-hot encoding as specified in section 2.5.3 Categorical Encoding
    X_encoded = pd.get_dummies(X_raw, columns=categorical_cols, drop_first=False)
    feature_columns = list(X_encoded.columns)
    print(f"Encoded feature dimension: {len(feature_columns)}")

    # 80/20 Stratified Split (random_state=42) - Section 2.6 Data Splitting
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    labels_order = ['Low', 'Medium', 'High']

    # 1. Decision Tree Classifier (criterion='gini', random_state=42) - Section 2.7
    print("Training Decision Tree Classifier...")
    dt_model = DecisionTreeClassifier(random_state=42, criterion='gini')
    dt_model.fit(X_train, y_train)
    y_pred_dt = dt_model.predict(X_test)

    dt_acc = accuracy_score(y_test, y_pred_dt)
    dt_p_w, dt_r_w, dt_f1_w, _ = precision_recall_fscore_support(y_test, y_pred_dt, average='weighted')
    dt_p_m, dt_r_m, dt_f1_m, _ = precision_recall_fscore_support(y_test, y_pred_dt, average='macro')
    dt_cm = confusion_matrix(y_test, y_pred_dt, labels=labels_order).tolist()
    dt_rep = classification_report(y_test, y_pred_dt, labels=labels_order, output_dict=True)

    # 2. Random Forest Classifier (n_estimators=200, random_state=42, n_jobs=-1) - Section 2.8
    print("Training Random Forest Classifier (200 trees)...")
    rf_model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)

    rf_acc = accuracy_score(y_test, y_pred_rf)
    rf_p_w, rf_r_w, rf_f1_w, _ = precision_recall_fscore_support(y_test, y_pred_rf, average='weighted')
    rf_p_m, rf_r_m, rf_f1_m, _ = precision_recall_fscore_support(y_test, y_pred_rf, average='macro')
    rf_cm = confusion_matrix(y_test, y_pred_rf, labels=labels_order).tolist()
    rf_rep = classification_report(y_test, y_pred_rf, labels=labels_order, output_dict=True)

    # Feature importances from Random Forest - Table 6
    fi_series = pd.Series(rf_model.feature_importances_, index=feature_columns).sort_values(ascending=False)
    feature_importances = [
        {"feature": feat, "importance": round(float(val), 6), "percentage": round(float(val * 100), 2)}
        for feat, val in fi_series.items()
    ]

    print("\n--- Model Evaluation Summary ---")
    print(f"Decision Tree Accuracy: {dt_acc * 100:.2f}% (Weighted F1: {dt_f1_w * 100:.2f}%)")
    print(f"Random Forest Accuracy: {rf_acc * 100:.2f}% (Weighted F1: {rf_f1_w * 100:.2f}%)")

    # Paper comparison benchmarks
    paper_comparison = {
        "decision_tree": {
            "accuracy": 72.39,
            "weighted_precision": 72.45,
            "weighted_recall": 72.39,
            "weighted_f1": 72.39,
            "low_f1": 69.12,
            "medium_f1": 74.50,
            "high_f1": 70.75,
            "confusion_matrix": [[197, 90, 4], [79, 523, 104], [3, 85, 237]]
        },
        "random_forest": {
            "accuracy": 82.22,
            "weighted_precision": 85.02,
            "weighted_recall": 82.22,
            "weighted_f1": 81.56,
            "low_f1": 72.84,
            "medium_f1": 85.45,
            "high_f1": 80.93,
            "confusion_matrix": [[173, 118, 0], [10, 687, 9], [1, 97, 227]]
        },
        "top_features": [
            {"rank": 1, "feature": "Attendance", "importance": 0.256158},
            {"rank": 2, "feature": "Hours Studied", "importance": 0.164922},
            {"rank": 3, "feature": "Previous Scores", "importance": 0.080975},
            {"rank": 4, "feature": "Tutoring Sessions", "importance": 0.040811},
            {"rank": 5, "feature": "Sleep Hours", "importance": 0.039703},
            {"rank": 6, "feature": "Physical Activity", "importance": 0.034276},
            {"rank": 7, "feature": "Access to Resources – High", "importance": 0.015773},
            {"rank": 8, "feature": "Parental Involvement – High", "importance": 0.014487},
            {"rank": 9, "feature": "Family Income – Low", "importance": 0.013737},
            {"rank": 10, "feature": "Peer Influence – Positive", "importance": 0.013707}
        ]
    }

    artifacts = {
        'rf_model': rf_model,
        'dt_model': dt_model,
        'feature_columns': feature_columns,
        'categorical_cols': categorical_cols,
        'numerical_cols': numerical_cols,
        'categorical_options': categorical_options,
        'numerical_stats': numerical_stats,
        'imputation_modes': imputation_modes,
        'labels_order': labels_order,
        'class_distribution': class_distribution,
        'total_records': total_records,
        'metrics': {
            'dt': {
                'accuracy': round(float(dt_acc * 100), 2),
                'weighted_precision': round(float(dt_p_w * 100), 2),
                'weighted_recall': round(float(dt_r_w * 100), 2),
                'weighted_f1': round(float(dt_f1_w * 100), 2),
                'macro_precision': round(float(dt_p_m * 100), 2),
                'macro_recall': round(float(dt_r_m * 100), 2),
                'macro_f1': round(float(dt_f1_m * 100), 2),
                'confusion_matrix': dt_cm,
                'classification_report': dt_rep
            },
            'rf': {
                'accuracy': round(float(rf_acc * 100), 2),
                'weighted_precision': round(float(rf_p_w * 100), 2),
                'weighted_recall': round(float(rf_r_w * 100), 2),
                'weighted_f1': round(float(rf_f1_w * 100), 2),
                'macro_precision': round(float(rf_p_m * 100), 2),
                'macro_recall': round(float(rf_r_m * 100), 2),
                'macro_f1': round(float(rf_f1_m * 100), 2),
                'confusion_matrix': rf_cm,
                'classification_report': rf_rep
            }
        },
        'feature_importances': feature_importances,
        'paper_comparison': paper_comparison
    }

    with open(artifacts_path, 'wb') as f:
        pickle.dump(artifacts, f)

    print(f"Artifacts successfully saved to: {artifacts_path}")
    return artifacts

if __name__ == '__main__':
    train_and_save()
