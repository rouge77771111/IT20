"""
Flask Web Application for Predicting Student Academic Performance
Paper: Predicting Student Academic Performance Using Decision Tree and Random Forest Machine Learning Models
Authors: Centino, Norjeane P. & Saludes, Rouge P. L. (University of Mindanao)
"""

import os
import io
import pickle
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for, flash

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'student-academic-performance-prediction-key-2026')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_PATH = os.path.join(BASE_DIR, 'models', 'model_artifacts.pkl')

# Global cache for loaded model artifacts
artifacts = None

def load_artifacts():
    global artifacts
    if artifacts is None:
        if not os.path.exists(ARTIFACTS_PATH):
            print("Model artifacts not found. Initiating training pipeline...")
            from train_models import train_and_save
            artifacts = train_and_save()
        else:
            with open(ARTIFACTS_PATH, 'rb') as f:
                artifacts = pickle.load(f)
            print("Model artifacts successfully loaded from disk.")
    return artifacts

# Initialize artifacts on startup
with app.app_context():
    load_artifacts()

def preprocess_input(records_df):
    """
    Takes a DataFrame containing the 19 raw student features,
    imputes missing categorical values, and one-hot encodes to match training features.
    """
    art = load_artifacts()
    df = records_df.copy()

    # Impute missing values with dataset modes
    for col, mode_val in art['imputation_modes'].items():
        if col in df.columns:
            df[col] = df[col].fillna(mode_val)
        else:
            df[col] = mode_val

    # Ensure numerical columns are numeric
    for col in art['numerical_cols']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(art['numerical_stats'][col]['default'])

    # Categorical one-hot encoding
    df_encoded = pd.get_dummies(df[art['numerical_cols'] + art['categorical_cols']], columns=art['categorical_cols'], drop_first=False)

    # Reindex columns to match exact trained feature columns
    df_aligned = df_encoded.reindex(columns=art['feature_columns'], fill_value=0)
    return df_aligned

def generate_interventions(student_data, predicted_level, rf_probs):
    """
    Generates tailored academic intervention recommendations based on student variables
    and model prediction, following Section 1.1 and 5.2 of the research paper.
    """
    interventions = []
    attendance = float(student_data.get('Attendance', 80))
    hours_studied = float(student_data.get('Hours_Studied', 20))
    prev_scores = float(student_data.get('Previous_Scores', 75))
    tutoring = float(student_data.get('Tutoring_Sessions', 1))
    sleep = float(student_data.get('Sleep_Hours', 7))
    parental = student_data.get('Parental_Involvement', 'Medium')
    resources = student_data.get('Access_to_Resources', 'Medium')
    peer = student_data.get('Peer_Influence', 'Neutral')

    # Attendance Recommendations (Top Feature - Importance ~25.6%)
    if attendance < 75:
        interventions.append({
            'category': 'Attendance Alert (High Priority)',
            'badge': 'Urgent',
            'badge_color': 'bg-rose-100 text-rose-800 border-rose-300',
            'icon': 'fa-triangle-exclamation',
            'detail': f'Current attendance is {attendance:.1f}% (critical threshold < 75%). Attendance was determined to be the #1 most influential predictor of academic performance in the study. Establish immediate attendance monitoring and conduct a parent-counselor check-in.'
        })
    elif attendance < 85:
        interventions.append({
            'category': 'Attendance Monitoring',
            'badge': 'Moderate',
            'badge_color': 'bg-amber-100 text-amber-800 border-amber-300',
            'icon': 'fa-calendar-check',
            'detail': f'Current attendance is {attendance:.1f}%. Encourage regular class participation to reach the upper quartile (> 88%).'
        })

    # Study Time Recommendations (Feature #2 - Importance ~16.5%)
    if hours_studied < 15:
        interventions.append({
            'category': 'Study Hours Intervention',
            'badge': 'Urgent',
            'badge_color': 'bg-rose-100 text-rose-800 border-rose-300',
            'icon': 'fa-clock',
            'detail': f'Student studies {hours_studied:.1f} hours/week (below the dataset mean of ~20 hrs/week). Implement a structured weekly study timetable and recommend peer study sessions.'
        })
    elif hours_studied >= 25:
        interventions.append({
            'category': 'Study Habit Commendation',
            'badge': 'Positive',
            'badge_color': 'bg-emerald-100 text-emerald-800 border-emerald-300',
            'icon': 'fa-award',
            'detail': f'Student spends {hours_studied:.1f} hours/week studying, which significantly reinforces high performance retention.'
        })

    # Previous Score & Remediation (Feature #3 - Importance ~8.1%)
    if prev_scores < 65:
        interventions.append({
            'category': 'Academic Remediation',
            'badge': 'Urgent',
            'badge_color': 'bg-rose-100 text-rose-800 border-rose-300',
            'icon': 'fa-book-open',
            'detail': f'Previous score ({prev_scores:.1f}) indicates learning gaps in foundation concepts. Recommend diagnostic assessment and departmental remedial classes.'
        })

    # Tutoring Support (Feature #4 - Importance ~4.1%)
    if tutoring == 0 and (predicted_level == 'Low' or prev_scores < 70):
        interventions.append({
            'category': 'Tutoring Assignment',
            'badge': 'Recommended',
            'badge_color': 'bg-blue-100 text-blue-800 border-blue-300',
            'icon': 'fa-chalkboard-user',
            'detail': 'Student currently attends 0 tutoring sessions. Assigning 1 to 2 targeted tutoring sessions monthly can bridge performance gaps.'
        })

    # Sleep & Behavioral Well-being (Feature #5 & #6)
    if sleep < 6:
        interventions.append({
            'category': 'Sleep Hygiene & Well-being',
            'badge': 'Advisory',
            'badge_color': 'bg-purple-100 text-purple-800 border-purple-300',
            'icon': 'fa-bed',
            'detail': f'Student averages only {sleep:.1f} hours of sleep nightly. Sleep deprivation negatively impairs examination retention and cognitive focus.'
        })

    # Parental Involvement & Resources Support
    if parental == 'Low' or resources == 'Low':
        interventions.append({
            'category': 'Educational Support & Resources',
            'badge': 'Support',
            'badge_color': 'bg-indigo-100 text-indigo-800 border-indigo-300',
            'icon': 'fa-handshake',
            'detail': 'Low access to resources or parental involvement identified. Facilitate access to university library terminals, textbook loan programs, and faculty advising.'
        })

    # Peer Influence Advisory
    if peer == 'Negative':
        interventions.append({
            'category': 'Peer Environment',
            'badge': 'Advisory',
            'badge_color': 'bg-amber-100 text-amber-800 border-amber-300',
            'icon': 'fa-users',
            'detail': 'Student experiences negative peer influence. Pair student with student mentor groups or positive academic study circles.'
        })

    if not interventions:
        interventions.append({
            'category': 'General Academic Maintenance',
            'badge': 'Good Standing',
            'badge_color': 'bg-emerald-100 text-emerald-800 border-emerald-300',
            'icon': 'fa-circle-check',
            'detail': 'Student profile aligns with stable academic progress. Maintain periodic advisor check-ins and recognize continued performance.'
        })

    return interventions

@app.route('/')
def index():
    art = load_artifacts()
    return render_template(
        'index.html',
        categorical_cols=art['categorical_cols'],
        categorical_options=art['categorical_options'],
        numerical_cols=art['numerical_cols'],
        numerical_stats=art['numerical_stats'],
        metrics=art['metrics'],
        imputation_modes=art['imputation_modes']
    )

@app.route('/predict', methods=['POST'])
def predict():
    art = load_artifacts()
    try:
        # Collect form input values
        input_data = {}
        for col in art['numerical_cols']:
            val = request.form.get(col, '')
            try:
                input_data[col] = float(val) if val != '' else art['numerical_stats'][col]['default']
            except ValueError:
                input_data[col] = art['numerical_stats'][col]['default']

        for col in art['categorical_cols']:
            input_data[col] = request.form.get(col, art['imputation_modes'][col])

        model_choice = request.form.get('model_choice', 'rf')

        # Convert to DataFrame and align features
        raw_df = pd.DataFrame([input_data])
        X_proc = preprocess_input(raw_df)

        # Random Forest prediction
        rf_pred = art['rf_model'].predict(X_proc)[0]
        rf_probs_raw = art['rf_model'].predict_proba(X_proc)[0]
        rf_classes = list(art['rf_model'].classes_)
        rf_prob_dict = {cls: round(float(prob) * 100, 1) for cls, prob in zip(rf_classes, rf_probs_raw)}

        # Decision Tree prediction
        dt_pred = art['dt_model'].predict(X_proc)[0]
        dt_probs_raw = art['dt_model'].predict_proba(X_proc)[0]
        dt_classes = list(art['dt_model'].classes_)
        dt_prob_dict = {cls: round(float(prob) * 100, 1) for cls, prob in zip(dt_classes, dt_probs_raw)}

        primary_pred = rf_pred if model_choice in ['rf', 'both'] else dt_pred
        primary_probs = rf_prob_dict if model_choice in ['rf', 'both'] else dt_prob_dict

        # Interventions
        interventions = generate_interventions(input_data, primary_pred, primary_probs)

        # Style helpers
        badge_styles = {
            'High': {'color': 'emerald', 'bg': 'bg-emerald-100', 'text': 'text-emerald-800', 'border': 'border-emerald-400', 'icon': 'fa-circle-check', 'label': 'High Performance (70–101)'},
            'Medium': {'color': 'amber', 'bg': 'bg-amber-100', 'text': 'text-amber-800', 'border': 'border-amber-400', 'icon': 'fa-circle-info', 'label': 'Medium Performance (65–69)'},
            'Low': {'color': 'rose', 'bg': 'bg-rose-100', 'text': 'text-rose-800', 'border': 'border-rose-400', 'icon': 'fa-triangle-exclamation', 'label': 'Low Performance (55–64) - Academic Risk'}
        }

        result = {
            'model_choice': model_choice,
            'primary_pred': primary_pred,
            'rf_pred': rf_pred,
            'rf_probs': rf_prob_dict,
            'dt_pred': dt_pred,
            'dt_probs': dt_prob_dict,
            'interventions': interventions,
            'input_data': input_data,
            'badge': badge_styles.get(primary_pred, badge_styles['Medium'])
        }

        return render_template(
            'index.html',
            categorical_cols=art['categorical_cols'],
            categorical_options=art['categorical_options'],
            numerical_cols=art['numerical_cols'],
            numerical_stats=art['numerical_stats'],
            metrics=art['metrics'],
            imputation_modes=art['imputation_modes'],
            result=result,
            form_values=input_data,
            selected_model=model_choice
        )

    except Exception as e:
        flash(f"Error during prediction: {str(e)}", "danger")
        return redirect(url_for('index'))

@app.route('/batch', methods=['GET', 'POST'])
def batch():
    art = load_artifacts()
    if request.method == 'GET':
        return render_template('batch.html')

    if 'csv_file' not in request.files:
        flash('No file selected. Please upload a valid CSV file.', 'warning')
        return redirect(url_for('batch'))

    file = request.files['csv_file']
    if file.filename == '':
        flash('No file selected.', 'warning')
        return redirect(url_for('batch'))

    if not file.filename.endswith('.csv'):
        flash('Only CSV format is supported. Please upload a .csv file.', 'danger')
        return redirect(url_for('batch'))

    try:
        df = pd.read_csv(file)
        if len(df) == 0:
            flash('The uploaded CSV file is empty.', 'danger')
            return redirect(url_for('batch'))

        model_selected = request.form.get('model_choice', 'rf')
        chosen_model = art['rf_model'] if model_selected == 'rf' else art['dt_model']

        # Preprocess features
        X_proc = preprocess_input(df)

        predictions = chosen_model.predict(X_proc)
        probabilities = chosen_model.predict_proba(X_proc)
        classes = list(chosen_model.classes_)

        df_results = df.copy()
        df_results['Predicted_Performance'] = predictions

        # Add probabilities
        for i, cls in enumerate(classes):
            df_results[f'Prob_{cls}'] = [round(float(p[i]) * 100, 1) for p in probabilities]

        # Add Risk Flag
        df_results['Risk_Flag'] = df_results['Predicted_Performance'].apply(
            lambda x: 'High Risk' if x == 'Low' else ('Moderate' if x == 'Medium' else 'On Track')
        )

        # Summary statistics
        total_students = len(df_results)
        counts = df_results['Predicted_Performance'].value_counts().to_dict()
        low_count = counts.get('Low', 0)
        med_count = counts.get('Medium', 0)
        high_count = counts.get('High', 0)

        summary = {
            'total': total_students,
            'low_count': low_count,
            'med_count': med_count,
            'high_count': high_count,
            'low_pct': round((low_count / total_students) * 100, 1),
            'med_pct': round((med_count / total_students) * 100, 1),
            'high_pct': round((high_count / total_students) * 100, 1),
            'model_name': 'Random Forest (82.2% Accuracy)' if model_selected == 'rf' else 'Decision Tree (72.4% Accuracy)'
        }

        # Cache results in temporary CSV for download
        os.makedirs(os.path.join(BASE_DIR, 'scratch'), exist_ok=True)
        cached_file = os.path.join(BASE_DIR, 'scratch', 'latest_batch_predictions.csv')
        df_results.to_csv(cached_file, index=False)

        # Pass preview of first 100 records
        preview_data = df_results.head(100).to_dict(orient='records')
        table_columns = ['Predicted_Performance', 'Risk_Flag'] + [c for c in ['Attendance', 'Hours_Studied', 'Previous_Scores', 'Tutoring_Sessions'] if c in df_results.columns]

        return render_template(
            'batch.html',
            summary=summary,
            records=preview_data,
            table_columns=table_columns,
            has_results=True,
            can_download=True
        )

    except Exception as e:
        flash(f"Error processing CSV: {str(e)}", 'danger')
        return redirect(url_for('batch'))

@app.route('/download-predictions')
def download_predictions():
    cached_file = os.path.join(BASE_DIR, 'scratch', 'latest_batch_predictions.csv')
    if os.path.exists(cached_file):
        return send_file(
            cached_file,
            mimetype='text/csv',
            as_attachment=True,
            download_name='predicted_student_performance.csv'
        )
    flash('No prediction file available to download. Please perform a batch prediction first.', 'warning')
    return redirect(url_for('batch'))

@app.route('/sample-csv')
def sample_csv():
    """Generates a downloadable clean CSV sample template for testing batch predictions."""
    sample_data = [
        {
            'Hours_Studied': 23, 'Attendance': 84, 'Parental_Involvement': 'Low',
            'Access_to_Resources': 'High', 'Extracurricular_Activities': 'No',
            'Sleep_Hours': 7, 'Previous_Scores': 73, 'Motivation_Level': 'Low',
            'Internet_Access': 'Yes', 'Tutoring_Sessions': 0, 'Family_Income': 'Low',
            'Teacher_Quality': 'Medium', 'School_Type': 'Public', 'Peer_Influence': 'Positive',
            'Physical_Activity': 3, 'Learning_Disabilities': 'No',
            'Parental_Education_Level': 'High School', 'Distance_from_Home': 'Near', 'Gender': 'Male'
        },
        {
            'Hours_Studied': 10, 'Attendance': 64, 'Parental_Involvement': 'Low',
            'Access_to_Resources': 'Low', 'Extracurricular_Activities': 'No',
            'Sleep_Hours': 5, 'Previous_Scores': 58, 'Motivation_Level': 'Low',
            'Internet_Access': 'No', 'Tutoring_Sessions': 0, 'Family_Income': 'Low',
            'Teacher_Quality': 'Low', 'School_Type': 'Public', 'Peer_Influence': 'Negative',
            'Physical_Activity': 1, 'Learning_Disabilities': 'No',
            'Parental_Education_Level': 'High School', 'Distance_from_Home': 'Far', 'Gender': 'Female'
        },
        {
            'Hours_Studied': 32, 'Attendance': 96, 'Parental_Involvement': 'High',
            'Access_to_Resources': 'High', 'Extracurricular_Activities': 'Yes',
            'Sleep_Hours': 8, 'Previous_Scores': 95, 'Motivation_Level': 'High',
            'Internet_Access': 'Yes', 'Tutoring_Sessions': 3, 'Family_Income': 'High',
            'Teacher_Quality': 'High', 'School_Type': 'Private', 'Peer_Influence': 'Positive',
            'Physical_Activity': 4, 'Learning_Disabilities': 'No',
            'Parental_Education_Level': 'Postgraduate', 'Distance_from_Home': 'Near', 'Gender': 'Female'
        },
        {
            'Hours_Studied': 18, 'Attendance': 78, 'Parental_Involvement': 'Medium',
            'Access_to_Resources': 'Medium', 'Extracurricular_Activities': 'Yes',
            'Sleep_Hours': 7, 'Previous_Scores': 68, 'Motivation_Level': 'Medium',
            'Internet_Access': 'Yes', 'Tutoring_Sessions': 1, 'Family_Income': 'Medium',
            'Teacher_Quality': 'Medium', 'School_Type': 'Public', 'Peer_Influence': 'Neutral',
            'Physical_Activity': 3, 'Learning_Disabilities': 'No',
            'Parental_Education_Level': 'College', 'Distance_from_Home': 'Moderate', 'Gender': 'Male'
        },
        {
            'Hours_Studied': 14, 'Attendance': 68, 'Parental_Involvement': 'Low',
            'Access_to_Resources': 'Medium', 'Extracurricular_Activities': 'No',
            'Sleep_Hours': 6, 'Previous_Scores': 61, 'Motivation_Level': 'Low',
            'Internet_Access': 'Yes', 'Tutoring_Sessions': 0, 'Family_Income': 'Low',
            'Teacher_Quality': 'Medium', 'School_Type': 'Public', 'Peer_Influence': 'Negative',
            'Physical_Activity': 2, 'Learning_Disabilities': 'Yes',
            'Parental_Education_Level': 'High School', 'Distance_from_Home': 'Far', 'Gender': 'Male'
        }
    ]
    df_sample = pd.DataFrame(sample_data)
    output = io.StringIO()
    df_sample.to_csv(output, index=False)
    output.seek(0)
    return send_file(
        io.BytesIO(output.getvalue().encode('utf-8')),
        mimetype='text/csv',
        as_attachment=True,
        download_name='student_template_sample.csv'
    )

@app.route('/analytics')
def analytics():
    art = load_artifacts()
    return render_template(
        'analytics.html',
        metrics=art['metrics'],
        feature_importances=art['feature_importances'],
        paper_comparison=art['paper_comparison'],
        class_distribution=art['class_distribution'],
        total_records=art['total_records']
    )

@app.route('/about')
def about():
    art = load_artifacts()
    return render_template('about.html', total_records=art.get('total_records', 6607))

@app.route('/api/predict', methods=['POST'])
def api_predict():
    art = load_artifacts()
    try:
        data = request.get_json(force=True)
        model_choice = data.get('model', 'rf')
        raw_df = pd.DataFrame([data])
        X_proc = preprocess_input(raw_df)

        chosen_model = art['rf_model'] if model_choice == 'rf' else art['dt_model']
        prediction = chosen_model.predict(X_proc)[0]
        probs_raw = chosen_model.predict_proba(X_proc)[0]
        classes = list(chosen_model.classes_)
        prob_dict = {cls: round(float(p) * 100, 2) for cls, p in zip(classes, probs_raw)}

        interventions = generate_interventions(data, prediction, prob_dict)

        return jsonify({
            'status': 'success',
            'model_used': 'Random Forest' if model_choice == 'rf' else 'Decision Tree',
            'prediction': prediction,
            'probabilities': prob_dict,
            'interventions': interventions
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'service': 'Student Academic Performance Prediction System',
        'models_loaded': artifacts is not None
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
