# Student Academic Performance Prediction System

> **A Machine Learning Decision-Support Web Application**  
> Based on the Research Paper:  
> **"Predicting Student Academic Performance Using Decision Tree and Random Forest Machine Learning Models"**  
> *Authors: Norjeane P. Centino & Rouge P. L. Saludes*  
> *University of Mindanao, Matina, Davao City, Philippines*

---

## 📌 Overview

This project is an end-to-end machine learning web application that predicts student academic performance levels (**Low: 55–64**, **Medium: 65–69**, and **High: 70–101**) by analyzing 19 multivariate academic, behavioral, and environmental factors.

The system operationalizes the findings of Centino & Saludes (University of Mindanao), providing educators with an early warning and decision-support tool to facilitate timely academic interventions before final examinations.

---

## 🌟 Key Features

1. **Individual Student Assessment & Predictor (`/`)**
   - Interactive assessment form evaluating the 19 student factors.
   - Model selection: **Random Forest (200 trees, 82.2% Accuracy)** or baseline **Decision Tree (72.4% Accuracy)**, or **Side-by-Side Comparison**.
   - Outputs class prediction (`Low`, `Medium`, `High`) and confidence probability distribution.
   - **Actionable Intervention Plan**: Automated recommendations highlighting attendance, study hours, tutoring, and behavioral support tailored to student risk levels.
   - Quick one-click testing presets (*High Achiever*, *Average Student*, *At-Risk Student*).

2. **Batch Cohort Evaluation (`/batch`)**
   - Upload student roster spreadsheets (`.csv`) to assess an entire class in seconds.
   - Instant cohort diagnostics: total evaluated, distribution breakdown, and percentage of at-risk students.
   - Downloadable annotated CSV file with predicted levels and risk flags.
   - Built-in sample CSV generator (`/sample-csv`) for immediate testing.

3. **Research Analytics & Model Comparison Dashboard (`/analytics`)**
   - Interactive visualization of **Figure 2** (Decision Tree vs. Random Forest Accuracy and Weighted F1-Score).
   - Interactive bar chart of **Table 6** (Top 10 Feature Importances: Attendance at 25.6%, Hours Studied at 16.5%, Previous Scores at 8.1%, etc.).
   - Full confusion matrix heatmaps comparing single tree baseline errors with ensemble stability.

4. **Academic Documentation & Study Background (`/about`)**
   - Complete background on research objectives, Kaggle dataset (6,607 records by *lainguyn123*), data preprocessing methodology, and literature references.

5. **REST API Endpoint (`/api/predict`)**
   - Programmatic JSON API to integrate prediction into school learning management systems (LMS).
   - `/health` endpoint for Railway status and uptime monitoring.

---

## 📊 Dataset & Model Benchmark Summary

| Evaluation Metric | Baseline Decision Tree | Random Forest Ensemble | Paper Gain (RF vs DT) |
| :--- | :---: | :---: | :---: |
| **Accuracy** | **72.39%** (or 72.84%) | **82.22%** (or 82.75%) | **+9.83%** |
| **Weighted Precision** | 72.45% | **85.02%** | **+12.57%** |
| **Weighted Recall** | 72.39% | **82.22%** | **+9.83%** |
| **Weighted F1-Score** | 72.39% | **81.56%** | **+9.17%** |

### Top Student Performance Factors:
1. **Attendance**: 25.62% importance
2. **Hours Studied**: 16.49% importance
3. **Previous Scores**: 8.10% importance
4. **Tutoring Sessions**: 4.08% importance
5. **Sleep Hours**: 3.97% importance
6. **Physical Activity**: 3.43% importance

---

## 💻 Running the System Locally

### Prerequisites
- Python 3.10+ (tested with Python 3.11, 3.12, 3.14)
- Git

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Verify / Train Models (Optional)
The models are already trained and saved in `models/model_artifacts.pkl`. To retrain from the dataset:
```bash
python train_models.py
```

### 3. Start the Web Server
```bash
python app.py
```
Open your browser and navigate to: **`http://localhost:5000`**

---

## 🚀 Step-by-Step Railway Deployment Guide

This project includes production-ready deployment configurations (`Dockerfile`, `Procfile`, `railway.json`, `runtime.txt`, `requirements.txt`).

### Option 1: Deploy via GitHub (Recommended & Free, 1-Click)

1. **Initialize Git and commit the project:**
   ```bash
   cd C:\Users\Lenovo\OneDrive\Desktop\IT20
   git init
   git add .
   git commit -m "Initial commit of Student Academic Performance Prediction System"
   ```

2. **Push to your GitHub account:**
   - Create a new repository on [GitHub](https://github.com/new) (e.g. `student-academic-predictor`).
   - Run:
     ```bash
     git remote add origin https://github.com/<YOUR-USERNAME>/student-academic-predictor.git
     git branch -M main
     git push -u origin main
     ```

3. **Deploy on Railway:**
   - Go to [Railway.app](https://railway.app) and sign in with GitHub.
   - Click **"New Project"** -> Select **"Deploy from GitHub repo"**.
   - Select your repository (`student-academic-predictor`).
   - Railway automatically detects `Dockerfile` / `railway.json` and begins building.

4. **Generate Public Live Domain:**
   - In the Railway dashboard, select your deployed service.
   - Go to **Settings** -> Scroll down to **Networking**.
   - Click **"Generate Domain"** (e.g. `https://student-academic-predictor-production.up.railway.app`).
   - Your system is now live on the internet!

---

### Option 2: Deploy directly via Railway CLI

If you have Node.js installed, deploy directly from your local terminal:

1. **Login to Railway:**
   ```bash
   npx @railway/cli login
   ```
   *(A browser tab will open to authorize your account)*

2. **Create and link a Railway project:**
   ```bash
   npx @railway/cli init
   ```
   *(Select "Empty Project" and enter a name)*

3. **Deploy the code:**
   ```bash
   npx @railway/cli up
   ```

4. **Generate a public URL:**
   ```bash
   npx @railway/cli domain
   ```

---

## 📁 Project Structure

```
IT20/
├── data/
│   └── StudentPerformanceFactors.csv   # Kaggle dataset (6,607 records)
├── models/
│   └── model_artifacts.pkl             # Trained RF, DT, and metrics cache
├── static/
│   ├── css/
│   │   └── custom.css                  # Custom animations & scrollbars
│   └── js/
│       └── main.js                     # Real-time form synchronization
├── templates/
│   ├── base.html                       # Global layout with Tailwind & Chart.js
│   ├── index.html                      # Individual student predictor form
│   ├── batch.html                      # Batch CSV cohort evaluation
│   ├── analytics.html                  # Model comparison & feature importances
│   └── about.html                      # University of Mindanao study details
├── app.py                              # Flask application & prediction API
├── train_models.py                     # ML pipeline & model trainer
├── Dockerfile                          # Production container configuration
├── Procfile                            # Gunicorn web process definition
├── railway.json                        # Railway build & deploy settings
├── runtime.txt                         # Python version specification
├── requirements.txt                    # Python package dependencies
├── .dockerignore                       # Docker build exclusions
├── .gitignore                          # Git exclusions
└── README.md                           # Documentation and deployment manual
```

---

## 📜 Academic Attribution

- **Paper Title:** Predicting Student Academic Performance Using Decision Tree and Random Forest Machine Learning Models
- **Authors:** Norjeane P. Centino & Rouge P. L. Saludes
- **Affiliation:** University of Mindanao, Matina, Davao City, Philippines
- **Dataset Source:** Kaggle (`lainguyn123`, Student Performance Factors)
