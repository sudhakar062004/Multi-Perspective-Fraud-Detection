# A Multi-perspective Fraud Detection Method for Multi-Participant E-commerce Transactions

## Overview
E-commerce platforms involve complex interactions among multiple participants (buyers, sellers, and financial intermediaries). Fraudulent activities often span across accounts, devices, transaction timing, and participant interactions. This project implements a **Multi-perspective Fraud Detection System** built on **Django** and powered by **Machine Learning** algorithms (Proposed Polynomial SVM and Extension Random Forest Classifier) with **Process Mining (PCA Dimensionality Reduction)** visualization.

---

## Key Features & Modules
1. **User Authentication Module**:
   - Secure login portal for administrative access.
   - Credentials: Username `admin` / Password `admin`.

2. **Dataset Loader & Processing**:
   - Upload and parse e-commerce transaction data (`Dataset/Fraud_transaction.csv`).
   - Handles multi-type data, categorical label encoding, and feature scaling.

3. **Process Mining (User Behaviour Analysis)**:
   - Applies Principal Component Analysis (PCA) 2D projection on multi-dimensional transaction features.
   - Generates behavioral scatter plots clustering **Normal (0)** vs **Fraud (1)** transaction patterns.

4. **Machine Learning Algorithms (Proposed SVM & Extension Random Forest)**:
   - **Proposed SVM Classifier**: Polynomial kernel SVM with degree-3 non-linear mapping.
   - **Extension Random Forest Classifier**: High-accuracy ensemble learning algorithm.
   - Computes evaluation metrics: **Accuracy, Precision, Recall, and F1-Score**.
   - Generates visual comparison bar charts rendered directly in the web UI.

5. **Fraud Detection Engine**:
   - Upload unseen transactions (`Dataset/testData.csv`).
   - Automatically classifies each transaction as either **NORMAL** or **FRAUD**.

---

## Directory Structure
```
.
├── Dataset/
│   ├── Fraud_transaction.csv     # Training dataset (1,100 records, 394 attributes)
│   └── testData.csv              # Test dataset (8 sample transactions)
├── Fraud/
│   ├── settings.py               # Django configuration
│   ├── urls.py                   # Root URL routing
│   ├── wsgi.py                   # WSGI web server interface
│   └── __init__.py
├── FraudApp/
│   ├── migrations/               # Database migrations
│   ├── static/
│   │   ├── images/               # Template graphics & UI banners
│   │   ├── files/                # Project supplementary assets
│   │   └── style.css             # UI design & styling
│   ├── templates/
│   │   ├── index.html            # Landing / Home page
│   │   ├── UserLogin.html        # Admin login page
│   │   ├── UserScreen.html       # Main analysis dashboard
│   │   ├── LoadDataset.html      # Dataset upload page
│   │   └── DetectFraud.html      # Test data upload page
│   ├── views.py                  # Core backend & ML logic
│   ├── urls.py                   # App URL routing
│   ├── models.py                 # Data models
│   └── tests.py                  # Full automated test suite
├── db.sqlite3                    # SQLite database
├── manage.py                     # Django management script
├── requirements.txt              # Python package dependencies
├── run.bat                       # One-click Windows runner
├── test.py                       # Standalone ML pipeline test script
└── README.md                     # Documentation
```

---

## Prerequisites & Installation

### 1. Requirements
- Python 3.8 to Python 3.13
- Required packages (installable via pip):
  ```bash
  pip install -r requirements.txt
  ```

---

## How to Run

### Method 1: One-Click Run (Windows)
Double-click `run.bat` in this folder. It will:
1. Verify required libraries.
2. Run database migrations.
3. Automatically open your default web browser at `http://127.0.0.1:8000/`.
4. Launch the local development server.

### Method 2: Manual Command Line
```bash
# 1. Apply migrations
python manage.py migrate

# 2. Start the Django development server
python manage.py runserver 127.0.0.1:8000
```
Then visit `http://127.0.0.1:8000/` in your browser.

---

## How to Test the Project

### Automated Django Test Suite
Run the built-in test suite covering all 8 workflow stages:
```bash
python manage.py test
```

### Standalone Python ML Script
Run the standalone data science pipeline:
```bash
python test.py
```

---

## Step-by-Step User Workflow
1. Open `http://127.0.0.1:8000/`.
2. Click **User Login**, enter:
   - **Username**: `admin`
   - **Password**: `admin`
3. Click **Load & Process Dataset**:
   - Choose **Full Benchmark Dataset (590,540 rows)** &mdash; `Dataset/train_transaction.csv`
   - Or choose **Sample Dataset (1,100 rows)** &mdash; `Dataset/Fraud_transaction.csv`
   - Click **Load & Process Dataset**
4. Click **Process Mining** to generate and view the 2D PCA User Behaviour Graph.
5. Click **Run ML Algorithms** to train the Proposed SVM and Extension Random Forest models and inspect comparative accuracy metrics and performance graphs.
6. Click **Detect Fraud**, upload `sample_single_normal_transaction.csv` or `sample_single_fraud_transaction.csv`, and submit to see real-time classification predictions (**Normal** / **Fraud**) for each transaction.

---

## Full 590,540 Rows Benchmark Evaluation Results

To comprehensively evaluate the multi-perspective fraud detection system on industrial-scale big data, the system was benchmarked across the entire IEEE-CIS dataset (**590,540 transactions with 393 attributes**).

### 1. Dataset Breakdown
- **Total Records:** 590,540
- **Normal Transactions (Class 0):** 569,877 (96.50%)
- **Fraudulent Transactions (Class 1):** 20,663 (3.50%)
- **Training Set (80%):** 472,432 transactions
- **Testing Set (20%):** 118,108 transactions

### 2. Comparative Algorithm Performance
| Metric | Proposed SVM (Polynomial Degree-3) | Extension Random Forest (Full 590k Dataset) |
| :--- | :---: | :---: |
| **Accuracy** | 96.80% | **97.71%** |
| **Precision** | 45.83% | **93.49%** |
| **Recall** | 17.74% | **37.16%** |
| **F1-Score** | 25.58% | **53.19%** |
| **ROC-AUC Score** | 71.64% | **90.93%** |
| **Training Time** | O(N²) quadratic scaling | **30.22s (Multi-core Parallel Bagging)** |

### 3. Confusion Matrix on 118,108 Test Transactions
```
                      Predicted Normal    Predicted Fraud
Actual Normal (0)          113,868               107
Actual Fraud  (1)            2,597             1,536
```
- **False Positive Rate:** Only 0.09% (107 false alarms out of 113,975 genuine transactions)
- **High Precision:** 93.49% of flagged transactions are genuine fraud cases.

### 4. Running the Standalone 500k Evaluation Script
```bash
python evaluate_500k.py
```
This automatically produces 4 publication-quality evaluation graphs:
1. `full_500k_pca_process_mining.png` (PCA Behavioral Graph on 590k rows)
2. `full_500k_metrics_comparison.png` (Algorithm Comparative Bar Chart)
3. `full_500k_roc_curve.png` (ROC-AUC Curve)
4. `full_500k_confusion_matrix.png` (Heatmap Confusion Matrix)

