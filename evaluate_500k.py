"""
Multi-Perspective Fraud Detection - Full 590,540 Rows Evaluation Script
========================================================================
This script trains and evaluates the Multi-Perspective Fraud Detection system
on the full 590,540 transaction benchmark dataset (Dataset/train_transaction.csv).
It generates performance metrics, confusion matrices, and ROC curves for
academic project reports and presentations.
"""

import time
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def run_full_evaluation():
    print("=" * 70)
    print("A Multi-Perspective Fraud Detection Method for Multi-Participant")
    print("               E-Commerce Transactions")
    print("       FULL 590,540 ROWS BENCHMARK EVALUATION PIPELINE")
    print("=" * 70)
    
    csv_path = os.path.join(os.path.dirname(__file__), "Dataset", "train_transaction.csv")
    if not os.path.exists(csv_path):
        print(f"Error: Could not find {csv_path}")
        return

    # 1. Load Data
    t0 = time.time()
    print(f"\n[Step 1/5] Loading full dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    load_time = time.time() - t0
    total_records = len(df)
    total_features = df.shape[1] - 1
    print(f"-> Loaded {total_records:,} records with {total_features} attributes in {load_time:.2f}s")
    
    # Class distribution
    fraud_counts = df['isFraud'].value_counts()
    normal_count = fraud_counts.get(0, 0)
    fraud_count = fraud_counts.get(1, 0)
    fraud_pct = (fraud_count / total_records) * 100
    print(f"-> Class Distribution: Normal = {normal_count:,} ({100-fraud_pct:.2f}%), Fraud = {fraud_count:,} ({fraud_pct:.2f}%)")

    # 2. Preprocess & Memory Optimization
    print("\n[Step 2/5] Performing multi-perspective feature encoding and memory optimization...")
    t_prep = time.time()
    for col in df.columns:
        if df[col].dtype == 'object' or str(df[col].dtype).startswith('str'):
            df[col] = df[col].fillna('0').astype('category').cat.codes
        elif df[col].dtype == 'float64':
            df[col] = df[col].fillna(0).astype(np.float32)
        else:
            df[col] = df[col].fillna(0).astype(np.int32)
            
    y = df['isFraud'].values
    X = df.drop(['isFraud'], axis=1).values
    prep_time = time.time() - t_prep
    print(f"-> Preprocessing completed in {prep_time:.2f}s")

    # 3. Process Mining (2D PCA Visualization)
    print("\n[Step 3/5] Generating Process Mining 2D PCA Behavioral Graph...")
    t_pca = time.time()
    # Sample 10,000 representative points for clean visualization without overplotting
    rng = np.random.RandomState(42)
    sample_idx = rng.choice(total_records, size=min(10000, total_records), replace=False)
    X_sample = StandardScaler().fit_transform(X[sample_idx])
    y_sample = y[sample_idx]
    
    pca = PCA(n_components=2, random_state=42)
    XX = pca.fit_transform(X_sample)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(XX[y_sample == 0, 0], XX[y_sample == 0, 1], label=f'Normal (0) - {np.sum(y_sample==0):,} pts', alpha=0.4, s=15, c='#1f77b4')
    plt.scatter(XX[y_sample == 1, 0], XX[y_sample == 1, 1], label=f'Fraud (1) - {np.sum(y_sample==1):,} pts', alpha=0.8, s=25, c='#d62728')
    plt.legend(frameon=True, facecolor='white', framealpha=0.9)
    plt.title("Process Mining User Behaviour Graph\n(Full 590k Dataset PCA Projection)", fontsize=13, fontweight='bold')
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.grid(True, linestyle='--', alpha=0.5)
    pca_img_path = "full_500k_pca_process_mining.png"
    plt.savefig(pca_img_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"-> Process Mining PCA Graph saved: {pca_img_path} ({time.time()-t_pca:.2f}s)")

    # 4. Train/Test Split (80% Train, 20% Test with Stratification)
    print("\n[Step 4/5] Splitting into 80% Training and 20% Testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"-> Training Set: {len(X_train):,} transactions")
    print(f"-> Testing Set : {len(X_test):,} transactions")

    # 5. Model Training & Evaluation
    print("\n[Step 5/5] Training Multi-Perspective Machine Learning Models...")
    
    # Model A: Proposed SVM (with Standard Scaling on stratified subset due to O(N^2) complexity)
    print("  -> Training Proposed Polynomial Kernel SVM on scaled stratified subset...")
    svm_subset_idx = rng.choice(len(X_train), size=min(4000, len(X_train)), replace=False)
    svm_test_idx = rng.choice(len(X_test), size=min(2000, len(X_test)), replace=False)
    
    sc_svm = StandardScaler()
    X_svm_train = sc_svm.fit_transform(X_train[svm_subset_idx])
    X_svm_test = sc_svm.transform(X_test[svm_test_idx])
    
    svm_cls = svm.SVC(kernel='poly', C=1.0, gamma='scale', degree=3, probability=True, random_state=42)
    t_svm = time.time()
    svm_cls.fit(X_svm_train, y_train[svm_subset_idx])
    svm_train_time = time.time() - t_svm
    svm_preds = svm_cls.predict(X_svm_test)
    svm_probs = svm_cls.predict_proba(X_svm_test)[:, 1]
    
    svm_acc = accuracy_score(y_test[svm_test_idx], svm_preds) * 100
    svm_prec = precision_score(y_test[svm_test_idx], svm_preds, zero_division=0) * 100
    svm_rec = recall_score(y_test[svm_test_idx], svm_preds, zero_division=0) * 100
    svm_f1 = f1_score(y_test[svm_test_idx], svm_preds, zero_division=0) * 100
    svm_auc = roc_auc_score(y_test[svm_test_idx], svm_probs) * 100
    print(f"     Proposed SVM trained in {svm_train_time:.2f}s -> Acc: {svm_acc:.2f}%, Prec: {svm_prec:.2f}%, Rec: {svm_rec:.2f}%, F1: {svm_f1:.2f}%, AUC: {svm_auc:.2f}%")

    # Model B: Extension Random Forest on ALL 472,432 training samples
    print("  -> Training Extension Random Forest on ALL 472,432 training samples (50 trees, n_jobs=-1)...")
    t_rf = time.time()
    rf = RandomForestClassifier(n_estimators=50, max_depth=20, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_train_time = time.time() - t_rf
    print(f"     Extension Random Forest trained in {rf_train_time:.2f}s!")

    print("  -> Evaluating Random Forest on ALL 118,108 test transactions...")
    t_eval = time.time()
    rf_preds = rf.predict(X_test)
    rf_probs = rf.predict_proba(X_test)[:, 1]
    print(f"     Inference completed in {time.time()-t_eval:.2f}s")
    
    rf_acc = accuracy_score(y_test, rf_preds) * 100
    rf_prec = precision_score(y_test, rf_preds, zero_division=0) * 100
    rf_rec = recall_score(y_test, rf_preds, zero_division=0) * 100
    rf_f1 = f1_score(y_test, rf_preds, zero_division=0) * 100
    rf_auc = roc_auc_score(y_test, rf_probs) * 100
    cm = confusion_matrix(y_test, rf_preds)

    # 6. Generate Charts for Academic Report
    # Chart 1: Performance Bar Chart
    plt.figure(figsize=(8, 5))
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    svm_vals = [svm_acc, svm_prec, svm_rec, svm_f1, svm_auc]
    rf_vals = [rf_acc, rf_prec, rf_rec, rf_f1, rf_auc]
    x = np.arange(len(metrics_names))
    width = 0.35
    
    plt.bar(x - width/2, svm_vals, width, label='Proposed SVM (Poly Kernel)', color='#3498db')
    plt.bar(x + width/2, rf_vals, width, label='Extension Random Forest (Full 590k)', color='#2ecc71')
    plt.ylabel('Percentage (%)', fontsize=11)
    plt.title('Algorithm Performance Comparison on 590,540 Rows Dataset', fontsize=12, fontweight='bold')
    plt.xticks(x, metrics_names)
    plt.ylim(0, 110)
    for i in range(len(metrics_names)):
        plt.text(x[i] - width/2, svm_vals[i] + 1.5, f"{svm_vals[i]:.1f}%", ha='center', fontsize=9)
        plt.text(x[i] + width/2, rf_vals[i] + 1.5, f"{rf_vals[i]:.1f}%", ha='center', fontsize=9, fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.savefig('full_500k_metrics_comparison.png', dpi=200, bbox_inches='tight')
    plt.close()

    # Chart 2: ROC-AUC Curve
    fpr, tpr, _ = roc_curve(y_test, rf_probs)
    plt.figure(figsize=(7, 5.5))
    plt.plot(fpr, tpr, color='#e74c3c', lw=2.5, label=f'Random Forest ROC (AUC = {rf_auc:.2f}%)')
    plt.plot([0, 1], [0, 1], color='#7f8c8d', lw=1.5, linestyle='--', label='Chance Line (AUC = 50%)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=11)
    plt.ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=11)
    plt.title('Receiver Operating Characteristic (ROC) Curve\nEvaluated on 118,108 Test Transactions', fontsize=12, fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig('full_500k_roc_curve.png', dpi=200, bbox_inches='tight')
    plt.close()

    # Chart 3: Confusion Matrix
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix on 118,108 Test Transactions', fontsize=12, fontweight='bold')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Normal (0)', 'Fraud (1)'])
    plt.yticks(tick_marks, ['Normal (0)', 'Fraud (1)'])
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black", fontsize=11, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('full_500k_confusion_matrix.png', dpi=200, bbox_inches='tight')
    plt.close()

    print("\n" + "=" * 70)
    print("FINAL EVALUATION REPORT ON 590,540 TRANSACTIONS:")
    print("=" * 70)
    print(f"Total Transactions : {total_records:,}")
    print(f"Normal Cases       : {normal_count:,} ({100-fraud_pct:.2f}%)")
    print(f"Fraudulent Cases   : {fraud_count:,} ({fraud_pct:.2f}%)")
    print(f"Input Attributes   : {total_features}")
    print(f"Training Split     : {len(X_train):,} transactions")
    print(f"Testing Split      : {len(X_test):,} transactions")
    print("-" * 70)
    print("EXTENSION RANDOM FOREST PERFORMANCE (FULL 590K BENCHMARK):")
    print(f"  * Accuracy       : {rf_acc:.2f}%")
    print(f"  * Precision      : {rf_prec:.2f}%")
    print(f"  * Recall         : {rf_rec:.2f}%")
    print(f"  * F1-Score       : {rf_f1:.2f}%")
    print(f"  * ROC-AUC Score  : {rf_auc:.2f}%")
    print(f"  * Confusion Matrix:")
    print(f"      True Normal  (TN) : {cm[0, 0]:,}   False Fraud (FP) : {cm[0, 1]:,}")
    print(f"      False Normal (FN) : {cm[1, 0]:,}   True Fraud  (TP) : {cm[1, 1]:,}")
    print("-" * 70)
    print("SAVED REPORT ARTIFACTS:")
    print("  1. full_500k_pca_process_mining.png    -> Process Mining Behavioral Graph")
    print("  2. full_500k_metrics_comparison.png   -> Algorithm Comparison Chart")
    print("  3. full_500k_roc_curve.png            -> ROC-AUC Curve")
    print("  4. full_500k_confusion_matrix.png     -> Heatmap Confusion Matrix")
    print(f"Total Pipeline Runtime: {time.time()-t0:.2f} seconds")
    print("=" * 70)

if __name__ == '__main__':
    run_full_evaluation()
