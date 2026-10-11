import sys
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

def train_baseline_model():
    """Trains the fraud detection model on training data."""
    train_path = os.path.join(os.path.dirname(__file__), "Dataset", "Fraud_transaction.csv")
    dataset = pd.read_csv(train_path)

    for col in dataset.columns:
        if dataset[col].dtype == 'object' or str(dataset[col].dtype).startswith('str'):
            dataset[col] = dataset[col].fillna('0')
        else:
            dataset[col] = dataset[col].fillna(0)

    Y = dataset['isFraud'].values.ravel()
    dataset.drop(['isFraud'], axis=1, inplace=True)

    label_encoders = {}
    for col in dataset.columns:
        if dataset[col].dtype == 'object' or str(dataset[col].dtype).startswith('str'):
            le = LabelEncoder()
            dataset[col] = le.fit_transform(dataset[col].astype(str))
            label_encoders[col] = le

    for col in dataset.columns:
        if dataset[col].dtype == 'object' or str(dataset[col].dtype).startswith('str'):
            dataset[col] = dataset[col].fillna('0')
        else:
            dataset[col] = dataset[col].fillna(0)

    X = dataset.values
    sc = StandardScaler()
    X = sc.fit_transform(X)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, Y)

    return model, sc, label_encoders

def predict_transaction(csv_file_path, model, sc, label_encoders):
    """Predicts fraud for transactions in the given CSV file."""
    test_df = pd.read_csv(csv_file_path)

    for col in test_df.columns:
        if test_df[col].dtype == 'object' or str(test_df[col].dtype).startswith('str'):
            test_df[col] = test_df[col].fillna('0')
        else:
            test_df[col] = test_df[col].fillna(0)

    proc_df = test_df.copy()
    for col, le in label_encoders.items():
        if col in proc_df.columns:
            known_classes = set(le.classes_)
            proc_df[col] = proc_df[col].astype(str).map(
                lambda s: le.transform([s])[0] if s in known_classes else -1
            )

    for col in proc_df.columns:
        if proc_df[col].dtype == 'object' or str(proc_df[col].dtype).startswith('str'):
            proc_df[col] = proc_df[col].fillna('0')
        else:
            proc_df[col] = proc_df[col].fillna(0)

    X_test = sc.transform(proc_df.values)
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)

    print("=" * 70)
    print(f"File Tested: {os.path.basename(csv_file_path)}")
    print("=" * 70)
    for i in range(len(preds)):
        tid = test_df.iloc[i].get('TransactionID', f'Tx #{i+1}')
        amt = test_df.iloc[i].get('TransactionAmt', 'N/A')
        prod = test_df.iloc[i].get('ProductCD', 'N/A')
        card1 = test_df.iloc[i].get('card1', 'N/A')
        card4 = test_df.iloc[i].get('card4', 'N/A')
        p_email = test_df.iloc[i].get('P_emaildomain', 'N/A')
        
        fraud_prob = probs[i][1] * 100
        result = "FRAUD (High Risk)" if preds[i] == 1 else "NORMAL (Legitimate)"

        print(f"Transaction ID : {tid}")
        print(f"Amount         : ${amt}")
        print(f"Product Code   : {prod}")
        print(f"Card / Network : {card1} ({card4})")
        print(f"Email Domain   : {p_email}")
        print(f"Fraud Risk     : {fraud_prob:.1f}%")
        print(f"CLASSIFICATION : >>> {result} <<<")
        print("-" * 70)

if __name__ == "__main__":
    print("Initializing Multi-Perspective Fraud Detection Model...")
    model, sc, label_encoders = train_baseline_model()
    print("Model ready!\n")

    if len(sys.argv) > 1:
        target_csv = sys.argv[1]
        predict_transaction(target_csv, model, sc, label_encoders)
    else:
        # Test both single normal and single fraud examples
        base_dir = os.path.dirname(__file__)
        normal_csv = os.path.join(base_dir, "Dataset", "sample_single_normal_transaction.csv")
        fraud_csv = os.path.join(base_dir, "Dataset", "sample_single_fraud_transaction.csv")

        print("[TEST 1: Testing Single Normal Real-Time Transaction]")
        predict_transaction(normal_csv, model, sc, label_encoders)

        print("\n[TEST 2: Testing Single Fraud Real-Time Transaction]")
        predict_transaction(fraud_csv, model, sc, label_encoders)
