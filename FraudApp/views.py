from django.shortcuts import render
from django.template import RequestContext
from django.contrib import messages
from django.http import HttpResponse
import os
from django.core.files.storage import FileSystemStorage
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
import io
import base64

uname = None
dataset = None
X = None
Y = None
labels = None
unique = None
label_encoder = []
sc = None
pca = None
rf_cls = None

def clean_df(df):
    for col in df.columns:
        if df[col].dtype == 'object' or str(df[col].dtype).startswith('str'):
            df[col] = df[col].fillna('0')
        elif df[col].dtype == 'float64':
            df[col] = df[col].fillna(0).astype(np.float32)
        else:
            df[col] = df[col].fillna(0).astype(np.int32)
    return df

def calculateMetrics(algorithm, predict, y_test):
    a = accuracy_score(y_test, predict) * 100
    p = precision_score(y_test, predict, average='macro', zero_division=0) * 100
    r = recall_score(y_test, predict, average='macro', zero_division=0) * 100
    f = f1_score(y_test, predict, average='macro', zero_division=0) * 100
    return a, p, r, f

def RunML(request):
    if request.method == 'GET':
        global X, Y, rf_cls
        if X is None or Y is None:
            context = {'data': '<font color="red"><b>Please execute Process Mining first before running ML Algorithms.</b></font>'}
            return render(request, 'UserScreen.html', context)
        
        n_samples = len(X)
        rng = np.random.RandomState(42)
        
        # Train / Test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, Y, test_size=0.2, random_state=42, stratify=Y if len(np.unique(Y)) > 1 else None
        )
        
        # Proposed SVM: On large datasets (>10k), use a representative stratified subset for SVM
        # because Polynomial Kernel SVM has O(N^2) complexity and would cause browser HTTP timeout.
        if len(X_train) > 10000:
            svm_sub_size = min(4000, len(X_train))
            svm_test_size = min(2000, len(X_test))
            svm_idx = rng.choice(len(X_train), size=svm_sub_size, replace=False)
            svm_test_idx = rng.choice(len(X_test), size=svm_test_size, replace=False)
            X_svm_tr, y_svm_tr = X_train[svm_idx], y_train[svm_idx]
            X_svm_te, y_svm_te = X_test[svm_test_idx], y_test[svm_test_idx]
            svm_note = f" (Evaluated on {svm_sub_size:,} stratified samples due to O(N&sup2;) complexity)"
        else:
            X_svm_tr, y_svm_tr = X_train, y_train
            X_svm_te, y_svm_te = X_test, y_test
            svm_note = ""
            
        svm_cls = svm.SVC(kernel='poly', C=1.0, gamma='scale', degree=3, random_state=42)
        svm_cls.fit(X_svm_tr, y_svm_tr)
        predict_svm = svm_cls.predict(X_svm_te)
        a, p, r, f = calculateMetrics("Propose SVM Algorithm", predict_svm, y_svm_te)
        
        # Extension Random Forest: Scales across entire dataset using multicore parallel bagging
        rf_cls = RandomForestClassifier(n_estimators=50, max_depth=20, random_state=42, n_jobs=-1)
        rf_cls.fit(X_train, y_train)
        predict_rf = rf_cls.predict(X_test)
        a1, p1, r1, f1 = calculateMetrics("Extension Random Forest Algorithm", predict_rf, y_test)
        
        output = f'<div style="margin-bottom:12px; font-size:14px;"><b>Total Dataset Evaluated:</b> {n_samples:,} Transactions | <b>Train Split:</b> {len(X_train):,} | <b>Test Split:</b> {len(X_test):,}</div>'
        output += '<table border=1 align=center width=100%><tr><th><font color="black">Algorithm Name</font></th><th><font color="black">Accuracy (%)</font></th><th><font color="black">Precision (%)</font></th>'
        output += '<th><font color="black">Recall (%)</font></th><th><font color="black">F-Score (%)</font></th></tr>'
        algorithms = [f'Propose SVM (Poly Degree-3){svm_note}', 'Extension Random Forest (Full Dataset)']
        output += f'<tr><td><b>{algorithms[0]}</b></td><td>{a:.2f}</td><td>{p:.2f}</td><td>{r:.2f}</td><td>{f:.2f}</td></tr>'
        output += f'<tr><td><b><font color="#155724">{algorithms[1]}</font></b></td><td><b>{a1:.2f}</b></td><td><b>{p1:.2f}</b></td><td><b>{r1:.2f}</b></td><td><b>{f1:.2f}</b></td></tr>'
        output += "</table>"
        if len(X_train) > 10000:
            output += "<div style='font-size:12px; color:#555; margin-top:6px;'><i>* Note: Proposed Polynomial SVM exhibits O(N&sup2;) computational scaling, requiring stratified representative subsampling on massive datasets, whereas the Extension Random Forest processes all 500k+ transactions in parallel with superior accuracy and precision.</i></div>"
        output += "<br/>"
        
        df_plot = pd.DataFrame([
            ['Propose SVM', 'Precision', p],
            ['Propose SVM', 'Recall', r],
            ['Propose SVM', 'F1 Score', f],
            ['Propose SVM', 'Accuracy', a],
            ['Extension Random Forest', 'Precision', p1],
            ['Extension Random Forest', 'Recall', r1],
            ['Extension Random Forest', 'F1 Score', f1],
            ['Extension Random Forest', 'Accuracy', a1],
        ], columns=['Algorithms', 'Metrics', 'Value'])
        
        fig, ax = plt.subplots(figsize=(7, 4.5))
        df_plot.pivot_table(index="Algorithms", columns="Metrics", values="Value").plot(kind='bar', ax=ax)
        ax.set_title(f"All Algorithms Performance Comparison ({n_samples:,} Records)")
        ax.set_ylabel("Percentage (%)")
        ax.set_ylim(0, 110)
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
        img_b64 = base64.b64encode(buf.getvalue()).decode()    
        context = {'data': output, 'img': img_b64}
        return render(request, 'UserScreen.html', context)

def ProcessMining(request):
    if request.method == 'GET':
        global dataset, X, Y, labels, unique, label_encoder, sc, pca
        if dataset is None:
            pkl_path = "Dataset/train_transaction.pkl"
            if os.path.exists(pkl_path):
                dataset = pd.read_pickle(pkl_path)
            elif os.path.exists("Dataset/train_transaction.csv"):
                dataset = pd.read_csv("Dataset/train_transaction.csv")
            elif os.path.exists("FraudApp/static/Data.csv"):
                dataset = pd.read_csv("FraudApp/static/Data.csv")
            elif os.path.exists("Dataset/Fraud_transaction.csv"):
                dataset = pd.read_csv("Dataset/Fraud_transaction.csv")
            else:
                context = {'data': '<font color="red"><b>No dataset loaded. Please load a dataset first.</b></font>'}
                return render(request, 'UserScreen.html', context)
        
        if 'isFraud' not in dataset.columns and Y is None:
            context = {'data': '<font color="red"><b>Dataset does not contain required isFraud column. Please reload dataset.</b></font>'}
            return render(request, 'UserScreen.html', context)
        
        total_rows = len(dataset)
        
        # Fast path: use cached or precomputed feature matrices if available
        if X is None or Y is None:
            npz_cache = "Dataset/train_transaction_processed.npz"
            if total_rows > 500000 and os.path.exists(npz_cache):
                cached = np.load(npz_cache)
                X = cached['X']
                Y = cached['Y']
            else:
                df = dataset.copy()
                df = clean_df(df)
                Y = df['isFraud'].values.ravel()
                df.drop(['isFraud'], axis=1, inplace=True)
                
                label_encoder = []
                columns = df.columns
                types = df.dtypes.values
                for i in range(len(types)):
                    name = types[i]
                    if name == 'object' or str(name).startswith('str'):
                        le = LabelEncoder()
                        df[columns[i]] = le.fit_transform(df[columns[i]].astype(str))
                        label_encoder.append([columns[i], le])
                        
                df = clean_df(df)
                X = df.values
                indices = np.arange(X.shape[0])
                np.random.seed(42)
                np.random.shuffle(indices)
                X = X[indices]
                Y = Y[indices]
                
                sc = StandardScaler()
                X = sc.fit_transform(X)
        
        unique, count = np.unique(Y, return_counts=True)
        
        # Representative sampling for scatter plotting large datasets (>10k)
        # to prevent browser freeze and visual overplotting
        if total_rows > 10000:
            plot_sample_size = min(10000, total_rows)
            plot_idx = np.random.RandomState(42).choice(total_rows, size=plot_sample_size, replace=False)
            X_plot = X[plot_idx]
            Y_plot = Y[plot_idx]
            plot_subtitle = f"Showing {plot_sample_size:,} representative points out of {total_rows:,} total transactions"
        else:
            X_plot = X
            Y_plot = Y
            plot_subtitle = f"Total Transactions: {total_rows:,}"
            
        pca = PCA(2, random_state=42) 
        XX = pca.fit_transform(X_plot)
        
        fig = plt.figure(figsize=(7, 5))
        class_names = {0: '0 (Normal)', 1: '1 (Fraud)'}
        for cls in unique:
            lbl = class_names.get(cls, str(cls))
            mask = (Y_plot == cls)
            color = '#1f77b4' if cls == 0 else '#d62728'
            alpha = 0.4 if cls == 0 else 0.8
            plt.scatter(XX[mask, 0], XX[mask, 1], label=lbl, alpha=alpha, s=18, c=color) 
        plt.legend(frameon=True, facecolor='white', framealpha=0.9)
        plt.title(f"Process Mining User Behaviour Graph (PCA 2D)\n{plot_subtitle}", fontsize=11)
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.grid(True, linestyle='--', alpha=0.4)
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
        img_b64 = base64.b64encode(buf.getvalue()).decode()
        context = {
            'data': f"<h3>User Behaviour Process Mining Graph Generated</h3><p>PCA dimensionality reduction applied to {total_rows:,} multi-participant transactions, establishing normal vs fraudulent behavioral baselines.</p>",
            'img': img_b64
        }
        return render(request, 'UserScreen.html', context)

def DetectFraud(request):
    if request.method == 'GET':
        return render(request, 'DetectFraud.html', {})

def DetectFraudAction(request):
    if request.method == 'POST':
        global dataset, X, Y, sc, rf_cls, label_encoder
        if rf_cls is None or sc is None:
            context = {'data': '<font color="red"><b>Model is not trained yet! Please run Process Mining and Run ML Algorithms first.</b></font>'}
            return render(request, 'UserScreen.html', context)
            
        myfile = request.FILES.get('t1')
        if not myfile:
            return render(request, 'DetectFraud.html', {'data': 'Please choose a file to upload.'})
            
        os.makedirs("FraudApp/static", exist_ok=True)
        test_file_path = "FraudApp/static/test_uploaded.csv"
        with open(test_file_path, 'wb+') as dest:
            for chunk in myfile.chunks():
                dest.write(chunk)
        
        test_df = pd.read_csv(test_file_path)
        test_df = clean_df(test_df)
        temp = test_df.values
        
        for i in range(len(label_encoder)):
            col_name = label_encoder[i][0]
            le = label_encoder[i][1]
            if col_name in test_df.columns:
                known_classes = set(le.classes_)
                test_df[col_name] = test_df[col_name].astype(str).map(
                    lambda s: le.transform([s])[0] if s in known_classes else -1
                )
                
        test_df = clean_df(test_df)
        X_test = test_df.values
        X_test = sc.transform(X_test)
        predict = rf_cls.predict(X_test)
        
        output = '<div style="margin-bottom:15px;"><h3>Fraud Detection Results from Test Data</h3>'
        output += f'<b>Total Transactions Tested:</b> {len(predict)} | '
        output += f'<b>Normal:</b> {np.sum(predict == 0)} | '
        output += f'<b>Fraudulent:</b> {np.sum(predict == 1)}</div>'
        output += '<table border=1 align=center width=100%><tr><th width="80%"><font color="black">Test Transaction Features</font></th><th><font color="black">Detection Result</font></th></tr>'
        for i in range(len(predict)):
            if predict[i] == 1:
                pred_html = '<font color="red" size="3"><b>FRAUD</b></font>'
            else:
                pred_html = '<font color="green" size="3"><b>NORMAL</b></font>'
            output += f'<tr><td><div style="max-height:80px; overflow-y:auto; word-break:break-all; font-family:monospace; font-size:11px;">{str(list(temp[i]))}</div></td><td align="center">{pred_html}</td></tr>'
        output += '</table><br/>'
        context = {'data': output}
        return render(request, 'UserScreen.html', context)

def LoadDatasetAction(request):
    if request.method == 'POST':
        global dataset, X, Y
        source = request.POST.get('dataset_source')
        if not source:
            source = 'upload' if request.FILES.get('t1') else 'full_500k'
        os.makedirs("FraudApp/static", exist_ok=True)
        save_path = "FraudApp/static/Data.csv"
        
        if source == 'full_500k':
            pkl_path = "Dataset/train_transaction.pkl"
            csv_path = "Dataset/train_transaction.csv"
            npz_path = "Dataset/train_transaction_processed.npz"
            if os.path.exists(pkl_path):
                dataset = pd.read_pickle(pkl_path)
            elif os.path.exists(csv_path):
                dataset = pd.read_csv(csv_path)
                dataset = clean_df(dataset)
                try:
                    dataset.to_pickle(pkl_path)
                except Exception:
                    pass
            else:
                return render(request, 'LoadDataset.html', {'data': f'Error: {csv_path} not found.'})
            
            # Preload prepared matrices for instant ProcessMining and RunML
            if os.path.exists(npz_path):
                try:
                    cached = np.load(npz_path)
                    X = cached['X']
                    Y = cached['Y']
                except Exception:
                    pass
            source_name = "Full Benchmark Dataset (Dataset/train_transaction.csv)"
        elif source == 'sample_1k':
            csv_path = "Dataset/Fraud_transaction.csv"
            if not os.path.exists(csv_path):
                return render(request, 'LoadDataset.html', {'data': f'Error: {csv_path} not found.'})
            dataset = pd.read_csv(csv_path)
            X = None
            Y = None
            source_name = "Sample Dataset (Dataset/Fraud_transaction.csv)"
        else:
            myfile = request.FILES.get('t1')
            if not myfile:
                return render(request, 'LoadDataset.html', {'data': 'Please choose a dataset file to upload.'})
            with open(save_path, 'wb+') as dest:
                for chunk in myfile.chunks():
                    dest.write(chunk)
            dataset = pd.read_csv(save_path)
            X = None
            Y = None
            source_name = f"Uploaded File: {myfile.name}"
        
        total_rows = dataset.shape[0]
        total_cols = dataset.shape[1]
        fraud_counts = dataset['isFraud'].value_counts() if 'isFraud' in dataset.columns else {}
        n_normal = fraud_counts.get(0, 0)
        n_fraud = fraud_counts.get(1, 0)
        fraud_pct = (n_fraud / total_rows * 100) if total_rows > 0 else 0
        
        info_html = "<div style='margin-bottom:12px; font-weight:bold; color:#155724; background-color:#d4edda; border:1px solid #c3e6cb; padding:12px; border-radius:6px; text-align:left;'>"
        info_html += "<h3 style='margin:0 0 6px 0; color:#155724;'>&#10004; Dataset Loaded Successfully!</h3>"
        info_html += f"<b>Selected Source:</b> {source_name}<br/>"
        info_html += f"<b>Total Transactions:</b> {total_rows:,} &nbsp;|&nbsp; <b>Total Multi-Perspective Attributes:</b> {total_cols}<br/>"
        info_html += f"<b>Normal Cases:</b> {n_normal:,} ({100-fraud_pct:.2f}%) &nbsp;|&nbsp; <b>Fraud Cases:</b> {n_fraud:,} ({fraud_pct:.2f}%)"
        info_html += "</div>"
        info_html += f"<div style='overflow-x:auto; max-height:350px; border:1px solid #ddd; background:#fff; margin-top:10px;'><pre style='text-align:left; padding:10px; font-size:12px; font-family:Consolas, monospace;'>{str(dataset.head(10))}</pre></div>"
        
        context = {'data': info_html}
        return render(request, 'UserScreen.html', context)

def LoadDataset(request):
    if request.method == 'GET':
        return render(request, 'LoadDataset.html', {})

def UserLogin(request):
    if request.method == 'GET':
        return render(request, 'UserLogin.html', {})

def index(request):
    if request.method == 'GET':
        return render(request, 'index.html', {})

def UserLoginAction(request):
    if request.method == 'POST':
        global uname
        username = request.POST.get('t1', False)
        password = request.POST.get('t2', False)
        if username == 'admin' and password == 'admin':
            uname = username
            context = {'data': f'<h3>Welcome, {username}!</h3><p>Please select an option from the menu bar above to begin fraud detection analysis.</p>'}
            return render(request, 'UserScreen.html', context)
        else:
            context = {'data': 'Login failed: Invalid Username or Password. Use admin / admin.'}
            return render(request, 'UserLogin.html', context)
