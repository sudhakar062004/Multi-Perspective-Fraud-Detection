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
        else:
            df[col] = df[col].fillna(0)
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
        
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        
        svm_cls = svm.SVC(kernel='poly', C=3.0, gamma=0.25, tol=0.1, degree=3)
        svm_cls.fit(X_train, y_train)
        predict_svm = svm_cls.predict(X_test)
        a, p, r, f = calculateMetrics("Propose SVM Algorithm", predict_svm, y_test)
        
        rf_cls = RandomForestClassifier(random_state=42)
        rf_cls.fit(X_train, y_train)
        predict_rf = rf_cls.predict(X_test)
        a1, p1, r1, f1 = calculateMetrics("Extension Random Forest Algorithm", predict_rf, y_test)
        
        output = ''
        output += '<table border=1 align=center width=100%><tr><th><font color="black">Algorithm Name</font></th><th><font color="black">Accuracy (%)</font></th><th><font color="black">Precision (%)</font></th>'
        output += '<th><font color="black">Recall (%)</font></th><th><font color="black">F-Score (%)</font></th></tr>'
        algorithms = ['Propose SVM', 'Extension Random Forest']
        output += f'<tr><td><b>{algorithms[0]}</b></td><td>{a:.2f}</td><td>{p:.2f}</td><td>{r:.2f}</td><td>{f:.2f}</td></tr>'
        output += f'<tr><td><b>{algorithms[1]}</b></td><td>{a1:.2f}</td><td>{p1:.2f}</td><td>{r1:.2f}</td><td>{f1:.2f}</td></tr>'
        output += "</table><br/>"
        
        df = pd.DataFrame([
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
        df.pivot_table(index="Algorithms", columns="Metrics", values="Value").plot(kind='bar', ax=ax)
        ax.set_title("All Algorithms Performance Comparison Graph")
        ax.set_ylabel("Percentage (%)")
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
            if os.path.exists("FraudApp/static/Data.csv"):
                dataset = pd.read_csv("FraudApp/static/Data.csv")
            elif os.path.exists("Dataset/Fraud_transaction.csv"):
                dataset = pd.read_csv("Dataset/Fraud_transaction.csv")
            else:
                context = {'data': '<font color="red"><b>No dataset loaded. Please upload a dataset first.</b></font>'}
                return render(request, 'UserScreen.html', context)
        
        if 'isFraud' not in dataset.columns:
            if os.path.exists("Dataset/Fraud_transaction.csv"):
                dataset = pd.read_csv("Dataset/Fraud_transaction.csv")
            else:
                context = {'data': '<font color="red"><b>Dataset does not contain required isFraud column. Please reload dataset.</b></font>'}
                return render(request, 'UserScreen.html', context)
        
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
        pca = PCA(2) 
        XX = pca.fit_transform(X)
        
        fig = plt.figure(figsize=(7, 5))
        class_names = {0: '0 (Normal)', 1: '1 (Fraud)'}
        for cls in unique:
            lbl = class_names.get(cls, str(cls))
            plt.scatter(XX[Y == cls, 0], XX[Y == cls, 1], label=lbl, alpha=0.7) 
        plt.legend()
        plt.title("Process Mining User Behaviour Graph (PCA 2D)")
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
        img_b64 = base64.b64encode(buf.getvalue()).decode()
        context = {'data': "<h3>User Behaviour Process Mining Graph Generated</h3><p>PCA dimensionality reduction projection showing distinct clusters of Normal vs Fraud user behaviours.</p>", 'img': img_b64}
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
        myfile = request.FILES.get('t1')
        if not myfile:
            return render(request, 'LoadDataset.html', {'data': 'Please choose a dataset file.'})
        os.makedirs("FraudApp/static", exist_ok=True)
        save_path = "FraudApp/static/Data.csv"
        with open(save_path, 'wb+') as dest:
            for chunk in myfile.chunks():
                dest.write(chunk)
        dataset = pd.read_csv(save_path)
        
        info_html = f"<div style='margin-bottom:12px; font-weight:bold; color:#155724; background-color:#d4edda; border:1px solid #c3e6cb; padding:10px; border-radius:4px;'>"
        info_html += f"Dataset Loaded Successfully! &nbsp;|&nbsp; Total Records: {dataset.shape[0]} &nbsp;|&nbsp; Total Attributes: {dataset.shape[1]}</div>"
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
