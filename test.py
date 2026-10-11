import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

def run_test():
    dataset = pd.read_csv("Dataset/Fraud_transaction.csv")
    print(dataset)
    print(np.unique(dataset['isFraud'], return_counts=True))

    for col in dataset.columns:
        if dataset[col].dtype == 'object' or str(dataset[col].dtype).startswith('str'):
            dataset[col] = dataset[col].fillna('0')
        else:
            dataset[col] = dataset[col].fillna(0)
    Y = dataset['isFraud'].values.ravel()
    dataset.drop(['isFraud'], axis=1, inplace=True)

    label_encoder = []
    columns = dataset.columns
    types = dataset.dtypes.values
    for i in range(len(types)):
        name = types[i]
        if name == 'object' or str(name).startswith('str'):
            le = LabelEncoder()
            dataset[columns[i]] = pd.Series(le.fit_transform(dataset[columns[i]].astype(str)))
            label_encoder.append([columns[i], le])

    for col in dataset.columns:
        if dataset[col].dtype == 'object' or str(dataset[col].dtype).startswith('str'):
            dataset[col] = dataset[col].fillna('0')
        else:
            dataset[col] = dataset[col].fillna(0)
    X = dataset.values

    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]

    sc = StandardScaler()
    X = sc.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

    svm_cls = svm.SVC(kernel='poly', C=3.0, gamma=0.25, tol=0.1, degree=3)
    svm_cls.fit(X_train, y_train)
    predict = svm_cls.predict(X_test)
    acc = accuracy_score(y_test, predict)
    print("SVM Accuracy:", acc)

    rf_cls = RandomForestClassifier()
    rf_cls.fit(X_train, y_train)
    predict = rf_cls.predict(X_test)
    acc = accuracy_score(y_test, predict)
    print("Random Forest Accuracy:", acc)

    unique, count = np.unique(Y, return_counts=True)

    pca = PCA(2) 
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(7, 7))
    for cls in unique:
        plt.scatter(X_pca[Y == cls, 0], X_pca[Y == cls, 1], label=cls) 
    plt.legend()
    plt.title("Process Mining User Behaviour Graph")
    plt.savefig("process_mining_graph.png")
    print("Graph saved to process_mining_graph.png")

if __name__ == '__main__':
    run_test()
