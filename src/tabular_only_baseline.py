"""Tabular-only control: how well do the synthetic vitals predict the label with NO image?"""
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

train = pd.read_csv("data/processed/train.csv")
test = pd.read_csv("data/processed/test.csv")
for df in (train, test):
    df["male"] = (df["Patient Gender"] == "Male").astype(int)

subsets = {
    "temp + SpO2": ["Temperature", "SpO2"],
    "all four features": ["Patient Age", "male", "Temperature", "SpO2"],
    "age + gender only (control)": ["Patient Age", "male"],
}
for name, cols in subsets.items():
    scaler = StandardScaler().fit(train[cols])
    clf = LogisticRegression(max_iter=1000).fit(scaler.transform(train[cols]), train["Pneumonia"])
    prob = clf.predict_proba(scaler.transform(test[cols]))[:, 1]
    print(f"{name:30s} test AUROC = {roc_auc_score(test['Pneumonia'], prob):.4f}")