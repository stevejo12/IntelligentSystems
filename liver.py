import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler as Scaler
from sklearn.naive_bayes import GaussianNB

dataset = pd.read_csv('ILPD.csv', header=None)

train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)

train_data.columns = [
    "Age", "Gender", "TB",
    "DB", "Alkphos", "SGPT",
    "SGOT", "TP", "ALB",
    "AG", "Selector"]

test_data.columns = [
    "Age", "Gender", "TB",
    "DB", "Alkphos", "SGPT",
    "SGOT", "TP", "ALB",
    "AG", "Selector"]

median = train_data['AG'].median()
print(median)
train_data['AG'].fillna(median, inplace=True)
test_data['AG'].fillna(median, inplace=True)

maps = {'Female': 0.0, 'Male': 1.0}
train_data['Gender'] = train_data['Gender'].map(maps)
test_data['Gender'] = test_data['Gender'].map(maps)

train_labels = train_data["Selector"].copy()
train_data = train_data.drop("Selector", axis=1)

test_labels = test_data["Selector"].copy()
test_data = test_data.drop("Selector", axis=1)

scaler = Scaler()
scaler.fit(train_data)
train_scaled = scaler.transform(train_data)
test_scaled = scaler.transform(test_data)

df = pd.DataFrame(data=train_scaled)
print(df.head())

clf = GaussianNB()
clf.fit(train_scaled, train_labels)
print("Prediction accuracy :",clf.score(test_scaled,test_labels)*100,"%")

