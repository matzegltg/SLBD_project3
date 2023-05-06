import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

data = pd.read_csv('samples.csv', header = None)
data = data.drop(0, axis=1)
data = data.drop(0, axis=0)

X = np.array(data).astype(float)
print(X.shape)

y_df = pd.read_csv('labels.csv', header = None)
y_df = y_df.drop(0, axis=1)
y_df = y_df.drop(0, axis=0)
y_str = np.array(y_df).reshape(-1)
print(y_str.shape)

y = []
for label in y_str:
    if label == "BC":
        y.append(0)
    if label == "GBM":
        y.append(1)
    if label == "KI":
        y.append(2)
    if label == "LU":
        y.append(3)
    if label == "OV":
        y.append(4)
    if label == "U":
        y.append(5)
y = np.array(y)

np.save("X", X)
np.save("y_str", y_str)
np.save("y", y)

