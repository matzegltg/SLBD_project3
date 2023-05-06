import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
data = pd.read_csv('Cancerdata.txt', sep="\t")

# column 0 contains classes
data.columns = [str(i) for i in range(3000)]

# select all data except of cancer class labels
data_m = data.loc[:, data.columns != '0'].to_numpy()

# check whether dataset is normalized per feature
#mean, std = np.mean(data.loc[:, data.rows == '1'].to_numpy()), np.std(data.loc[:, data.columns == '1'].to_numpy())
#print(mean, std)

pca = PCA()
pca.fit(data_m)

y = pca.singular_values_

# Control sum of eigenvalues
print(f"Sum of eigenvalues: {sum(y)}")
fig, ax = plt.subplots()

ax.scatter(np.array(range(1,83)),y)
ax.set_title("Scree plot")
ax.set_xlabel("Principal Component")
ax.set_ylabel("Eigenvalue")
plt.show()

# get eigenvectors
comp = pca.components_
print(comp)