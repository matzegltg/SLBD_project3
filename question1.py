import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.mixture import GaussianMixture
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

# load cancerdata
# 2999 dimensions
# 82 observations

data = pd.read_csv('Cancerdata.txt', sep="\t")

# column 0 contains classes
data.columns = [str(i) for i in range(3000)]

# select all data except of cancer class labels
data_m = data.loc[:, data.columns != '0'].to_numpy()

# TODO: Understanding Hierarchical clustering
# TODO: 4th cluster method
# 1: Kmeans, 2: GMM, 3: Hierarchical clustering (bottom up, agglomerative)
cluster_method = 3
n_clusters = [2,3,4,5]
princip_components = 10
silhouette = True
mean_silhouette_scores = []

for n_cluster in n_clusters:
    reduced_data = PCA(n_components=princip_components).fit_transform(data_m)
    if cluster_method == 1:
        clusterer = KMeans(n_clusters=n_cluster)
        labels = clusterer.fit(reduced_data)
        cluster_labels = labels.labels_
        
    elif cluster_method == 2:
        clusterer = GaussianMixture(n_components=n_cluster)
        clusterer.fit(reduced_data)
        cluster_labels = clusterer.predict(reduced_data)

    elif cluster_method == 3:
        Z = linkage(reduced_data, "average")
        cluster_labels = fcluster(Z, t=n_cluster, criterion="maxclust")
        
    elif cluster_method == 4:
        pass
          
    fig, axes = plt.subplots(nrows=1, ncols=2)
    
    axes[0].scatter(reduced_data[:,0], reduced_data[:,1], c=data['0'])
    axes[0].set_title("Original data")
    axes[1].set_title(f"Cluster method {cluster_method}, with {princip_components} pcs and {n_cluster} clusters")
    axes[1].scatter(reduced_data[:,0], reduced_data[:,1], c=cluster_labels)
    plt.show()
    
    if silhouette:
        fig, axes = plt.subplots(nrows=1, ncols=1)
        # visualize silhouette score
        # The silhouette coefficient can range from -1, 1
        axes.set_xlim([-0.5, 1])
        silhouette_avg = silhouette_score(reduced_data, cluster_labels)
        
        mean_silhouette_scores.append(silhouette_avg)
        print("For n_clusters =", n_cluster, "The average silhouette_score is :", silhouette_avg,)
        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(reduced_data, cluster_labels)
        
        y_lower = 5
        for i in range(n_cluster):
            # Aggregate silhouette scores for samples belonging to cluster i and sort them
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
            ith_cluster_silhouette_values.sort()
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i
            axes.fill_betweenx(np.arange(y_lower, y_upper),0,ith_cluster_silhouette_values,alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            axes.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 5 # 5 for the 0 samples
        axes.set_title(f"Silhouette plot for k = {n_cluster}")
        axes.set_xlabel("Silhouette coefficient")
        axes.set_ylabel("Observations with corresponding cluster label")

        # The vertical line for average silhouette score of all the values
        axes.axvline(x=silhouette_avg, color="red", linestyle="--")
        axes.set_yticks([])  # Clear the yaxis labels / ticks
        plt.show()
    
    # decision boundary visualizable
    if princip_components == 2:
        fig, axes = plt.subplots(nrows=1, ncols=1)

        axes.scatter(reduced_data[:,0], reduced_data[:,1], c=data['0'],s=20, edgecolors='black')
        # Step size of the mesh. Decrease to increase the quality of the VQ.
        h = 0.02  # point in the mesh [x_min, x_max]x[y_min, y_max].
        # Plot the decision boundary. For that, we will assign a color to each
        x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
        y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        
        # Obtain labels for each point in mesh. Use last trained model.
        Z = clusterer.predict(np.c_[xx.ravel(), yy.ravel()])
        
        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        
        # plot decision boundaries
        axes.imshow(Z, interpolation="nearest", extent=(xx.min(), xx.max(), yy.min(), yy.max()), aspect="auto", origin="lower", cmap="viridis", alpha=0.2)
        
        # mark centroids of kmeans
        if cluster_method == 1:
            # Plot the centroids as a white X
            centroids = clusterer.cluster_centers_
            axes.scatter(centroids[:, 0], centroids[:, 1], marker="x", s=169, color="b", zorder=10, alpha=0.5)
        
        axes.set_title("Decision boundary after clustering with original data")
        plt.show()
    if cluster_method == 3:
        plt.figure()
        dn = dendrogram(Z, labels=data.loc[:, data.columns == '0'].to_numpy())
        plt.show()


fig, ax = plt.subplots(1,1)
ax.scatter(n_clusters, mean_silhouette_scores)
ax.plot(n_clusters, mean_silhouette_scores)
ax.set_title("Average silhouette scores over different number of clusters")
ax.set_xlabel("Number of clusters")
ax.set_ylabel("Average silhouette score")

plt.show()
