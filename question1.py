import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.mixture import GaussianMixture

# load cancerdata
# 2999 dimensions
# 82 observations

data = pd.read_csv('Cancerdata.txt', sep="\t")

# column 0 contains classes
data.columns = [str(i) for i in range(3000)]

# select all data except of cancer class labels
data_m = data.loc[:, data.columns != '0'].to_numpy()


# 1: Kmeans, 2: GMM
cluster_method = 3

if cluster_method == 1 or cluster_method == 2 or cluster_method == 3:
    n_clusters = [2,3,4,5,6]
    
    mean_silhouette_scores = []
    
    for n_cluster in n_clusters:
        reduced_data = PCA(n_components=2).fit_transform(data_m)
        if cluster_method == 2:
            clusterer = GaussianMixture(n_components=n_cluster)
            clusterer.fit(reduced_data)
            cluster_labels = clusterer.predict(reduced_data)
        elif cluster_method == 1:
            clusterer = KMeans(n_clusters=n_cluster)
            labels = clusterer.fit(reduced_data)
            cluster_labels = labels.labels_
            
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
        
        fig, axes = plt.subplots(nrows=1, ncols=2)

        
        # visualize original labels maintain same label colour
        rows_0 = data.index[data['0'] == 0].tolist()
        rows_1 = data.index[data['0'] == 1].tolist()
        rows_2 = data.index[data['0'] == 2].tolist()

        rows_0 = [elem - 1 for elem in rows_0]
        rows_1 = [elem - 1 for elem in rows_1]
        rows_2 = [elem - 1 for elem in rows_2]
        
        axes[0].scatter(reduced_data[rows_0,0], reduced_data[rows_0,1], c="#107c13",s=20, edgecolors='black')
        axes[0].scatter(reduced_data[rows_1,0], reduced_data[rows_1,1], c="#75107c",s=20, edgecolors='black')
        axes[0].scatter(reduced_data[rows_2,0], reduced_data[rows_2,1], c="#ffde00",s=20, edgecolors='black')
        
        # plot decision boundaries
        axes[0].imshow(
            Z,
            interpolation="nearest",
            extent=(xx.min(), xx.max(), yy.min(), yy.max()),
            aspect="auto",
            origin="lower",
            cmap="viridis",
            alpha=0.2
        )

        if cluster_method == 1:
            # Plot the centroids as a white X
            centroids = clusterer.cluster_centers_
            axes[0].scatter(
                centroids[:, 0],
                centroids[:, 1],
                marker="x",
                s=169,
                linewidths=3,
                color="b",
                zorder=10,
                alpha=0.5
            )

        if cluster_method == 1:
            name = f"K means with k = {n_cluster}"
        elif cluster_method == 2:
            name = f"GMM with n_components = {n_cluster}"
        axes[0].set_title(
            f"{name}, clustering on the cancer dataset (PCA-reduced data)\n"
            "Centroids are marked with blue cross\n"
            "Colored points are the true label"
        )
        axes[0].set_xlim(x_min, x_max)
        axes[0].set_ylim(y_min, y_max)
        axes[0].set_xlabel("PCA 1")
        axes[0].set_ylabel("PCA 2")

        
        # visualize silhouette score
        # The silhouette coefficient can range from -1, 1
        axes[1].set_xlim([-0.5, 1])

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
            axes[1].fill_betweenx(np.arange(y_lower, y_upper),0,ith_cluster_silhouette_values,alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            axes[1].text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 5 # 5 for the 0 samples
        axes[1].set_title(f"Silhouette plot for k = {n_cluster}")
        axes[1].set_xlabel("Silhouette coefficient")
        axes[1].set_ylabel("Observations with corresponding cluster label")

        # The vertical line for average silhouette score of all the values
        axes[1].axvline(x=silhouette_avg, color="red", linestyle="--")

        axes[1].set_yticks([])  # Clear the yaxis labels / ticks
        
        plt.show()
    
    fig, ax = plt.subplots(1,1)
    ax.scatter(n_clusters, mean_silhouette_scores)
    ax.plot(n_clusters, mean_silhouette_scores)
    ax.set_title("Average silhouette scores over different number of clusters")
    ax.set_xlabel("Number of clusters")
    ax.set_ylabel("Average silhouette score")
    plt.show()
else:
    
    # check whether dataset is normalized per feature
    mean, std = np.mean(data.loc[:, data.rows == '1'].to_numpy()), np.std(data.loc[:, data.columns == '1'].to_numpy())
    print(mean, std)

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
