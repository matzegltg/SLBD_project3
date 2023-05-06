import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.mixture import GaussianMixture


# load the data

X = np.load("X.npy")
# save np.load
np_load_old = np.load
# modify the default parameters of np.load
np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)
y = np.load("y.npy")
# restore np.load for future normal usage
np.load = np_load_old


# highlight the imbalance in classes
print("nb of samples belonging to class BC", np.sum(y == 0))
print("nb of samples belonging to class GBM", np.sum(y == 1))
print("nb of samples belonging to class KI", np.sum(y == 2))
print("nb of samples belonging to class LU", np.sum(y == 3))
print("nb of samples belonging to class OV", np.sum(y == 4))
print("nb of samples belonging to class U", np.sum(y == 5))

# add pseudo observations


def add_pseudo_observations(X, y, num_pseudo, weight1=0.85, weight2=0.15, class_to_augment=None):
  classes = np.unique(y)

  X_pseudo = X.copy()
  y_pseudo = y.copy()

  for c in classes:
    if class_to_augment is not None and c != class_to_augment:
        continue

    X_c = X[y == c]
    n_c = X_c.shape[0]
    if n_c < 2:
        continue

    for i in range(num_pseudo):
      # Choose two random observations from the same class
      idx1, idx2 = np.random.choice(n_c, size=2, replace=False)
      obs1, obs2 = X_c[idx1], X_c[idx2]
      pseudo_obs = weight1 * obs1 + weight2 * obs2

      # Append the new observation and corresponding label
      X_pseudo = np.vstack((X_pseudo, pseudo_obs))
      y_pseudo = np.append(y_pseudo, c)

  return X_pseudo, y_pseudo


# removing observations
def remove_observations(X, y, num_rm, class_to_delete_from):
  indices_c = np.where(y == class_to_delete_from)[0]
  for i in range(num_rm):
    # Choose random indexes from indices_c
    id = np.random.choice(indices_c)
  # Remove the observation and corresponding label at index idx
    X_rm = np.delete(X, id, axis=0)
    y_rm = np.delete(y, id, axis=0)
    indices_c = np.where(y == class_to_delete_from)[0]
  return X_rm, y_rm


sil_sc = {'5': 0, '6': 0, '7': 0}
n_simulations = 3
for i in range(n_simulations):
  # Add pseudo-observations to class i
  X_pseudo, y_pseudo = add_pseudo_observations(
      X, y, num_pseudo=50, class_to_augment=5)

  #print(X.shape, y.shape)
  #print(X_pseudo.shape, y_pseudo.shape)

  # Check the number of samples in each class before and after augmentation
  print("\n------------after adding pseudo-observations------------")
  print("nb of samples belonging to class BC", np.sum(y_pseudo == 0))
  print("nb of samples belonging to class GBM", np.sum(y_pseudo == 1))
  print("nb of samples belonging to class KI", np.sum(y_pseudo == 2))
  print("nb of samples belonging to class LU", np.sum(y_pseudo == 3))
  print("nb of samples belonging to class OV", np.sum(y_pseudo == 4))
  print("nb of samples belonging to class U", np.sum(y_pseudo == 5))

  X_rm, y_rm = remove_observations(X_pseudo, y_pseudo, 20, 0)
  X_rm, y_rm = remove_observations(X_rm, y_rm, 20, 1)
  X_rm, y_rm = remove_observations(X_rm, y_rm, 20, 2)
  X_rm, y_rm = remove_observations(X_rm, y_rm, 20, 3)

  print("\n------------after  removing-observations------------")
  print("nb of samples belonging to class BC", np.sum(y_rm == 0))
  print("nb of samples belonging to class GBM", np.sum(y_rm == 1))
  print("nb of samples belonging to class KI", np.sum(y_rm == 2))
  print("nb of samples belonging to class LU", np.sum(y_rm == 3))
  print("nb of samples belonging to class OV", np.sum(y_rm == 4))
  print("nb of samples belonging to class U", np.sum(y_rm == 5))

  cluster_method = 2

  n_clusters = [5, 6, 7]

  princip_components = 10
  silhouette = True
  mean_silhouette_scores = []
  for n_cluster in n_clusters:
      reduced_data = PCA(n_components=princip_components).fit_transform(X_rm)
      if cluster_method == 2:
          clusterer = GaussianMixture(n_components=n_cluster)
          clusterer.fit(reduced_data)
          cluster_labels = clusterer.predict(reduced_data)
      elif cluster_method == 1:
          clusterer = KMeans(n_clusters=n_cluster)
          labels = clusterer.fit(reduced_data)
          cluster_labels = labels.labels_

      fig, axes = plt.subplots(nrows=1, ncols=2)

      axes[0].scatter(reduced_data[:, 0], reduced_data[:, 1], c=y_rm)
      axes[0].set_title("Original data")
      axes[1].scatter(reduced_data[:, 0], reduced_data[:, 1], c=cluster_labels)
      axes[1].set_title(
          f"Cluster method {cluster_method}, with {princip_components} pcs and {n_cluster} clusters")

      plt.show()
      if silhouette:
          fig, axes = plt.subplots(nrows=1, ncols=1)
          # visualize silhouette score
          # The silhouette coefficient can range from -1, 1
          axes.set_xlim([-0.5, 1])
          silhouette_avg = silhouette_score(reduced_data, cluster_labels)

          mean_silhouette_scores.append(silhouette_avg)
          print("For n_clusters =", n_cluster,
                "The average silhouette_score is :", silhouette_avg,)
          # Compute the silhouette scores for each sample
          sample_silhouette_values = silhouette_samples(
              reduced_data, cluster_labels)

          y_lower = 5
          for i in range(n_cluster):
              # Aggregate silhouette scores for samples belonging to cluster i and sort them
              ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
              ith_cluster_silhouette_values.sort()
              size_cluster_i = ith_cluster_silhouette_values.shape[0]
              y_upper = y_lower + size_cluster_i
              axes.fill_betweenx(np.arange(y_lower, y_upper),
                                 0, ith_cluster_silhouette_values, alpha=0.7)

              # Label the silhouette plots with their cluster numbers at the middle
              axes.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

              # Compute the new y_lower for next plot
              y_lower = y_upper + 5  # 5 for the 0 samples
          axes.set_title(f"Silhouette plot for k = {n_cluster}")
          axes.set_xlabel("Silhouette coefficient")
          axes.set_ylabel("Observations with corresponding cluster label")

          # The vertical line for average silhouette score of all the values
          axes.axvline(x=silhouette_avg, color="red", linestyle="--")
          axes.set_yticks([])  # Clear the yaxis labels / ticks
      # decision boundary visualizable
      if princip_components == 2:
          fig, axes = plt.subplots(nrows=1, ncols=1)

          axes.scatter(reduced_data[:, 0], reduced_data[:, 1],
                       c=X_rm['0'], s=20, edgecolors='black')
          # Step size of the mesh. Decrease to increase the quality of the VQ.
          h = 0.02  # point in the mesh [x_min, x_max]x[y_min, y_max].
          # Plot the decision boundary. For that, we will assign a color to each
          x_min, x_max = reduced_data[:, 0].min(
          ) - 1, reduced_data[:, 0].max() + 1
          y_min, y_max = reduced_data[:, 1].min(
          ) - 1, reduced_data[:, 1].max() + 1
          xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                               np.arange(y_min, y_max, h))

          # Obtain labels for each point in mesh. Use last trained model.
          Z = clusterer.predict(np.c_[xx.ravel(), yy.ravel()])

          # Put the result into a color plot
          Z = Z.reshape(xx.shape)

          # plot decision boundaries
          axes.imshow(Z, interpolation="nearest", extent=(xx.min(), xx.max(), yy.min(
          ), yy.max()), aspect="auto", origin="lower", cmap="viridis", alpha=0.2)

          # mark centroids of kmeans
          if cluster_method == 1:
              # Plot the centroids as a white X
              centroids = clusterer.cluster_centers_
              axes.scatter(centroids[:, 0], centroids[:, 1],
                           marker="x", s=169, color="b", zorder=10, alpha=0.5)

          axes.set_title(
              "Decision boundary after clustering with original data")
          plt.show()

  fig, ax = plt.subplots(1, 1)
  for i in range(3):
    key = str(n_clusters[i])
    print(key)
    val = sil_sc[key]
    val += mean_silhouette_scores[i]
    sil_sc[str(n_clusters[i])] = val
  #ax.scatter(n_clusters, mean_silhouette_scores)
  #ax.plot(n_clusters, mean_silhouette_scores)
  #ax.set_title("Average silhouette scores over different number of clusters")
  #ax.set_xlabel("Number of clusters")
  #ax.set_ylabel("Average silhouette score")

print(sil_sc)


x = sil_sc.keys()
y = []
for elem in sil_sc.values():
    y.append(elem/n_simulations)

plt.scatter(x, y)
plt.plot(x, y)

plt.show()

#{'5': 0.4151265381466433, '6': 0.4143683742401991, '7': 0.3869372227580372}
