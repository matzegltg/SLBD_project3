library(readr)
library(dplyr)
library(tidyr)
library(ggplot2)
library(cluster)
library(factoextra)
library(NbClust)
library(mclust)
library(stats)
library(cluster)
library(pheatmap)
#BiocManager::install("ConsensusClusterPlus")
#browseVignettes("ConsensusClusterPlus")
library(ConsensusClusterPlus)

set.seed(2)

# Load data
# 2999 dimensions
# 82 observations
data <- read.table("Cancerdata.txt", header = TRUE)

# Split data into classes and predictors
data_classes <- data$lab
data_features <- data[, -1]

#Clustering
cluster_method <- 2
n_clusters <- c(2,3,4,5)
princip_components <- 10

# Empty vectors to store the silhouette scores and average silhouette score
sil_scores_kmeans <- c()
sil_scores_kmedoids <- c()
sil_scores_gmm <- c()
sil_scores_wardmethod <- c()
avg_sil_scores <- c()

# compute principal components
pca <- prcomp(data_features, center = TRUE, scale = FALSE)

# extract proportion of variance explained by each principal component
prop_var <- pca$sdev^2/sum(pca$sdev^2)

# plot scree plot
plot(prop_var, type = "b", xlab = "Number of Principal Components", ylab = "Proportion of Variance Explained")

reduced_data <- prcomp(data_features, center = TRUE, scale = FALSE)$x[, 1:princip_components]


for (n_cluster in n_clusters) {
  #K means
  kmeans_clust <- kmeans(reduced_data, n_cluster)
  kmeans_clust_labels <- kmeans_clust$cluster
  sil_kmeans <- silhouette(kmeans_clust_labels, dist(reduced_data))
  sil_scores_kmeans <- c(sil_scores_kmeans, mean(sil_kmeans[, 3]))
  plot(sil_kmeans)
  fviz_silhouette(sil_kmeans, palette = "jco", ggtheme = theme_classic())
  
  #K medoids
  kmedoids_clust <- pam(reduced_data, n_cluster)
  kmedoids_clust_labels <- kmedoids_clust$cluster
  #plot(kmedoids_clust$silinfo$widths[,3],col=kmedoids_clust$silinfo$widths[,1],type="h")
  sil_kmedoids <- silhouette(kmedoids_clust_labels, dist(reduced_data))
  sil_scores_kmedoids <- c(sil_scores_kmedoids, mean(sil_kmedoids[, 3]))
  plot(sil_kmedoids)
  fviz_silhouette(sil_kmedoids, palette = "jco", ggtheme = theme_classic())
  
  #GMM
  gmm_clust <- Mclust(reduced_data, G = n_cluster)
  gmm_clust_labels <- gmm_clust$classification
  sil_gmm <- silhouette(gmm_clust_labels, dist(reduced_data))
  sil_scores_gmm <- c(sil_scores_gmm, mean(sil_gmm[, 3]))
  plot(sil_gmm)
  fviz_silhouette(sil_gmm, palette = "jco", ggtheme = theme_classic())
  
  #Ward method hierarchical clustering
  wardmethod_clust<-agnes(reduced_data,  metric = "euclidean",
                    stand = FALSE, method = "ward", keep.data = FALSE)
  pltree(wardmethod_clust,main="Ward method", cex=0.83,xlab="")
  wardmethod_clust_labels<-cutree(wardmethod_clust,n_cluster)
  sil_wardmethod <- silhouette(wardmethod_clust_labels, dist(reduced_data))
  sil_scores_wardmethod <- c(sil_scores_wardmethod, mean(sil_wardmethod[, 3]))
  plot(sil_wardmethod)
  fviz_silhouette(sil_wardmethod, palette = "jco", ggtheme = theme_classic())
  
  # Scatter plot of first 2 principal components with original labels
  if (n_cluster == 3) {
    data_classes_plot <- data_classes + 1
    plot(reduced_data[, 1], reduced_data[, 2], col = data_classes_plot, pch = 19,
         main = paste("Original labels"), xlab = "PC1", ylab = "PC2")
    legend("topright", legend = unique(kmeans_clust_labels), col = unique(kmeans_clust_labels), pch = 19)
  }
  
  # Scatter plot of first 2 principal components with cluster labels
  plot(reduced_data[, 1], reduced_data[, 2], col = kmeans_clust_labels, pch = 19,
       main = paste("K-means with", n_cluster, "clusters"), xlab = "PC1", ylab = "PC2")
  legend("topright", legend = unique(kmeans_clust_labels), col = unique(kmeans_clust_labels), pch = 19)
  
  plot(reduced_data[, 1], reduced_data[, 2], col = kmedoids_clust_labels, pch = 19,
       main = paste("K-medoids with", n_cluster, "clusters"), xlab = "PC1", ylab = "PC2")
  legend("topright", legend = unique(kmedoids_clust_labels), col = unique(kmedoids_clust_labels), pch = 19)
  
  plot(reduced_data[, 1], reduced_data[, 2], col = gmm_clust_labels, pch = 19,
       main = paste("GMM with", n_cluster, "clusters"), xlab = "PC1", ylab = "PC2")
  legend("topright", legend = unique(gmm_clust_labels), col = unique(gmm_clust_labels), pch = 19)
  
  plot(reduced_data[, 1], reduced_data[, 2], col = wardmethod_clust_labels, pch = 19,
       main = paste("Ward method hierarchical clustering with", n_cluster, "clusters"), xlab = "PC1", ylab = "PC2")
  legend("topright", legend = unique(wardmethod_clust_labels), col = unique(wardmethod_clust_labels), pch = 19)
  
  #Compute adjusted rand score between prediction with k=3 and original labels
  if (n_cluster==3) {
    ari <- adjustedRandIndex(kmeans_clust_labels, data_classes)
    print(paste("Adjusted Rand Index for K-means: ", ari))
    ari <- adjustedRandIndex(kmedoids_clust_labels, data_classes)
    print(paste("Adjusted Rand Index for K-medoids: ", ari))
    ari <- adjustedRandIndex(gmm_clust_labels, data_classes)
    print(paste("Adjusted Rand Index for GMM: ", ari))
    ari <- adjustedRandIndex(wardmethod_clust_labels, data_classes)
    print(paste("Adjusted Rand Index for Ward Method: ", ari))
  }
}


#Heatmaps
pheatmap(reduced_data, clustering_method = "complete", cluster_cols = TRUE, cluster_rows = TRUE)
pheatmap(reduced_data, clustering_method = "average", cluster_cols = TRUE, cluster_rows = TRUE)
pheatmap(reduced_data, clustering_method = "ward", cluster_cols = TRUE, cluster_rows = TRUE)


# Create a new plot with the average silhouette score for each model
x <- c(2, 3, 4, 5)
plot(x, sil_scores_kmeans, type = "l", col = "red", xlab = "number of clusters", ylab = "avg silhouette score", ylim = c(0,0.25))
lines(x, sil_scores_kmedoids, type = "l", col = "blue")
lines(x, sil_scores_gmm, type = "l", col = "green")
lines(x, sil_scores_wardmethod, type = "l", col = "purple")
# Add a legend
legend("bottomright", legend = c("K-means", "K-medoids", "GMM", "Ward Method"), col = c("red", "blue", "green", "purple"), lty = 1, cex = 0.8)



dist_matrix <- dist(reduced_data, method = "euclidean")
cc_km <- ConsensusClusterPlus(t(reduced_data),maxK=5,reps=100,pItem=0.8,pFeature=1,
                         clusterAlg="km")

cc_pam <- ConsensusClusterPlus(dist_matrix,maxK=5,reps=100,pItem=0.8,pFeature=1,
                           clusterAlg="pam")

cc_hc <- ConsensusClusterPlus(dist_matrix,maxK=5,reps=100,pItem=0.8,pFeature=1,
                           clusterAlg="hc")


    