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
sil_scores_completelink <- c()
avg_sil_scores <- c()

for (n_cluster in n_clusters) {
  reduced_data <- prcomp(data_features, center = TRUE, scale = FALSE)$x[, 1:princip_components]
  
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
  completelink_clust<-agnes(reduced_data,  metric = "euclidean",
                    stand = FALSE, method = "ward", keep.data = FALSE)
  pltree(completelink_clust,main="Ward method", cex=0.83,xlab="")
  completelink_clust_labels<-cutree(completelink_clust,n_cluster)
  sil_completelink <- silhouette(completelink_clust_labels, dist(reduced_data))
  sil_scores_completelink <- c(sil_scores_completelink, mean(sil_completelink[, 3]))
  plot(sil_completelink)
  fviz_silhouette(sil_completelink, palette = "jco", ggtheme = theme_classic())
  
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
  
  plot(reduced_data[, 1], reduced_data[, 2], col = completelink_clust_labels, pch = 19,
       main = paste("Ward method hierarchical clustering with", n_cluster, "clusters"), xlab = "PC1", ylab = "PC2")
  legend("topright", legend = unique(completelink_clust_labels), col = unique(completelink_clust_labels), pch = 19)
  
  #Compute adjusted rand score between prediction with k=3 and original labels
  if (n_cluster==3) {
    ari <- adjustedRandIndex(kmeans_clust_labels, data_classes)
    print(paste("Adjusted Rand Index for K-means: ", ari))
    ari <- adjustedRandIndex(kmedoids_clust_labels, data_classes)
    print(paste("Adjusted Rand Index for K-medoids: ", ari))
    ari <- adjustedRandIndex(gmm_clust_labels, data_classes)
    print(paste("Adjusted Rand Index for GMM: ", ari))
    ari <- adjustedRandIndex(completelink_clust_labels, data_classes)
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
lines(x, sil_scores_completelink, type = "l", col = "purple")
# Add a legend
legend("bottomright", legend = c("K-means", "K-medoids", "GMM", "Ward Method"), col = c("red", "blue", "green", "purple"), lty = 1, cex = 0.8)


BiocManager::install("ConsensusClusterPlus")
browseVignettes("ConsensusClusterPlus")
library(ConsensusClusterPlus)
# this can be a bit slot depending on the number of features and observations!
ii<-sample(seq(1,2887),1000)
guse<-vv[1:2000]
options(warn=-1)
cc<-ConsensusClusterPlus(as.matrix(reduced_data),maxK=5,reps=100,pItem=.6,pFeature=.6,
                         clusterAlg="km")
options(warn=0)




    