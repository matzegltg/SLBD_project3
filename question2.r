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
library(irlba)


load("TCGAData.RData")
ls() #returns a list of all the objects you just loaded (and anything else in your environment)

X = TCGA
y = TCGAclassstr

add_pseudo_obs <- function(X, y, num_add, class_to_add_to) {
  indices_c <- which(y == class_to_add_to)
  X_c <- X[indices_c,]
  n_c <- length(indices_c)
  new_X <- matrix(NA, nrow=num_add, ncol=ncol(X))
  new_y <- rep(class_to_add_to,num_add)
  for (i in 1:num_add) {
    obs_idx <- sample(1:n_c, size=2, replace=TRUE)
    new_X[i,] <- 0.85*X_c[obs_idx[1],] + 0.15*X_c[obs_idx[2],]
  }
  # Add the new observations to the original data
  X_new <- rbind(X, new_X)
  y_new <- c(y, new_y)
  return(list(X_pseudo = X_new, y_pseudo = y_new))
}


remove_observations <- function(X, y, num_rm, class_to_remove_from) {
  indices_c <- which(y == class_to_remove_from)
  n_c <- length(indices_c)
  idx_rm <- sample(indices_c, size=num_rm, replace=FALSE)
  X_rm <- X[-idx_rm,]
  y_rm <- y[-idx_rm]
  return(list(X_rm = X_rm, y_rm = y_rm))
}


#Clustering
cluster_method <- 2
n_clusters <- c(5,6,7)
princip_components <- 10

# Empty vectors to store the silhouette scores and average silhouette score
sil_scores_kmeans <- c(0,0,0)
sil_scores_kmedoids <- c(0,0,0)
sil_scores_gmm <- c(0,0,0)
sil_scores_wardmethod <- c(0,0,0)

pca_result <- prcomp_irlba(X, n = princip_components)
reduced_data <- pca_result$x[,1:princip_components]


# compute principal components
#pca <- prcomp(data_features, center = TRUE, scale = FALSE)

# extract proportion of variance explained by each principal component
prop_var <- pca_result$sdev^2/sum(pca_result$sdev^2)

# plot scree plot
plot(prop_var, type = "b", xlab = "Number of Principal Components", ylab = "Proportion of Variance Explained")

#reduced_data <- pca$x[, 1:princip_components]

nb_tests = 1

for (count in 1:nb_tests) {
  print(count)
  sil_scores_kmeans_ <- c()
  sil_scores_kmedoids_ <- c()
  sil_scores_gmm_ <- c()
  sil_scores_wardmethod_ <- c()
  modified_data <- add_pseudo_obs(reduced_data, y, 28, "GBM")
  modified_data <- add_pseudo_obs(modified_data$X_pseudo, modified_data$y_pseudo, 143, "U")
  modified_data_rm <- remove_observations(modified_data$X_pseudo, modified_data$y_pseudo, 1015,"BC" ) 
  modified_data_rm <- remove_observations(modified_data_rm$X_rm, modified_data_rm$y_rm, 406, "KI")
  modified_data_rm <- remove_observations(modified_data_rm$X_rm, modified_data_rm$y_rm, 66, "OV")
  modified_data_rm <- remove_observations(modified_data_rm$X_rm, modified_data_rm$y_rm, 371, "LU")
  data_classes <- modified_data_rm$y_rm
  data_features <- modified_data_rm$X_rm
  
  
  for (n_cluster in n_clusters) {
    
    #K means
    kmeans_clust <- kmeans(reduced_data, n_cluster)
    kmeans_clust_labels <- kmeans_clust$cluster
    sil_kmeans <- silhouette(kmeans_clust_labels, dist(reduced_data))
    sil_scores_kmeans_ <- c(sil_scores_kmeans_, mean(sil_kmeans[, 3]))
    fviz_silhouette(sil_kmeans, palette = "jco", ggtheme = theme_classic())
    #plot(sil_kmeans)
  
    #K medoids
    kmedoids_clust <- pam(reduced_data, n_cluster)
    kmedoids_clust_labels <- kmedoids_clust$cluster
    sil_kmedoids <- silhouette(kmedoids_clust_labels, dist(reduced_data))
    sil_scores_kmedoids_ <- c(sil_scores_kmedoids_, mean(sil_kmedoids[, 3]))
    fviz_silhouette(sil_kmedoids, palette = "jco", ggtheme = theme_classic())
    #plot(sil_kmedoids)
  
    #GMM
    gmm_clust <- Mclust(reduced_data, G = n_cluster)
    gmm_clust_labels <- gmm_clust$classification
    sil_gmm <- silhouette(gmm_clust_labels, dist(reduced_data))
    sil_scores_gmm_ <- c(sil_scores_gmm_, mean(sil_gmm[, 3]))
    fviz_silhouette(sil_gmm, palette = "jco", ggtheme = theme_classic())
    #plot(sil_gmm)
  
    # #Ward method hierarchical clustering
    wardmethod_clust<-agnes(reduced_data,  metric = "euclidean",
                            stand = FALSE, method = "ward", keep.data = FALSE)
    pltree(wardmethod_clust,main="Ward method", cex=0.83,xlab="")
    wardmethod_clust_labels<-cutree(wardmethod_clust,n_cluster)
    sil_wardmethod <- silhouette(wardmethod_clust_labels, dist(reduced_data))
    sil_scores_wardmethod_ <- c(sil_scores_wardmethod_, mean(sil_wardmethod[, 3]))
    fviz_silhouette(sil_wardmethod, palette = "jco", ggtheme = theme_classic())
    #plot(sil_wardmethod)
  
    # # Scatter plot of first 2 principal components with original labels
    if (n_cluster == 6) {
      data_classes_plot <- as.numeric(factor(data_classes, levels = c("BC", "GBM", "KI", "OV", "U", "LU")), labels = c(0, 1, 2, 3, 4, 5))
      plot(reduced_data[, 1], reduced_data[, 2], col = data_classes_plot, pch = 19,
           main = paste("Original labels"), xlab = "PC1", ylab = "PC2")
      legend("bottomright", legend = unique(kmeans_clust_labels), col = unique(kmeans_clust_labels), pch = 19)
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
    # if (n_cluster==6) {
    #   ari <- adjustedRandIndex(kmeans_clust_labels, data_classes)
    #   print(paste("Adjusted Rand Index for K-means: ", ari))
    #   ari <- adjustedRandIndex(kmedoids_clust_labels, data_classes)
    #   print(paste("Adjusted Rand Index for K-medoids: ", ari))
    #   ari <- adjustedRandIndex(gmm_clust_labels, data_classes)
    #   print(paste("Adjusted Rand Index for GMM: ", ari))
    #   ari <- adjustedRandIndex(wardmethod_clust_labels, data_classes)
    #   print(paste("Adjusted Rand Index for Ward Method: ", ari))
    # }
  }
  sil_scores_kmeans <- sil_scores_kmeans + sil_scores_kmeans_
  sil_scores_kmedoids <- sil_scores_kmedoids + sil_scores_kmedoids_
  sil_scores_gmm <- sil_scores_gmm + sil_scores_gmm_
  sil_scores_wardmethod <- sil_scores_wardmethod + sil_scores_wardmethod_
  
}


sil_scores_kmeans <- sil_scores_kmeans/nb_tests
sil_scores_kmedoids <- sil_scores_kmedoids/nb_tests
sil_scores_gmm <- sil_scores_gmm/nb_tests
sil_scores_wardmethod <- sil_scores_wardmethod/nb_tests

# Create a new plot with the average silhouette score for each model
x <- c(5, 6, 7)
plot(x, sil_scores_kmeans, type = "l", col = "red", xlab = "number of clusters", ylab = "avg silhouette score", ylim = c(0.3,0.45))
lines(x, sil_scores_kmedoids, type = "l", col = "blue")
lines(x, sil_scores_gmm, type = "l", col = "green")
lines(x, sil_scores_wardmethod, type = "l", col = "purple")
# Add a legend
legend("bottomright", legend = c("K-means", "K-medoids", "GMM", "Ward Method"), col = c("red", "blue", "green", "purple"), lty = 1, cex = 0.8)


#Heatmaps
pheatmap(reduced_data, clustering_method = "complete", cluster_cols = TRUE, cluster_rows = TRUE)
pheatmap(reduced_data, clustering_method = "average", cluster_cols = TRUE, cluster_rows = TRUE)
pheatmap(reduced_data, clustering_method = "ward", cluster_cols = TRUE, cluster_rows = TRUE)


dist_matrix <- dist(reduced_data, method = "euclidean")
cc_km <- ConsensusClusterPlus(t(reduced_data),maxK=5,reps=100,pItem=0.8,pFeature=1,
                              clusterAlg="km")

cc_pam <- ConsensusClusterPlus(dist_matrix,maxK=5,reps=100,pItem=0.8,pFeature=1,
                               clusterAlg="pam")

cc_hc <- ConsensusClusterPlus(dist_matrix,maxK=5,reps=100,pItem=0.8,pFeature=1,
                              clusterAlg="hc")



