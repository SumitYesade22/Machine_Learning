# DBSCAN (Density-Based Clustering)

## 1. Definition

DBSCAN stands for **Density-Based Spatial Clustering of Applications with Noise**.

DBSCAN is an unsupervised, density-based clustering algorithm whose main objective is to identify dense regions in a dataset as clusters and separate them from less-dense regions, where points may be classified as noise.

It groups data points based on their density and proximity rather than relying on predefined cluster centers.

## 2. Why DBSCAN When K-Means Is Available?

K-Means is a widely used clustering algorithm, but DBSCAN provides an alternative approach to clustering.

### K-Means

- Requires the number of clusters (`K`) to be specified beforehand.
- Groups data points based on their distance from cluster centroids.
- Works well when clusters are relatively compact and centroid-based grouping is appropriate.
- Assigns every data point to a cluster.

### DBSCAN

- Does not require the number of clusters to be specified beforehand.
- Identifies clusters based on the density of nearby data points.
- Can identify clusters with arbitrary shapes.
- Can identify noise points that do not belong to any cluster.

## 3. When Do We Use DBSCAN?

DBSCAN can be useful when:

- The number of clusters is unknown.
- Clusters have irregular or arbitrary shapes.
- The dataset contains noise or outliers.
- The density and proximity of data points represent meaningful patterns.

## 4. Main Idea

The fundamental idea behind DBSCAN is:

**Dense regions of data points form clusters, while points in less-dense regions may be classified as noise.**

The algorithm identifies connected dense regions and groups their data points into clusters.
