# K-Means Clustering

## 1. K-Means Clustering: Geometric Intuition

K-Means is an unsupervised machine learning algorithm used to group similar data points into clusters.

It works by assigning each data point to the nearest centroid based on a distance metric, typically Euclidean distance.

### How K-Means Works

1. Specify the number of clusters `K`.
2. Initialize `K` centroids.
3. Assign each data point to its nearest centroid.
4. Calculate the mean of the points assigned to each cluster and update the centroid.
5. Repeat the assignment and centroid update steps until the centroids stop changing significantly or a stopping condition is reached.

### Important Terminology

- **Cluster:** A group of data points assigned to the same centroid.
- **Centroid:** The mean position of the data points belonging to a cluster.
- **Cluster assignment:** The cluster label assigned to each data point.
- **Convergence:** The point at which the algorithm stops because further updates are negligible or a stopping condition is met.

K-Means is a distance-based algorithm, so it is primarily designed for numerical features.

---

## 2. Objective Function and Within-Cluster Sum of Squares (Inertia)

The objective of K-Means is to minimize the sum of squared distances between data points and their assigned cluster centroids.

This objective is called **Within-Cluster Sum of Squares (WCSS)** or **Inertia**.

### Objective Function

`Inertia = Sum of squared distances of all points from their assigned centroids`

Mathematically:

`Inertia = Σ ||x_i - μ_c_i||²`

Where:

- `x_i` = ith data point
- `μ_c_i` = centroid of the cluster assigned to point i
- `||x_i - μ_c_i||²` = squared Euclidean distance between the point and its centroid

### Interpretation

Lower inertia means data points are closer to their assigned centroids.

However, inertia generally decreases as K increases because increasing the number of clusters allows points to be assigned to more specific groups.

Therefore, inertia alone cannot determine the optimal number of clusters.

### Important

K-Means minimizes inertia, but it does not guarantee finding the global minimum. It can converge to a local optimum depending on initialization.

---

## 3. How to Find the Number of Clusters (K)?

K-Means requires the number of clusters to be specified beforehand.

Two commonly used methods for choosing K are:

### 3.1 Elbow Method

The Elbow Method calculates inertia for different values of K.

As K increases, inertia decreases. We plot K against inertia and look for an elbow—a point where adding more clusters produces a relatively smaller reduction in inertia.

The elbow provides a reasonable candidate for K.

However, the elbow may not always be clearly visible.

### 3.2 Silhouette Score

The Silhouette Score evaluates how well data points fit into their assigned clusters compared to the nearest other cluster.

It considers both:

- **Cohesion:** How close a point is to other points in its own cluster.
- **Separation:** How far a point is from points in the nearest other cluster.

The score ranges from -1 to +1.

- Near +1: Point is well-matched to its own cluster and separated from other clusters.
- Near 0: Point may be near the boundary between clusters.
- Negative: Point may be closer to another cluster than to its assigned cluster.

### Silhouette Score Calculation

For an individual data point:

`a = Average distance from the point to other points in its own cluster`

`b = Minimum average distance from the point to points in any other cluster`

The nearest other cluster is identified by calculating the average distance to each other cluster and selecting the smallest average.

The individual Silhouette Score is:

`s = (b - a) / max(a, b)`

### Why Does It Consider Both a and b?

Consider two points:

- Point A: `a = 2`, `b = 2.5`
- Point B: `a = 2`, `b = 10`

Both points have the same average distance from their own cluster.

But Point B is much farther from the nearest other cluster, so it is more clearly separated.

This is why Silhouette Score considers both cohesion and separation.

### Overall Silhouette Score

The overall Silhouette Score is the average of the individual Silhouette Scores of all data points.

`Overall Silhouette Score = Sum of individual scores / Number of data points`

It is used to evaluate and compare clustering results, especially across different K values.

### Important Interpretations

- A higher score generally indicates better-separated and more cohesive clusters.
- A lower score indicates weaker separation or overlapping clusters.
- A low score does not automatically mean K-Means has failed.

The data itself may contain overlapping groups or may not have naturally distinct clusters.

Silhouette Score evaluates geometric clustering quality; it does not determine whether the resulting clusters are useful for a business objective.

---

## 4. Random Initialization Trap and K-Means++

### Random Initialization Trap

K-Means initially selects K centroids.

If these centroids are randomly selected, they may be poorly positioned—for example, multiple centroids may be selected within the same natural group.

This can result in poor cluster assignments.

Although K-Means repeatedly updates centroids, it is not guaranteed to recover the globally best clustering.

It may converge to a local optimum, where further iterations do not improve the objective.

### Why Does Initialization Matter?

K-Means uses the current centroid positions to assign data points.

The assignments determine the points used to calculate the next centroids.

Therefore, poor initial centroids can lead to poor assignments and subsequent centroid updates.

The algorithm can converge even when the resulting clustering is not the best possible solution.

### K-Means++

K-Means++ is a smarter centroid initialization technique designed to reduce the risk of poor initialization.

Its procedure:

1. Select the first centroid randomly.
2. Calculate each point's distance to its nearest already-selected centroid.
3. Select the next centroid probabilistically, with probability proportional to the squared nearest-centroid distance.
4. Repeat until K initial centroids are selected.
5. Run the normal K-Means algorithm.

### Important Clarification

K-Means++ does not directly select the farthest point.

It gives points farther from existing centroids a higher probability of being selected.

This increases the likelihood that initial centroids are spread across different regions of the dataset.

K-Means++ improves initialization but does not guarantee globally optimal clustering.

---

## 5. K-Means Clustering Implementation: Conceptual Workflow

The implementation follows these steps:

1. Prepare the numerical feature dataset.
2. Apply scaling if appropriate.
3. Specify K.
4. Initialize centroids.
5. Assign points to nearest centroids.
6. Update centroids using cluster means.
7. Repeat until convergence.
8. Evaluate the clustering using inertia and Silhouette Score.
9. Profile and interpret the resulting clusters.

### Important Model Attributes

- `labels_`: Cluster label assigned to each training data point.
- `cluster_centers_`: Coordinates of the final cluster centroids.
- `inertia_`: Sum of squared distances of points from their assigned centroids.

Cluster labels are identifiers, not meaningful names or rankings.

For example, Cluster 0 is not inherently better or more important than Cluster 1.

---

## 6. Scaling, Initialization, Cluster Shape, and Sensitivity to Outliers

### 6.1 Feature Scaling

K-Means uses distance calculations.

If features have different numerical ranges, features with larger ranges can dominate the distance calculation.

Example:

- Age: 20–60
- Annual Income: 20,000–200,000

Income may dominate the Euclidean distance because its numerical differences are much larger.

Scaling brings features to comparable scales so that distance is not dominated solely by numerical magnitude.

Common scaling methods:

- StandardScaler: Centers features around mean 0 with standard deviation 1.
- MinMaxScaler: Scales features to a specified range, commonly 0–1.

Scaling is generally important when numerical features have substantially different ranges.

However, scaling should depend on the problem and the meaning of feature magnitudes.

### 6.2 Initialization

Different initial centroid positions can produce different clustering results.

K-Means++ improves the initialization by selecting centroids with distance-based probabilities.

It reduces the risk of poor initialization but does not eliminate all sensitivity to initialization.

### 6.3 Cluster Shape

K-Means generally works best when clusters are:

- Compact
- Approximately spherical
- Relatively well-separated

It can struggle with:

- Curved or crescent-shaped clusters
- Elongated clusters
- Clusters with very different densities
- Clusters with strongly different sizes

This limitation occurs because K-Means assigns points based on distance to centroids.

For example, two crescent-shaped clusters may not be separated correctly by nearest-centroid assignment.

Algorithms such as DBSCAN or Spectral Clustering may be more appropriate for certain non-spherical structures.

### 6.4 Sensitivity to Outliers

K-Means calculates centroids using the mean of assigned points.

The mean is sensitive to extreme values.

Example:

Points: `2, 3, 4`

Mean:

`(2 + 3 + 4) / 3 = 3`

After adding an outlier:

`2, 3, 4, 100`

Mean:

`(2 + 3 + 4 + 100) / 4 = 27.25`

The centroid shifts significantly.

K-Means is sensitive to outliers because:

- Centroids are calculated using means.
- The objective function uses squared distances.
- Large distances contribute disproportionately to inertia.

---

## 7. Cluster Profiling, Interpreting Clusters, and Limitations of K-Means

### 7.1 Cluster Profiling

After clustering, each point receives a cluster label.

However, cluster labels alone do not explain what the clusters represent.

Cluster profiling summarizes the characteristics of each cluster using statistics such as:

- Mean
- Median
- Count
- Standard deviation
- Feature distributions
- Categorical feature proportions

This helps understand the differences between clusters.

### Example: Customer Segmentation

Suppose K-Means produces three clusters:

| Cluster | Average Income | Average Spending Score | Average Age |
|---|---:|---:|---:|
| 0 | Low | High | Young |
| 1 | High | Low | Older |
| 2 | High | High | Young |

Possible interpretations:

- Cluster 0: Young customers with lower income but high spending.
- Cluster 1: Customers with high income but lower spending.
- Cluster 2: Young customers with high income and high spending.

These interpretations are derived by examining cluster profiles.

K-Means itself does not assign semantic names such as "high-value customers."

### 7.2 Interpreting Clusters

Cluster interpretation involves:

1. Comparing feature statistics across clusters.
2. Identifying features that distinguish one cluster from another.
3. Understanding whether the differences are meaningful.
4. Connecting the cluster characteristics to the problem's objective.

For example, a business may use customer clusters to design different marketing strategies.

However, cluster membership alone does not prove customer value or predict behavior.

### 7.3 Limitations of K-Means

**1. K must be specified beforehand**

K-Means does not automatically discover the correct number of clusters.

Elbow Method and Silhouette Score can help select a reasonable K.

**2. Sensitive to initialization**

Different initial centroids may produce different results.

K-Means++ helps reduce this risk.

**3. Sensitive to outliers**

Extreme values can shift centroids and affect cluster assignments.

**4. Assumes centroid-based cluster structure**

K-Means works best for compact, approximately spherical clusters.

It may struggle with irregular or non-spherical clusters.

**5. Requires numerical features**

K-Means relies on distance calculations and mean-based centroids.

Categorical features cannot generally be used directly without an appropriate representation or alternative clustering algorithm.

**6. Feature scaling can affect results**

Features with larger numerical ranges may dominate distance calculations.

**7. Cluster labels have no inherent meaning**

Cluster 0, Cluster 1, and Cluster 2 are arbitrary identifiers.

Their meaning must be interpreted from the data.

**8. Good geometric clustering does not guarantee business usefulness**

A high Silhouette Score indicates well-separated clusters, but it does not prove that those clusters are useful for the business problem.

The quality of clustering must be evaluated in the context of the actual objective.

---

## Final Summary

K-Means groups data points by assigning them to the nearest centroid and updating centroids using the mean.

The main concepts to remember:

- **Inertia:** Measures within-cluster squared distances.
- **Elbow Method:** Helps identify a reasonable K by observing the reduction in inertia.
- **Silhouette Score:** Evaluates cohesion and separation.
- **K-Means++:** Improves centroid initialization.
- **Scaling:** Prevents large numerical ranges from dominating distances.
- **Cluster Profiling:** Helps understand what each cluster represents.
- **Limitations:** K-Means is sensitive to initialization, outliers, feature scaling, and cluster shape.

Ultimately, K-Means creates clusters based on geometric similarity. Understanding whether those clusters are meaningful requires evaluating the data, cluster profiles, and the problem's objective.