# Hierarchical Clustering

## 1. Introduction to Hierarchical Clustering

Hierarchical Clustering is an unsupervised machine learning algorithm used to group similar data points into clusters.

It creates a hierarchy of clusters rather than directly producing a single fixed number of clusters.

The algorithm organizes data points into a tree-like structure, where:

- Similar data points are grouped together.
- Clusters are progressively merged or divided.
- The hierarchy can be visualized using a dendrogram.
- The final number of clusters can be selected by cutting the hierarchy at a suitable level.

### Example

Suppose a company has customer data containing:

- Annual Income
- Spending Score

The objective is to identify customer segments such as:

- High-income, high-spending customers
- High-income, low-spending customers
- Low-income, high-spending customers
- Low-income, low-spending customers

Hierarchical Clustering can identify groups of customers based on the similarity of their feature values.

It does not require predefined cluster labels during the clustering process.

---

## 2. Types of Hierarchical Clustering

Hierarchical Clustering is divided into two main types.

### 2.1 Agglomerative Clustering

Agglomerative Clustering follows a bottom-up approach.

Initially, every data point is treated as an individual cluster.

The algorithm repeatedly merges the closest clusters until all data points belong to a single cluster or the required number of clusters is reached.

### 2.2 Divisive Clustering

Divisive Clustering follows a top-down approach.

Initially, all data points belong to one cluster.

The algorithm repeatedly divides clusters into smaller clusters until the desired structure is obtained.

### Difference

| Agglomerative | Divisive |
|---|---|
| Bottom-up approach | Top-down approach |
| Starts with individual data points | Starts with one large cluster |
| Repeatedly merges clusters | Repeatedly splits clusters |
| Commonly used in practical implementations | Less commonly used |

Our implementation uses **Agglomerative Clustering**.

---

## 3. Agglomerative Clustering

Agglomerative Clustering is a hierarchical clustering algorithm that progressively merges clusters based on their distance.

### Basic idea

Initially:

- Each data point forms its own cluster.
- The algorithm calculates the distance between clusters.
- The closest clusters are merged.
- The process continues until the required number of clusters is obtained.

### Important characteristics

- It is an unsupervised learning algorithm.
- It does not require target labels.
- It can generate a hierarchy of clusters.
- The final clustering depends on the linkage method and the selected number of clusters.
- It is a distance-based algorithm.

---

## 4. How Agglomerative Clustering Works

The general process is:

1. Treat every data point as an individual cluster.
2. Calculate the distances between clusters.
3. Identify the two closest clusters.
4. Merge those clusters.
5. Recalculate the distances between the newly formed cluster and the remaining clusters.
6. Repeat the merging process.
7. Stop when the required number of clusters is obtained or all points are merged into one cluster.

### Example

Suppose we have four data points:

A, B, C, and D.

Initially:

- Cluster 1 = A
- Cluster 2 = B
- Cluster 3 = C
- Cluster 4 = D

If A and B are the closest points, they are merged:

- Cluster 1 = {A, B}
- Cluster 2 = C
- Cluster 3 = D

The algorithm continues merging clusters based on the selected linkage method.

Eventually, the hierarchy can be cut to obtain a desired number of clusters.

---

## 5. Distance Metrics

Distance metrics measure how far apart data points are.

Hierarchical Clustering commonly uses distance metrics to determine similarity between observations.

### 5.1 Euclidean Distance

Euclidean distance is the straight-line distance between two points.

For two points:

A = (x1, y1)

B = (x2, y2)

The Euclidean distance is:

d(A, B) = sqrt((x2 - x1)^2 + (y2 - y1)^2)

It is commonly used for numerical features.

### 5.2 Why distance matters

The algorithm uses distances to decide which clusters should be merged.

If the distance between two clusters is small, they are considered closer according to the selected distance and linkage criteria.

### 5.3 Pairwise Distance Matrix

A pairwise distance matrix contains the distances between every pair of data points.

For n data points, it contains n × n distance values.

Properties:

- The diagonal contains zeros because each point has zero distance from itself.
- The matrix is symmetric for Euclidean distance.
- The distance between A and B is equal to the distance between B and A.

In our implementation, the pairwise Euclidean distance matrix was calculated to understand the distances between observations.

Note: The pairwise distance matrix is useful for understanding the data, but the clustering algorithm does not require us to manually calculate it beforehand.

---

## 6. Linkage Methods

Linkage determines how the distance between two clusters is calculated.

Different linkage methods can produce different cluster structures from the same dataset.

We explored four linkage methods:

- Single
- Complete
- Average
- Ward

### 6.1 Single Linkage

Single linkage calculates the distance between two clusters using the minimum distance between any pair of points belonging to different clusters.

Formula:

D(A, B) = min d(a, b)

where:

- a belongs to cluster A
- b belongs to cluster B

#### Characteristics

- Merges clusters based on their closest points.
- Can identify elongated or irregularly shaped clusters.
- Can be sensitive to noise.
- May produce the chaining effect.

#### Chaining Effect

The chaining effect occurs when clusters are progressively connected through individual points or small groups of points.

This can result in a large cluster containing observations that are not necessarily close to one another overall.

In our implementation, Single Linkage with K = 4 produced a cluster containing only one observation.

This is an example of a potential issue with the resulting cluster structure.

### 6.2 Complete Linkage

Complete linkage calculates the distance between two clusters using the maximum distance between any pair of points belonging to different clusters.

Formula:

D(A, B) = max d(a, b)

#### Characteristics

- Considers the farthest pair of points between clusters.
- Generally favors compact clusters.
- Avoids some of the chaining behavior associated with Single Linkage.
- Can be sensitive to outliers because the maximum distance is considered.

### 6.3 Average Linkage

Average linkage calculates the average distance between all pairs of points belonging to two different clusters.

Formula:

D(A, B) = Average of d(a, b)

#### Characteristics

- Considers all pairwise distances between clusters.
- Provides a compromise between Single and Complete Linkage.
- Does not rely exclusively on the closest or farthest pair of points.

### 6.4 Ward Linkage

Ward linkage merges clusters based on the increase in within-cluster sum of squared errors (WCSS) caused by the merge.

It favors merges that produce the smallest increase in within-cluster variance.

#### Characteristics

- Often produces compact and relatively balanced clusters.
- Uses a variance-minimization criterion.
- Commonly used with Euclidean distance.
- Can be suitable when the goal is to identify compact numerical clusters.

Ward linkage is not simply the minimum or maximum distance between points.

It considers the increase in within-cluster sum of squares resulting from merging two clusters.

---

## 7. Dendrogram

A dendrogram is a tree-like visualization of the hierarchy created by Hierarchical Clustering.

It represents:

- Which clusters were merged.
- The order in which clusters were merged.
- The distance or linkage cost at which each merge occurred.
- The hierarchy of clusters.

### Important components

#### Leaves

The leaves represent individual observations.

#### Branches

Branches show how observations or clusters are connected through merges.

#### Merge Height

The height of a merge represents the distance or linkage criterion value at which two clusters were combined.

The interpretation of this height depends on the selected linkage method.

### Interpreting a dendrogram

A dendrogram can help identify candidate numbers of clusters.

If a dendrogram contains a large vertical gap between successive merge heights, cutting the tree within that gap may produce a reasonable candidate clustering.

However:

- A large gap does not guarantee the best clustering.
- The selected K should also be evaluated using cluster quality and domain context.
- Different linkage methods can produce different dendrogram structures.

### Dendrograms in our implementation

We generated dendrograms for:

- Single Linkage
- Complete Linkage
- Average Linkage
- Ward Linkage

We visually inspected the merge heights and identified K = 3 and K = 4 as candidate cluster counts.

These were candidates for evaluation, not automatically the final number of clusters.

---

## 8. Choosing the Number of Clusters (K)

The number of clusters is not automatically determined by Hierarchical Clustering.

A dendrogram provides a hierarchy, and a cut must be selected to obtain the final clusters.

### Candidate K

A candidate K is a possible number of clusters selected for further evaluation.

In our implementation, we selected:

- K = 3
- K = 4

These values were chosen after visually inspecting the dendrograms.

### Why evaluate multiple K values?

Different values of K produce different groupings.

For example:

- K = 3 produces three clusters.
- K = 4 produces four clusters.

Increasing K can split a larger cluster into smaller groups.

The objective is not simply to maximize the number of clusters or minimize the distance within clusters.

The objective is to obtain meaningful, reasonably separated, and interpretable groups.

---

## 9. Evaluating Clusters

After generating cluster labels, we evaluated the resulting cluster structures using:

- Silhouette Score
- WCSS
- Cluster sizes

These metrics provide different information about the clustering result.

### 9.1 Silhouette Score

The Silhouette Score evaluates how well a data point fits within its assigned cluster compared with other clusters.

It considers:

- Cohesion: how close a point is to other points in its own cluster.
- Separation: how far a point is from points in neighboring clusters.

For an individual data point:

s = (b - a) / max(a, b)

where:

- a = average distance from the point to other points in its own cluster
- b = average distance from the point to points in the nearest neighboring cluster

#### Interpretation

| Silhouette Score | Interpretation |
|---|---|
| Close to +1 | Well-separated clusters |
| Close to 0 | Points may lie near cluster boundaries |
| Negative | Points may be assigned to an inappropriate cluster |

The overall Silhouette Score is the average of the individual scores.

A higher score generally indicates better cohesion and separation.

However, it does not guarantee that the clusters are useful for the business problem.

### 9.2 WCSS

WCSS stands for Within-Cluster Sum of Squares.

It measures the sum of squared distances between each data point and its assigned cluster centroid.

Formula:

WCSS = Sum of squared distances of observations from their respective cluster centroids.

A lower WCSS indicates that the data points are closer to their cluster centroids.

#### Important consideration

WCSS generally decreases as the number of clusters increases.

Therefore, the smallest WCSS does not automatically indicate the best clustering.

For example:

- K = 3 produces a certain WCSS.
- K = 4 may produce a lower WCSS.
- K = 5 may produce an even lower WCSS.

This happens because increasing the number of clusters can reduce the within-cluster distances.

The WCSS can be used to compare candidate cluster counts, but it should not be interpreted alone.

The elbow method can be used to examine how WCSS changes as K increases.

### WCSS and Ward Linkage

Ward linkage specifically uses the increase in within-cluster sum of squares when merging clusters.

However, calculating WCSS after clustering is also possible for other linkage methods.

This does not mean those linkage methods directly optimize WCSS.

---

## 10. Feature Scaling in Hierarchical Clustering

Feature scaling is important when clustering uses distance-based calculations.

Consider two features:

- Annual Income measured in thousands of dollars.
- Spending Score measured from 1 to 100.

If the features have different scales or ranges, one feature can contribute more strongly to the distance calculation.

### Why scaling matters

Euclidean distance depends on the magnitude of feature values.

A feature with a larger numerical scale may dominate the distance even if it is not more important.

This can influence:

- Pairwise distances.
- Cluster merging.
- Dendrogram structure.
- Final cluster assignments.
- Silhouette Score and WCSS.

### Standardization

Standardization transforms each feature to have approximately:

- Mean = 0
- Standard deviation = 1

Formula:

z = (x - mean) / standard deviation

This allows features with different scales to contribute more comparably to distance calculations.

### Standardization in our implementation

We added standardization using StandardScaler.

The standardized dataset is stored separately as X_scaled.

The original X remains unchanged.

**Important:** The dendrograms, cluster labels, Silhouette Scores, and WCSS calculated before standardization were based on the original X.

Those results need to be regenerated using X_scaled before making the final clustering decision.

---

## 11. Role of X and y

In our synthetic dataset, we generated X and y using make_blobs.

### X

X contains the feature values used for clustering.

In our implementation, X contains 100 observations and 2 features.

The features represent:

- Annual Income
- Spending Score

Hierarchical Clustering uses X to calculate distances and create clusters.

### y

y contains the original cluster labels generated by make_blobs.

The labels represent the known groups from the synthetic data generation process.

They are not required by the clustering algorithm.

### Why y was not used during clustering

Hierarchical Clustering is an unsupervised learning algorithm.

It creates clusters using the feature values without relying on target labels.

Therefore:

- X is used to perform clustering.
- y is not used to fit the clustering algorithm.
- y can be used afterward to compare the generated clusters against the original synthetic groups.

This comparison can help evaluate how well the algorithm recovered the original group structure.

---

## 12. Train-Test Split in Clustering

Train-test split is commonly used in supervised machine learning.

For example:

- Training data is used to fit a model.
- Test data is used to evaluate predictions on unseen observations.

However, clustering is an unsupervised learning task.

### Is train-test split mandatory?

No.

For exploratory clustering, the algorithm is often applied to the entire dataset to discover its structure.

Train-test split is not mandatory in the same way as a typical supervised learning workflow.

### Important difference

AgglomerativeClustering in scikit-learn does not provide the standard predict() method used by many supervised estimators.

Therefore, the usual workflow:

Training data → Fit → Test data → Predict

does not directly apply in the same way.

### When train-test split may be useful

A split can be useful when the objective is to assess clustering stability or develop a strategy for assigning new observations to existing clusters.

However, the assignment of new observations requires an appropriate method because standard AgglomerativeClustering does not directly predict labels for unseen samples.

### In our implementation

We used the full synthetic dataset to explore the cluster structure.

A train-test split was not required for this exploratory exercise.

---

## 13. Implementation Workflow

The practical implementation followed this sequence.

### Step 1: Generate a synthetic dataset

We generated a dataset containing:

- 100 observations
- 2 numerical features
- 4 original groups

The dataset was created using make_blobs.

### Step 2: Understand the data

We created a DataFrame containing:

- Annual Income (k$)
- Spending Score (1-100)

We visualized the data using a scatter plot.

### Step 3: Calculate pairwise distances

We calculated the Euclidean distance between observations to understand the distance-based nature of clustering.

### Step 4: Generate a dendrogram

We generated a dendrogram using Ward linkage.

We also generated dendrograms using:

- Single
- Complete
- Average
- Ward

### Step 5: Identify candidate K values

After examining the dendrograms, we selected:

- K = 3
- K = 4

for further evaluation.

### Step 6: Generate cluster labels

We generated cluster labels using the combinations of linkage method and K.

### Step 7: Evaluate cluster quality

We calculated:

- Silhouette Score
- WCSS
- Cluster sizes

### Step 8: Standardize features

We added StandardScaler and stored the standardized data separately.

### Step 9: Re-run clustering

The clustering analysis should be repeated using the standardized features.

### Step 10: Final visualization and profiling

The final selected clustering should be visualized and the cluster profiles should be examined to understand the characteristics of each cluster.

---

## 14. Practical Implementation Summary

### Dataset

The synthetic dataset was generated with:

- 100 observations
- 2 numerical features
- 4 original groups
- Cluster standard deviation of 1.2

### Candidate cluster counts

We considered:

- K = 3
- K = 4

### Linkage methods

We explored:

- Single
- Complete
- Average
- Ward

### Results obtained before standardization

| Linkage | K | Silhouette Score | WCSS |
|---|---:|---:|---:|
| Complete | 4 | 0.751211 | 244.281728 |
| Average | 4 | 0.751211 | 244.281728 |
| Ward | 4 | 0.751211 | 244.281728 |
| Single | 3 | 0.735817 | 727.756455 |
| Average | 3 | 0.735817 | 727.756455 |
| Complete | 3 | 0.735817 | 727.756455 |
| Ward | 3 | 0.735817 | 727.756455 |
| Single | 4 | 0.573880 | 718.042458 |

### Interpretation

The results showed that K = 4 had a higher Silhouette Score than K = 3.

The WCSS was also lower for K = 4.

However, WCSS generally decreases as K increases, so this alone does not establish that K = 4 is better.

The cluster sizes also showed that Single Linkage with K = 4 produced one singleton cluster.

This was a potential concern for the resulting clustering.

### Important correction

The results above were obtained before standardization.

They are exploratory results and should not be treated as the final evaluation.

After standardization, the dendrograms, cluster labels, evaluation metrics, and cluster sizes may change.

The final choice of K and linkage method should be made after reviewing the standardized results.

---

## 15. Advantages and Limitations

### Advantages

- Does not require predefined cluster labels.
- Produces a hierarchy of clusters.
- Dendrogram provides a useful visualization.
- Multiple linkage methods are available.
- Can be useful for exploratory data analysis.
- Allows different cluster counts to be obtained from the same hierarchy.

### Limitations

- Can be computationally expensive for large datasets.
- Sensitive to the choice of distance metric.
- Sensitive to the linkage method.
- Can be affected by feature scaling.
- Does not automatically determine the best number of clusters.
- Standard AgglomerativeClustering does not provide a direct predict method for unseen observations.
- Different linkage methods can produce different cluster structures.

---

## Linkage Methods: When to Use Which?

Linkage methods determine how the distance between two clusters is calculated during Hierarchical Clustering.

Different linkage methods can produce different cluster structures from the same dataset.

### 1. Single Linkage

Single Linkage calculates the distance between two clusters using the minimum distance between any pair of points belonging to different clusters.

**Suitable for:**
- Irregularly shaped clusters
- Elongated clusters
- Non-spherical clusters
- Clusters connected through nearby points

**Limitations:**
- Can produce the chaining effect.
- May connect clusters through noise or outliers.
- Can result in poorly separated clusters.

**Example:** Identifying elongated or irregular groups of observations.

### 2. Complete Linkage

Complete Linkage calculates the distance between two clusters using the maximum distance between any pair of points belonging to different clusters.

**Suitable for:**
- Compact clusters
- Well-separated clusters
- Relatively spherical or rounded clusters
- Clusters where maximum within-group spread should be controlled

**Limitations:**
- Sensitive to outliers.
- Can split elongated clusters.
- May not perform well when clusters have very different sizes.

**Example:** Customer segmentation where compact groups are desired.

### 3. Average Linkage

Average Linkage calculates the average distance between all pairs of points belonging to two different clusters.

**Suitable for:**
- Compact or moderately elongated clusters
- Clusters with a relatively balanced structure
- Situations where a compromise between Single and Complete Linkage is desired

**Limitations:**
- Results depend on the distribution of observations.
- May not identify highly irregular cluster structures as effectively as Single Linkage.

**Example:** General-purpose clustering where neither the nearest nor the farthest pair alone should determine the merge.

### 4. Ward Linkage

Ward Linkage merges clusters based on the smallest increase in within-cluster sum of squares (WCSS).

**Suitable for:**
- Compact clusters
- Approximately spherical clusters
- Clusters with relatively similar sizes
- Well-separated groups
- Numerical datasets where within-cluster variance should be minimized

**Limitations:**
- Not suitable for strongly elongated or irregularly shaped clusters.
- Can struggle with clusters having very different sizes or densities.
- Typically used with Euclidean distance.

**Example:** Customer segmentation using numerical features such as income and spending score.

---

## Linkage Methods and Cluster Shapes

| Cluster Structure | Suitable Method | Explanation |
|---|---|---|
| Spherical / Rounded | Ward, Complete | These methods tend to favor compact groups. |
| Elongated | Single | Can merge clusters based on their closest neighboring points. |
| Irregular / Non-spherical | Single | Can capture irregular shapes, but may suffer from chaining. |
| Compact and Well-separated | Ward, Complete | Favor compact clusters with relatively small within-cluster spread. |
| Clusters with similar sizes | Ward | Often works well when clusters have relatively similar sizes. |
| Clusters with different sizes | Single may capture irregular structures | Ward and Complete can struggle with substantial size differences. |
| Clusters with noise / outliers | No linkage is universally robust | Single can chain; Complete and Ward may be influenced by outliers. |
| General-purpose / Balanced approach | Average | Uses the average of all pairwise distances between clusters. |

**Important:** These are tendencies, not strict rules. Cluster shape alone does not determine the best linkage method. Feature scaling, density, noise, distance metric, and the clustering objective also matter.


---

## K-Means vs Hierarchical Clustering: Comparison

| Feature | K-Means | Hierarchical Clustering |
|---|---|---|
| Approach | Centroid-based | Hierarchy-based |
| Cluster formation | Assigns points to centroids | Merges or splits clusters |
| Number of clusters | K specified beforehand | Can explore hierarchy before selecting K |
| Output | K clusters | Hierarchical tree and selected clusters |
| Main objective | Minimize WCSS | Depends on linkage method |
| Visualization | Scatter plot, centroids | Dendrogram, scatter plot |
| Scalability | Generally better for large datasets | Generally more expensive for large datasets |
| New observations | Standard predict method available | Standard AgglomerativeClustering has no direct predict method |
| Cluster shape | Generally favors compact, spherical clusters | Depends on linkage method |
| Feature scaling | Important | Important |

---

## Which Algorithm Is Suitable for Which Cluster Structure?

| Cluster Structure | K-Means | Hierarchical Clustering |
|---|---|---|
| Spherical clusters | Suitable | Ward or Complete Linkage |
| Compact and well-separated clusters | Suitable | Ward or Complete Linkage |
| Elongated clusters | Generally not suitable | Single Linkage may be suitable |
| Irregular / non-spherical clusters | Generally not suitable | Single Linkage may be suitable |
| Clusters with similar sizes | Suitable | Ward often works well |
| Clusters with very different sizes | May struggle | Depends on linkage; Single may capture irregular structures |
| Unknown number of clusters | Requires K selection | Dendrogram can help explore candidate K values |
| Large datasets | Generally preferable | Can be computationally expensive |

---

## Practical Selection Guide

### Use K-Means when:
- The dataset is large.
- The number of clusters is approximately known.
- The clusters are compact and approximately spherical.
- Fast clustering is required.

### Use Hierarchical Clustering when:
- The dataset is relatively small or medium-sized.
- The hierarchy of clusters is useful.
- The number of clusters is not known initially.
- Different linkage methods need to be explored.
- The dendrogram can help understand the cluster structure.

### Within Hierarchical Clustering:

- **Single Linkage:** Consider for elongated or irregularly shaped clusters.
- **Complete Linkage:** Consider for compact, well-separated clusters.
- **Average Linkage:** Consider as a balanced distance-based approach.
- **Ward Linkage:** Consider for compact, approximately spherical clusters with relatively similar sizes.

---

## Important Takeaways

- K-Means is centroid-based; Hierarchical Clustering is hierarchy-based.
- K-Means generally works best for compact, spherical clusters.
- Single Linkage can identify elongated and irregular cluster shapes but is susceptible to chaining.
- Complete Linkage tends to favor compact clusters and can be sensitive to outliers.
- Average Linkage considers all pairwise distances between clusters.
- Ward Linkage favors compact clusters by minimizing the increase in within-cluster variance.
- No single linkage method is best for every dataset.
- The final choice should be based on cluster structure, evaluation metrics, visualization, and domain requirements.

