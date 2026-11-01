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

# DBSCAN (Density-Based Spatial Clustering of Applications with Noise)

## 1. Introduction to DBSCAN

DBSCAN stands for **Density-Based Spatial Clustering of Applications with Noise**.

It is an unsupervised machine learning algorithm used to group data points based on their density.

Unlike K-Means, DBSCAN does not require the number of clusters to be specified beforehand.

It identifies clusters by finding regions where points are densely located and separates them from regions with low point density.

DBSCAN can also identify noise points that do not belong to any cluster.

### Key Characteristics

- Does not require the number of clusters beforehand.
- Can identify clusters of arbitrary shapes.
- Can identify noise and outliers.
- Uses density-based connectivity rather than centroids.
- Depends on two important parameters: `eps` and `min_samples`.

---

## 2. How DBSCAN Works

DBSCAN determines whether a point belongs to a cluster based on the number of neighboring points within a specified radius.

It uses two parameters:

### 2.1 eps (Epsilon)

`eps` defines the maximum distance between two points for them to be considered neighbors.

If the distance between two points is less than or equal to `eps`, they are considered neighbors.

For example:

If `eps = 2`, then all points within a distance of 2 from a particular point are considered its neighbors.

### 2.2 min_samples

`min_samples` defines the minimum number of points required within the `eps` neighborhood for a point to qualify as a Core Point.

In scikit-learn, the point itself is included in this count.

For example:

```python
eps = 2
min_samples = 4
```

A point must have at least 4 points in its `eps` neighborhood, including itself, to be classified as a Core Point.

---

## 3. Types of Points in DBSCAN

DBSCAN classifies points into three categories:

1. Core Points
2. Border Points
3. Noise Points

### 3.1 Core Point

A point is a Core Point if its `eps` neighborhood contains at least `min_samples` points, including itself.

A Core Point lies in a sufficiently dense region.

Example:

```text
eps = 2
min_samples = 4
```

If point A has 5 points in its `eps` neighborhood, including itself, A is a Core Point.

### 3.2 Border Point

A point is a Border Point if:

- It is not a Core Point.
- It lies within the `eps` neighborhood of at least one Core Point.

A Border Point belongs to a cluster but cannot expand the cluster.

This means that DBSCAN does not use a Border Point to discover additional neighboring points during cluster expansion.

### 3.3 Noise Point

A point is classified as Noise if it is not a Core Point and is not density-reachable from any Core Point.

A Noise Point does not belong to any discovered cluster.

However, a point initially classified as Noise may later become a Border Point when DBSCAN expands a cluster.

---

## 4. Density-Based Relationships

DBSCAN uses density-based relationships to determine which points belong to the same cluster.

### 4.1 Directly Density-Reachable

A point B is directly density-reachable from point A if:

- A is a Core Point.
- B lies within the `eps` neighborhood of A.

The relationship is directional because A must be a Core Point.

### 4.2 Density-Reachable

A point B is density-reachable from point A if there exists a sequence of points connecting A to B such that each point in the sequence is directly density-reachable from the preceding point.

For the intermediate points in this sequence, each must be a Core Point.

The final point can be a Core Point or a Border Point.

Example:

```text
Core A -> Core B -> Core C -> Border D
```

D is density-reachable from A through the sequence of Core Points.

### 4.3 Density-Connected

Two points A and B are density-connected if there exists a point O from which both A and B are density-reachable.

Density-connectedness is symmetric.

This relationship helps define the complete cluster: points belong to the same cluster when they are connected through sufficiently dense regions.

---

## 5. DBSCAN Algorithm: Step-by-Step

DBSCAN processes the dataset point by point.

### Step 1: Initialize

Initially, all points are unvisited.

### Step 2: Select an Unvisited Point

Select any unvisited point P and mark it as visited.

### Step 3: Find Its Neighbors

Find all points within the `eps` neighborhood of P.

This neighborhood includes P itself.

### Step 4: Check Whether P Is a Core Point

If the number of neighbors is less than `min_samples`:

- Mark P as Noise provisionally.
- Continue with the next unvisited point.

Otherwise:

- P is a Core Point.
- Create a new cluster.
- Add P to that cluster.

### Step 5: Expand the Cluster

For every point in the neighborhood of P:

- If the point is unvisited, mark it as visited and find its neighborhood.
- If it is a Core Point, add its neighbors to the cluster expansion process.
- If it is not a Core Point but is within the neighborhood of a Core Point, add it to the cluster as a Border Point.

### Step 6: Continue Expansion

Continue processing the newly discovered Core Points and their neighborhoods.

The cluster expands until no additional density-reachable points can be added.

### Step 7: Move to the Next Unvisited Point

After the current cluster has been fully expanded, select another unvisited point and repeat the process.

### Step 8: Terminate

The algorithm terminates when all points have been processed.

At this point:

- All discovered clusters have been expanded.
- Points that could not be assigned to a cluster remain Noise.

### Important Note

DBSCAN does not use an iterative optimization procedure like K-Means.

It does not repeatedly update centroids or minimize a cost function.

Its termination occurs after all points have been processed and all cluster expansions are complete.

---

## 6. Noise Reclassification During Cluster Expansion

A point may initially be classified as Noise because it does not have enough neighbors to qualify as a Core Point.

Later, when DBSCAN expands a cluster from a neighboring Core Point, that previously marked Noise point may be added to the cluster as a Border Point.

Therefore:

**Noise is a provisional classification during the algorithm's execution.**

It is not necessary to revisit the point and recalculate its neighborhood.

The point can be added to a cluster when it is encountered during expansion.

---

## 7. DBSCAN Example

Consider a dataset containing the following points:

```text
A, B, C, D, E, F, G, H
```

Assume:

```python
eps = 2
min_samples = 4
```

Suppose the neighborhood counts are:

| Point | Number of neighbors | Classification |
|---|---:|---|
| A | 5 | Core |
| B | 4 | Core |
| C | 4 | Core |
| D | 2 | Border |
| E | 1 | Noise |
| F | 5 | Core |
| G | 4 | Core |
| H | 2 | Border |

### Cluster Expansion

Suppose A is selected first.

A is a Core Point, so DBSCAN creates Cluster 1.

During expansion:

- B is found and is a Core Point.
- C is found and is a Core Point.
- D is found and is a Border Point.

The cluster expands through A, B, and C.

D is included in the cluster but does not expand it.

If E is not reachable from any Core Point, it remains Noise.

When DBSCAN reaches F, it may create another cluster, Cluster 2, and expand it through G and H.

### Final Result

```text
Cluster 1: A, B, C, D
Cluster 2: F, G, H
Noise: E
```

This example illustrates how Core Points expand clusters, Border Points are included without expanding clusters, and Noise points remain outside clusters.

---

## 8. Advantages of DBSCAN

### 8.1 No Need to Specify the Number of Clusters

Unlike K-Means, DBSCAN does not require the number of clusters to be specified beforehand.

It discovers clusters based on density.

### 8.2 Can Identify Arbitrary-Shaped Clusters

DBSCAN can identify clusters with irregular shapes.

For example:

- Circular clusters
- Crescent-shaped clusters
- Ring-shaped clusters

This is possible because DBSCAN uses density connectivity rather than distance from a centroid.

### 8.3 Can Identify Noise and Outliers

DBSCAN can identify points that do not belong to any cluster.

It does not force every point into a cluster.

### 8.4 Does Not Require Centroids

DBSCAN does not calculate centroids or minimize the distance between points and cluster centers.

### 8.5 Useful for Density-Based Data

It can work effectively when clusters are separated by regions of low density.

---

## 9. Disadvantages of DBSCAN

### 9.1 Sensitive to eps and min_samples

DBSCAN's results depend heavily on its parameters.

If `eps` is too small:

- Many points may become Noise.
- Clusters may split into smaller clusters.

If `eps` is too large:

- Different clusters may merge.
- Cluster boundaries may become less meaningful.

If `min_samples` is too small:

- More points may qualify as Core Points.

If `min_samples` is too large:

- Fewer points may qualify as Core Points.
- More points may become Noise.

### 9.2 Difficulty with Different Cluster Densities

DBSCAN uses a single `eps` value for the entire dataset.

If one cluster is dense and another is sparse, the same `eps` may not work well for both.

A small `eps` may identify the dense cluster but classify points from the sparse cluster as Noise.

A larger `eps` may identify the sparse cluster but merge nearby dense clusters.

### 9.3 Sensitive to Feature Scaling

DBSCAN uses distance calculations.

If features have different numerical ranges, features with larger ranges may dominate the distance calculation.

Therefore, feature scaling is often necessary.

### 9.4 Challenges with High-Dimensional Data

As dimensionality increases, distance calculations may become less meaningful.

This is related to the curse of dimensionality.

Consequently, DBSCAN may struggle to identify meaningful dense neighborhoods in high-dimensional datasets.

### 9.5 Parameter Selection Can Be Difficult

Choosing a suitable `eps` value is not always straightforward.

A k-distance plot can help, but selecting the appropriate value may still require experimentation.

### 9.6 Computational Cost

DBSCAN's computational complexity depends on the implementation and neighborhood-search method.

With efficient spatial indexing, it can perform well in favorable low-dimensional cases.

In the worst case, its time complexity can reach O(n²).

---

## 10. Scaling in DBSCAN

DBSCAN uses distance calculations to identify neighboring points.

Consider a dataset with these features:

| Feature | Range |
|---|---|
| Age | 18–60 |
| Income | 100,000–2,000,000 |

Income may dominate the distance calculation because of its much larger numerical range.

This can cause the neighborhood structure to be determined primarily by income.


---

## 11. Parameter Sensitivity

### Effect of eps

| eps | Possible Effect |
|---|---|
| Too small | More Noise, fragmented clusters |
| Appropriate | Meaningful density-based clusters |
| Too large | Clusters may merge |

### Effect of min_samples

| min_samples | Possible Effect |
|---|---|
| Too small | More points qualify as Core Points |
| Appropriate | Suitable density threshold |
| Too large | Fewer Core Points, more Noise |

The best combination depends on the dataset and the desired definition of density.

---

## 12. Variable-Density Clusters

DBSCAN can struggle when different clusters have substantially different densities.

For example:

```text
Cluster A: Dense
Cluster B: Sparse
```

A single `eps` value may not identify both clusters properly.

### Alternative Algorithms

- HDBSCAN
- OPTICS

These algorithms are designed to handle varying density structures more effectively than standard DBSCAN in many situations.

---

## 13. High-Dimensional Data Limitations

DBSCAN depends on the distance between points.

In high-dimensional spaces:

- Distances may become less discriminative.
- Neighborhood selection becomes difficult.
- Density estimation becomes challenging.
- Many points may be classified as Noise.

### Possible Solutions

- Feature selection
- Dimensionality reduction
- PCA
- Removing irrelevant features
- Using meaningful domain-specific features

These techniques may improve clustering, but they do not guarantee that DBSCAN will work well in high-dimensional data.


---

## 14. Key Points to Remember

- DBSCAN stands for Density-Based Spatial Clustering of Applications with Noise.
- It uses `eps` and `min_samples`.
- Core Points can expand clusters.
- Border Points belong to clusters but do not expand them.
- Noise Points are not currently density-reachable from Core Points.
- A point initially marked Noise can later become a Border Point.
- Density-reachable is directional.
- Density-connected is symmetric.
- DBSCAN does not require the number of clusters beforehand.
- It can identify arbitrary-shaped clusters.
- It struggles with clusters having different densities.
- Feature scaling is often important.
- DBSCAN does not use a centroid-based optimization or convergence process.
- The algorithm terminates when all points have been processed and all clusters have been expanded.

## DBSCAN: Parameter Selection for coding purpose

### 1. Selecting `min_samples`

- A common starting guideline is `min_samples >= number of features + 1`.
- Higher `min_samples` requires denser neighborhoods to qualify as Core Points.
- Lower `min_samples` allows less dense regions to qualify as Core Points.
- Try different values based on dataset characteristics and the desired density.

### 2. Selecting `eps`

- `eps` defines the maximum distance between neighboring points.
- Use a k-distance plot to help select a suitable `eps`.
- Set `k = min_samples` and calculate each point's k-th nearest neighbor distance.
- Sort and plot these distances in ascending order.
- Identify the elbow (sharp rise) in the plot.
- Choose an `eps` candidate around the elbow, then validate by visualizing clustering results.

### Why this approach?

- `min_samples` defines the density requirement for Core Points.
- The k-distance plot helps identify a distance threshold separating dense regions from sparse regions.
- The elbow provides a useful `eps` estimate, but it is a heuristic, not a guaranteed optimal value.

### Important

- Parameter selection depends on data characteristics and the clustering objective.
- Feature scaling is important when features have different numerical ranges.
- Validate parameter choices by checking cluster structure and noise points.
