# Ball Tree

## 1. Introduction to Ball Tree

Ball Tree is a tree-based data structure used for efficient nearest-neighbor search.

It organizes observations into groups represented by hyperspherical regions called **balls**.

Each ball is defined by:

- A center point (C)
- A radius (R)

All observations belonging to a ball lie within its radius from the center.

Mathematically:

`distance(X_i, C) <= R`

Where:
- `X_i` = an observation inside the ball
- `C` = center of the ball
- `R` = radius of the ball

Unlike KD Tree, which partitions the feature space using axis-aligned splitting dimensions, Ball Tree organizes observations using distance-based spherical bounds.

### 1.1 What Does a Ball Represent?

A ball is a hypersphere in a multidimensional space.

- 1D: An interval
- 2D: A circle
- 3D: A sphere
- Higher dimensions: A hypersphere

The center does not necessarily need to be an actual training observation.

The radius must be large enough to contain every observation assigned to that ball.

### 1.2 Why Ball Tree?

Brute-force nearest-neighbor search calculates the distance between the query and every training observation.

KD Tree reduces these calculations through axis-aligned spatial partitioning.

However, KD Tree can become inefficient when its axis-aligned regions provide weak pruning bounds.

Ball Tree uses center-and-radius bounds to determine whether an entire group of observations can be skipped.

This can be useful for certain data distributions and distance metrics.

---

## 2. Structure of Ball Tree

A Ball Tree is a binary tree.

Each internal node represents a ball containing a group of observations.

An internal node generally has two child nodes, each representing a smaller group of observations.

The child balls contain subsets of the parent node's observations.

### 2.1 Ball Representation

A ball is represented by:

`Ball = (Center, Radius)`

For a query point Q, let:

`d = distance(Q, C)`

Where C is the ball center.

The minimum possible distance from Q to any point inside the ball is:

`Lower Bound = max(0, d - R)`

The maximum possible distance is bounded by:

`Upper Bound = d + R`

These bounds help the algorithm perform efficient nearest-neighbor searches.

### 2.2 Important Characteristics

- Every ball contains its assigned observations.
- Child balls represent subsets of the parent observations.
- Ball regions can overlap.
- The center need not be an actual data point.
- The radius must cover all observations assigned to the ball.
- A leaf node stores actual observations rather than splitting them further.

---

## 3. Construction of Ball Tree

Ball Tree construction involves recursively organizing observations into smaller groups.

The general construction process is:

1. Start with all training observations in the root node.
2. Determine a ball that covers the observations.
3. Divide the observations into two child groups using the implementation's splitting strategy.
4. Construct child balls around these groups.
5. Repeat recursively until the leaf-size stopping condition is reached.

### 3.1 Node Center and Radius

For a group of observations, the center is calculated using the implementation's center calculation.

The radius must be sufficient to cover the observations.

Conceptually:

`Radius = max(distance(X_i, C))`

Where C is the node center and X_i represents observations assigned to that node.

This ensures:

`distance(X_i, C) <= R`

for every observation in that node.

---

## 4. Leaf Size

Leaf size determines how many observations are stored in leaf nodes.

Scikit-learn's Ball Tree uses the `leaf_size` parameter.

Example:

```python
from sklearn.neighbors import 
BallTree
tree = BallTree(
    X,
    leaf_size=10,
    metric="euclidean"
)
```

A smaller leaf size generally creates:

- More tree nodes.
- Smaller leaves.
- More tree traversal.
- Fewer observations to examine inside each visited leaf.

A larger leaf size generally creates:

- Fewer tree nodes.
- Larger leaves.
- Less tree traversal.
- More observations to examine inside each visited leaf.

Scikit-learn documents that the number of observations in a leaf is approximately between `leaf_size` and `2 * leaf_size`, except when the dataset itself is smaller than the leaf size.

Leaf size affects the trade-off between tree traversal and distance calculations.

It does not change the intended nearest-neighbor definition.

---

## 5. How Does Ball Tree Search Work?

Ball Tree uses the center and radius of each node to determine whether the node may contain a closer neighbor.

Suppose:

- Query point = Q
- Ball center = C
- Ball radius = R
- Distance from query to center = d

The minimum possible distance from the query to any point inside the ball is:

`Lower Bound = max(0, d - R)`

This is a lower bound, not necessarily the actual distance to the nearest training observation inside the ball.

### 5.1 Why Is It a Lower Bound?

Suppose:

`Q = 5`

`C = 0`

`R = 2`

The ball covers the interval:

`[-2, 2]`

Distance from Q to the center:

`d = |5 - 0| = 5`

Minimum possible distance:

`Lower Bound = 5 - 2 = 3`

The closest possible location in the ball is 2, which is at distance 3 from Q.

However, the actual observations might be located at 0 and 1.

Their distances from Q would be 5 and 4.

Therefore:

- Lower bound = 3
- Actual nearest observation distance = 4

The lower bound is not the actual nearest-neighbor distance.

---

## 6. Pruning in Ball Tree

Pruning means skipping an entire node or region when it cannot contain a better nearest-neighbor candidate.

Ball Tree uses the lower bound to decide whether a ball needs to be explored.

For KNN search, maintain the current K nearest candidates.

The distance to the current Kth nearest candidate is:

`Threshold = Current Kth Nearest Distance`

The pruning condition is:

`Lower Bound > Threshold`

If this condition is true, the entire ball can be skipped.

Why?

Because even the closest possible observation inside that ball is farther away than the current Kth nearest candidate.

Therefore, no observation inside that ball can improve the current result.

### 6.1 Pruning Example

Suppose:

`K = 2`

Current nearest candidates:

- Point A = distance 2
- Point B = distance 4

Therefore:

`Threshold = 4`

Now consider another ball.

Its lower bound is:

`Lower Bound = 7`

Comparison:

`7 > 4`

The entire ball can be pruned.

There is no need to calculate distances to its individual observations.

### 6.2 When Should a Ball Be Explored?

Suppose another ball has:

`Lower Bound = 3`

Current threshold:

`Threshold = 4`

Comparison:

`3 <= 4`

The ball must be explored because it may contain a point closer than the current second-nearest candidate.

### 6.3 Important Rule

| Condition | Action |
|---|---|
| Lower bound > Current Kth distance | Prune the ball |
| Lower bound <= Current Kth distance | Explore the ball |

Equality must be treated carefully because the ball may contain a point tied with the current Kth candidate.

---

## 7. Search Process Step by Step

A typical exact nearest-neighbor search follows this process:

1. Start at the root node.
2. Evaluate the child balls using their lower bounds.
3. Explore promising child nodes.
4. Continue recursively until reaching a leaf.
5. Calculate actual distances to observations stored in that leaf.
6. Update the current K nearest candidates.
7. Backtrack to examine other potentially useful balls.
8. Prune a ball when its lower bound is greater than the current Kth nearest distance.
9. Continue until every remaining unexplored ball can safely be pruned or has been explored.

The algorithm does not need to visit every leaf.

Its efficiency comes from eliminating entire groups of observations using geometric bounds.

---

## 8. Computational Complexity

Let:

- N = Number of training observations
- D = Number of dimensions

### 8.1 Construction Complexity

Construction generally involves:

- Computing node centers and radii.
- Partitioning observation indices.
- Recursively building child nodes.

Its cost depends on the implementation and data distribution.

Do not assume a guaranteed O(N log N) construction cost for every implementation and configuration.

### 8.2 Query Complexity

In favorable cases, Ball Tree can avoid examining many observations.

However, when pruning becomes ineffective, it may need to examine almost every training observation.

Worst-case distance computation can approach:

`O(N × D)`

### 8.3 Space Complexity

The tree stores:

- Training observations or their index representation.
- Node centers and radii.
- Tree structure and node metadata.

Overall storage is linear in the number of observations, apart from implementation-dependent tree overhead.

---

## 9. Advantages of Ball Tree

1. Uses center-and-radius bounds for pruning.
2. Can work effectively with certain non-Euclidean metrics.
3. Supports a broader range of metrics than Scikit-learn's KDTree.
4. Can perform well when spherical bounds represent the data distribution effectively.
5. Can avoid examining large groups of observations during nearest-neighbor search.

---

## 10. Limitations of Ball Tree

### 10.1 Curse of Dimensionality

As dimensionality increases, the feature space can become sparse.

Nearest and farthest distances may also become relatively similar under common data distributions.

This can reduce the effectiveness of distance-based pruning.

### 10.2 Weak Bounding Regions

If a ball has a large radius, its lower bound may be small.

Consequently, the algorithm may be unable to eliminate that ball.

### 10.3 Data Distribution

Ball Tree performance depends on how effectively its balls represent groups of observations.

Poorly separated or irregularly distributed data can produce weak pruning bounds.

### 10.4 Construction Overhead

Building a tree requires additional computations and memory.

For small datasets or workloads with few queries, brute-force search may be faster.

### 10.5 Not Guaranteed to Be Faster Than KD Tree

Ball Tree can outperform KD Tree for some datasets and metrics, but there is no universal winner.

Both should be benchmarked against brute-force search.

---

## 11. Ball Tree vs KD Tree vs Brute Force

| Property | Brute Force | KD Tree | Ball Tree |
|---|---|---|---|
| Data organization | No tree | Axis-aligned partitions | Center-radius balls |
| Search strategy | Examine all observations | Spatial pruning | Ball-bound pruning |
| Tree construction | Not required | Required | Required |
| Low-dimensional performance | Can be good | Often effective | Often effective |
| High-dimensional performance | Predictable exhaustive cost | Pruning may weaken | Pruning may weaken |
| Metric support | Broad | More restricted | Broader than KD Tree |
| Worst-case query | O(N × D) | O(N × D) | O(N × D) |

### When to Use Which?

**Brute Force**
- Small datasets.
- High-dimensional data where tree pruning is ineffective.
- Workloads where vectorized distance calculations are efficient.

**KD Tree**
- Low-dimensional numerical data.
- Axis-aligned spatial separation is effective.
- Supported distance metrics.

**Ball Tree**
- Data where spherical bounding regions provide useful lower bounds.
- Metrics supported by Ball Tree but not KD Tree.
- Workloads where benchmarking demonstrates better performance.
