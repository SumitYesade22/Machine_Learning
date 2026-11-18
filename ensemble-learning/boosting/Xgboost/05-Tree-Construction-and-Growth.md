# 5. Tree Construction and Growth

## 5.1 What is tree construction trying to do?

Once gradients and Hessians are available, XGBoost must decide:

-   Which feature should be used for a split?
-   Which threshold should be used?
-   Which candidate split gives the largest gain?
-   Should a node be split at all?
-   Which leaf should be expanded next?
-   When should tree growth stop?

For a candidate split, observations are divided into left and right
groups. Their gradients and Hessians are aggregated, and the split gain
is calculated.

## 5.2 Exact greedy split finding

The exact greedy approach considers candidate split positions based on
sorted feature values.

For a numerical feature:

1.  Sort observations by that feature.
2.  Move a candidate split position through the ordered values.
3.  Maintain cumulative gradient and Hessian sums for the left side.
4.  Obtain right-side sums from the parent totals minus the left totals.
5.  Calculate gain for each valid candidate.
6.  Choose the candidate with the highest gain, if it satisfies
    constraints.

The word *greedy* means the algorithm chooses the best local split at
the current node. It does not search every possible complete tree and
guarantee the globally optimal tree.

For a continuous feature with many distinct values, evaluating many
possible thresholds can be computationally expensive.

## 5.3 Approximate split finding

Approximate split finding reduces the number of candidate thresholds.

Instead of evaluating every possible split position, it proposes a
smaller set of candidate boundaries, often based on quantiles or a
weighted quantile sketch.

The candidate set is designed to retain useful split positions while
reducing computation and memory requirements.

In XGBoost's approximate formulation, Hessian information can be used as
weights when constructing the quantile sketch. This matters because
observations with different Hessians can have different influence on the
local objective.

Approximate split finding is not simply random threshold selection.

## 5.4 Histogram-based split finding

Histogram-based methods first map continuous feature values into a
limited number of bins.

For each feature, the algorithm aggregates gradient and Hessian
statistics for observations in each bin:

`Bin G = sum of gradients of observations in the bin`

`Bin H = sum of Hessians of observations in the bin`

It then evaluates candidate boundaries between bins.

### Small numerical example

Suppose a feature is Age and the observations have these derivative
statistics:

  Observation     Age    g   h
  ------------- ----- ---- ---
  A                18   +4   1
  B                20   +2   1
  C                21   +1   1
  D                22   -2   1
  E                25   -3   1
  F                28   -2   1

For illustration, group the values into three bins:

  Bin   Age range      G   H
  ----- ----------- ---- ---
  1     18--20        +6   2
  2     21--22        -1   2
  3     23--28        -5   2

The algorithm evaluates boundaries **between bins**.

Assume `lambda=1`, `gamma=0`.

Parent totals: `G_P=0`, `H_P=6`, so `S_P=0`.

### Candidate boundary after bin 1

Left: `G_L=6`, `H_L=2`.

Right: `G_R=-6`, `H_R=4`.

Scores:

`S_L = 36/(2+1)=12`

`S_R = 36/(4+1)=7.2`

Gain:

`Gain = (1/2)(12+7.2-0)=9.6`

### Candidate boundary after bin 2

Left: `G_L=5`, `H_L=4`.

Right: `G_R=-5`, `H_R=2`.

Scores:

`S_L=25/5=5`

`S_R=25/3≈8.333`

Gain:

`Gain=(1/2)(5+8.333)≈6.667`

The first boundary has higher gain, so the selected split is after
bin 1. In this illustrative grouping, the split condition is equivalent
to `Age <= 20` versus `Age > 20`.

**Binning groups values; split evaluation chooses a boundary between
groups.** A bin is not itself automatically a final tree leaf.

Actual XGBoost binning is implementation-specific and is controlled by
settings such as `max_bin` for histogram methods. Bins are not
necessarily equal-width intervals.

## 5.5 Prefix sums make histogram evaluation efficient

Once each bin has its aggregate `G` and `H`, the algorithm can scan bins
from left to right.

At each boundary:

-   Left `G` and `H` are cumulative sums up to that bin.
-   Right `G` and `H` are parent totals minus left totals.

This avoids recalculating the sums from all observations for every
boundary.

If there are `B` bins, there are at most `B-1` internal boundaries for a
feature at that node.

## 5.6 Exact, approximate, and histogram approaches compared

  -----------------------------------------------------------------------
  Approach          Candidate         Main idea         Trade-off
                    thresholds                          
  ----------------- ----------------- ----------------- -----------------
  Exact greedy      Many possible     Evaluate detailed More precise
                    positions between candidate         candidate search,
                    sorted values     positions         potentially more
                                                        expensive

  Approximate       Reduced           Evaluate a        Lower cost,
                    quantile-based    smaller           candidate
                    candidates        representative    approximation
                                      set               

  Histogram         Bin boundaries    Aggregate G/H per Efficient split
                                      bin and scan      evaluation,
                                      boundaries        threshold
                                                        resolution
                                                        depends on
                                                        binning
  -----------------------------------------------------------------------

Do not assume that one method is always best for every dataset. Dataset
size, sparsity, hardware, and version-specific implementation matter.

## 5.7 Missing values and default directions

Tree boosters can learn how to route missing feature values. For a
candidate split, missing observations can be assigned to one side or the
other; the algorithm can evaluate which default direction gives the
better objective improvement.

This is different from blindly replacing every missing value with the
mean or median.

However, missing-value handling is not a substitute for understanding
why values are missing. Data-quality problems and missingness mechanisms
still require analysis.

## 5.8 min_child_weight

`min_child_weight` is the minimum required **sum of Hessians** in a
child node, not necessarily a minimum number of rows.

Conceptually, a proposed split must produce children whose Hessian sums
meet the configured threshold.

Why use Hessian sum?

The Hessian measures local curvature and contributes to the denominator
of the optimal leaf weight and the split score. A child with very little
total Hessian may provide an unstable or overly aggressive correction.

### Regression example

For squared-error loss with `h=1` per observation:

`H = number of observations in the node`

Therefore, in this particular case, a Hessian-sum threshold behaves like
a sample-count threshold.

### Logistic example

For binary logistic loss:

`h = p(1-p)`

Suppose four observations have probabilities 0.5, 0.5, 0.9, and 0.1.

Their Hessians are:

-   0.5 × 0.5 = 0.25
-   0.5 × 0.5 = 0.25
-   0.9 × 0.1 = 0.09
-   0.1 × 0.9 = 0.09

Total Hessian is `0.68`, although there are four observations.

Thus, Hessian sum and row count are not interchangeable for logistic
loss.

Increasing `min_child_weight` makes the algorithm more conservative
about creating children with little Hessian mass. Lowering it permits
smaller-Hessian children and can allow more detailed trees.

## 5.9 max_depth

`max_depth` limits the maximum depth of a tree.

If the root is defined as depth 0:

-   Root: depth 0
-   Its children: depth 1
-   Their children: depth 2

A maximum depth of 2 allows nodes at depths 0, 1, and 2, but nodes at
depth 2 cannot be expanded further.

For a fully expanded binary tree with maximum depth `d`, the theoretical
maximum number of leaves is `2^d`. Real trees often have fewer leaves
because splits can be rejected or growth can stop early.

Increasing max_depth allows more complex feature interactions and finer
partitions, but can increase overfitting risk and computational cost.

## 5.10 grow_policy: depthwise versus lossguide

The growth policy determines which eligible leaf is expanded next.

### depthwise

The tree grows level by level. Nodes at a shallower level are expanded
before moving to deeper levels, subject to constraints.

This produces a more level-oriented growth pattern.

### lossguide

The algorithm prioritizes the leaf with the largest loss reduction /
gain among available candidates.

This can produce asymmetric trees: one branch may grow deeper while
another remains shallow.

Lossguide is commonly associated with histogram-based tree construction
and leaf-count controls such as `max_leaves`; supported combinations
depend on the XGBoost version and tree method.

## 5.11 How the controls work together

Think of the controls as answering different questions:

-   `min_child_weight`: Is the child sufficiently supported by Hessian
    mass?
-   `max_depth`: Is the tree allowed to grow this deep?
-   `gamma`: Is the objective improvement large enough to justify the
    split?
-   `grow_policy`: Which eligible leaf should be expanded next?
-   `max_leaves`: How many leaves may the tree contain, when supported
    and used?

They are not interchangeable.

A split may have positive gain but still be rejected because a child
violates `min_child_weight`, because it would exceed `max_depth`, or
because another configured constraint prevents it.

## 5.12 End-to-end tree-building sequence

1.  Start with the root containing all training observations.
2.  Calculate or access the gradient and Hessian statistics.
3.  Generate candidate feature thresholds using the selected tree
    method.
4.  For each candidate, calculate left and right `G` and `H`.
5.  Check child constraints such as `min_child_weight`.
6.  Calculate regularized gain.
7.  Select the best eligible split.
8.  Apply growth-policy and depth/leaf constraints.
9.  Repeat for eligible leaves until no useful or permitted split
    remains.
10. Calculate the final leaf weights.

The exact implementation contains optimizations, but this is the
conceptual sequence to understand.
