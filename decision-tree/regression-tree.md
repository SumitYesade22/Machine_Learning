# 5. Decision Tree Regression

## 5.1 Introduction

Decision Tree Regression is a supervised, non-parametric machine learning algorithm used to predict continuous numerical values.

It works by recursively splitting the dataset into smaller regions based on feature conditions. Each final region is represented by a leaf node that predicts a numerical value.

Unlike Decision Tree Classification, which predicts categorical classes, Regression Trees predict continuous values.

### Example

Suppose we want to predict house prices based on house size.

| House Size (sq. ft.) | Price (₹ Lakhs) |
|---:|---:|
| 500 | 30 |
| 700 | 40 |
| 900 | 50 |
| 1200 | 65 |
| 1500 | 80 |
| 1800 | 95 |

A Regression Tree may learn rules such as:

- If size ≤ 800, predict ₹35 Lakhs.
- If 800 < size ≤ 1350, predict ₹57.5 Lakhs.
- If size > 1350, predict ₹87.5 Lakhs.

These predictions are based on the mean target value of the training samples in each leaf.

---

## 5.2 How Does a Regression Tree Work?

The working process is similar to a Classification Tree, but the splitting criterion is different.

1. Start with the complete dataset at the root node.
2. Generate candidate splits using available features.
3. Divide the dataset into left and right child nodes.
4. Calculate the prediction for each child node.
5. Calculate the error produced by each candidate split.
6. Select the split that minimizes the weighted prediction error.
7. Repeat the process recursively for each child node.

The objective is to create regions where target values are as close as possible to their predicted value.

---

## 5.3 Prediction in Regression Trees

In a standard Decision Tree Regression model using squared-error loss, the prediction at each leaf node is the **mean of the target values** of the training samples belonging to that leaf.

Formula:

`Leaf Prediction = Sum of Target Values / Number of Samples`

### Example

Suppose a leaf contains the following house prices:

`[30, 40, 50]`

Prediction:

`Mean = (30 + 40 + 50) / 3 = 40`

Therefore, the leaf predicts ₹40 Lakhs for any new sample reaching that leaf.

**Important:** Every sample reaching the same leaf receives the same prediction.

---

## 5.4 Mean Squared Error (MSE)

MSE measures the average squared difference between actual target values and their predicted values.

Formula:

`MSE = (1/n) × Σ(y_i - ŷ_i)²`

Where:

- `n` = Number of samples
- `y_i` = Actual target value
- `ŷ_i` = Predicted target value

### Example

Actual values:

`[30, 40, 50]`

Mean prediction:

`ŷ = 40`

| Actual (y) | Predicted (ŷ) | Squared Error |
|---:|---:|---:|
| 30 | 40 | 100 |
| 40 | 40 | 0 |
| 50 | 40 | 100 |

`MSE = (100 + 0 + 100) / 3`

`MSE = 66.67`

Lower MSE indicates that predictions are closer to actual values.

---

## 5.5 How Does a Regression Tree Select the Best Split?

A Regression Tree evaluates candidate splits and selects the split that produces the **minimum weighted MSE**.

### Step 1: Generate Candidate Splits

For numerical features, values are sorted and candidate thresholds are generated between consecutive values.

Example:

`Age = [20, 22, 25, 28, 35]`

Candidate thresholds:

`[21, 23.5, 26.5, 31.5]`

### Step 2: Divide the Dataset

For each candidate threshold, divide the dataset into two child nodes.

Example:

`Age ≤ 23.5` → Left child

`Age > 23.5` → Right child

### Step 3: Calculate Child Node Predictions

Calculate the mean target value separately for the left and right child nodes.

### Step 4: Calculate Weighted MSE

`Weighted MSE = (N_left / N) × MSE_left + (N_right / N) × MSE_right`

Where:

- `N` = Total samples in the parent node
- `N_left` = Number of samples in the left child
- `N_right` = Number of samples in the right child

### Step 5: Select the Best Split

Evaluate all candidate splits and select the one with the **lowest weighted MSE**.

This process continues recursively.

---

## 5.6 Numerical Example: Best Split Selection

Consider the following dataset:

| X | Y |
|---:|---:|
| 1 | 10 |
| 2 | 12 |
| 3 | 14 |
| 4 | 30 |
| 5 | 32 |
| 6 | 34 |

Suppose we evaluate the split:

`X ≤ 3.5`

**Left Child:**

Target values = `[10, 12, 14]`

Mean = `12`

MSE:

`((10-12)² + (12-12)² + (14-12)²) / 3`

`= (4 + 0 + 4) / 3`

`= 2.67`

**Right Child:**

Target values = `[30, 32, 34]`

Mean = `32`

MSE:

`((30-32)² + (32-32)² + (34-32)²) / 3`

`= (4 + 0 + 4) / 3`

`= 2.67`

**Weighted MSE:**

`= (3/6 × 2.67) + (3/6 × 2.67)`

`= 2.67`

The parent node has:

Mean = `22`

Parent MSE:

`= ((10-22)² + (12-22)² + (14-22)² + (30-22)² + (32-22)² + (34-22)²) / 6`

`= (144 + 100 + 64 + 64 + 100 + 144) / 6`

`= 102.67`

Therefore:

`MSE Reduction = Parent MSE - Weighted Child MSE`

`= 102.67 - 2.67`

`= 100`

This split reduces the prediction error significantly.

---

## 5.7 Impurity Reduction in Regression Trees

In regression, the objective is to reduce the variation of target values within each child node.

The split quality can be measured using:

- Mean Squared Error (MSE)
- Variance reduction
- Mean Absolute Error (MAE)

For squared-error splitting:

`MSE Reduction = Parent MSE - Weighted Child MSE`

The algorithm selects the split with the maximum MSE reduction, which is equivalent to selecting the split with minimum weighted child MSE for a fixed parent node.

---

## 5.8 Regression Tree vs Classification Tree

| Aspect | Classification Tree | Regression Tree |
|---|---|---|
| Target variable | Categorical | Continuous |
| Prediction | Class label | Numerical value |
| Leaf prediction | Majority class | Mean target value (squared-error criterion) |
| Common impurity criteria | Gini, Entropy | MSE, variance |
| Split objective | Reduce class impurity | Reduce target variation |
| Example | Spam or Not Spam | House Price Prediction |

---

## 5.9 Characteristics of Regression Trees

- Non-parametric algorithm.
- Captures non-linear relationships.
- Does not require feature scaling.
- Handles numerical features and can represent complex decision boundaries.
- Produces piecewise constant predictions.
- Each leaf predicts a constant numerical value.
- Can overfit if the tree grows too deep.
- Sensitive to small changes in training data.

### Important Limitation: Piecewise Constant Predictions

Regression Trees divide the feature space into separate regions and assign a constant prediction to each region.

Therefore, predictions do not change smoothly with the input feature. They change when a sample crosses a decision threshold.

---

### Important Parameters

| Parameter | Description |
|---|---|
| `criterion` | Function used to measure split quality |
| `max_depth` | Maximum depth of the tree |
| `min_samples_split` | Minimum samples required to split an internal node |
| `min_samples_leaf` | Minimum samples required in a leaf node |
| `max_leaf_nodes` | Maximum number of leaf nodes |
| `random_state` | Controls randomness for reproducibility |

**Note:** Scikit-learn supports `squared_error`, `friedman_mse`, `absolute_error`, and `poisson` as regression criteria.

---

## 5.10 Key Takeaways

1. Decision Tree Regression predicts continuous numerical values.
2. Each leaf predicts the mean target value when using squared-error loss.
3. MSE measures the average squared prediction error.
4. The best split minimizes weighted child MSE.
5. Regression Trees produce piecewise constant predictions.
6. Deep trees can overfit the training data.

## Advantages of Decision Trees

1. **Easy to Understand and Interpret:** Decision Trees follow rule-based decisions, making their predictions easy to understand and explain.

2. **Non-Parametric:** They do not assume a predefined functional relationship between input features and the target variable.

3. **Captures Non-Linear Relationships:** Can model complex, non-linear relationships between features and the target variable.

4. **No Feature Scaling Required:** Standardization and normalization are generally not required.

5. **Handles Different Feature Types:** Can work with numerical and categorical features. However, categorical feature encoding may be required depending on the implementation.

6. **Automatic Feature Selection:** Selects features during the splitting process based on impurity reduction.

7. **Captures Feature Interactions:** Can learn interactions between features through successive splits.

8. **Minimal Data Preprocessing:** Generally requires less preprocessing compared to many other ML algorithms.

## Disadvantages of Decision Trees

1. **Prone to Overfitting:** Deep trees can learn noise and specific patterns in training data, resulting in poor generalization on unseen data.

2. **High Variance:** Small changes in training data can result in significantly different tree structures and predictions.

3. **Greedy Algorithm:** Selects the locally best split at each node rather than searching for the globally optimal tree.

4. **Unstable:** Decision Trees are sensitive to variations in training data.

5. **Bias Toward Features with Many Split Points:** Certain splitting criteria may favor features having more possible splitting points or category partitions.

6. **Poor Extrapolation:** Regression Trees cannot reliably predict trends beyond the range or patterns learned from training data.

7. **Piecewise Constant Predictions:** Regression Trees produce constant predictions within each leaf, which can make it difficult to represent smooth relationships and may cause sudden prediction changes at split boundaries.

8. **Complexity in Large Trees:** Deep trees with many nodes can become difficult to interpret.

9. **Limited Predictive Performance:** A single Decision Tree may have lower predictive performance than ensemble methods such as Random Forest and Gradient Boosting.

### Key Takeaway

- **Major disadvantages:** Overfitting, high variance, and instability.
- **Common solution:** Use pruning or ensemble methods such as Random Forest and Gradient Boosting to improve generalization.