# Categorical Encoding — Quick Revision Notes

## 1. Label Encoding

**Definition:** Converts each unique category into an integer label, such as Cat → 0, Dog → 1.

**Main use:** Classification target variable (`y`), such as Spam/Not Spam or Cat/Dog.

**Why not use it blindly for nominal input features?**

Suppose Color is encoded as Blue = 0, Green = 1, Red = 2. These colors have no natural order, but a model may treat the numbers as meaningful numerical values.

**Limitations:**
- May introduce artificial ordering for nominal input features.
- Unseen target classes can cause errors during transformation.
- Integer labels are identifiers, not meaningful rankings.

## 2. One-Hot Encoding (OHE)

**Definition:** Creates a separate binary column for each category.

Example: Color = Blue, Green, Red.

| Color | Blue | Green | Red |
|---|---:|---:|---:|
| Blue | 1 | 0 | 0 |
| Green | 0 | 1 | 0 |
| Red | 0 | 0 | 1 |

**Main use:** Nominal input features with no natural order, such as color, city, or brand.

**Why OHE?**

In linear regression, integer encoding can imply equal predicted changes between consecutive category values. OHE represents each category separately, allowing each category to have its own effect without imposing an artificial numerical order.

**Limitations:**
- High-cardinality features can create many columns.
- Can produce sparse data.
- With an intercept and all dummy columns, perfect multicollinearity can occur. Dropping a reference category can address this.
- Unseen categories require a handling strategy; `handle_unknown="ignore"` is one option.

## 3. Ordinal Encoding

**Definition:** Converts categories into integers according to their meaningful order.

Example:

Poor = 0, Average = 1, Good = 2, Excellent = 3.

**Main use:** Ordinal input features such as satisfaction levels, education levels, and product sizes.

**Important issue — Equal-spacing assumption**

For linear regression:

Price = β₀ + β₁ × Encoded_Value

If β₁ = 50, each one-level increase changes the prediction by 50. The price does not double. Instead, the model assumes the same predicted change between consecutive levels.

However, real-world effects between category levels may differ.

**When to choose OHE instead?**

Even if categories are ordinal, OHE can be useful when each category may have a different effect and we do not want to impose equal-step effects. However, OHE does not directly represent the natural order.

**Limitations:**
- Incorrect ordering misrepresents the feature.
- Numerical distances may be interpreted by the model.
- Missing and unseen categories need a handling strategy.

## 4. Quick Comparison

| Point | Label Encoding | One-Hot Encoding | Ordinal Encoding |
|---|---|---|---|
| Main purpose | Target class IDs | Nominal input features | Ordered input features |
| Typical use | Target `y` | Input `X` | Input `X` |
| Output | Integer labels | Binary columns | Integer values |
| Represents meaningful order? | No | No | Yes |
| Main issue | Artificial order if misused on nominal features | High dimensionality | Numerical spacing assumptions |

## Frequency Encoding

**Definition:** Frequency Encoding replaces each category with its relative frequency in the dataset.

### When to Use
- **High Cardinality:** When a nominal categorical feature has many unique categories and One-Hot Encoding would create too many columns.
- **Frequency as a Useful Signal:** When a category's popularity or occurrence frequency may help the model make predictions.
- **Dimensionality Reduction:** When we want to represent a high-cardinality feature using a single numerical column instead of creating multiple columns.

### Drawbacks
- **Information Loss:** Different categories can have the same frequency, making them indistinguishable after encoding.
- **No Target Relationship:** Frequency represents how common a category is, not how that category affects the target.
- **Unseen Categories:** Categories appearing in test or production data but not in training data require a fallback strategy.
- **Data Leakage:** Calculate frequencies using training data only, then apply the same mapping to validation and test data.

**Key Point:** Frequency Encoding is useful for high-cardinality nominal features when category frequency provides a potentially useful signal. Always compare its performance with alternative encoding methods.


## Count Encoding
- Here we replace the category by the number of times it has occured . The drawbacks are same as that of Frequency encoding 


# Target Mean Encoding

## 1. Why Do We Use Target Mean Encoding?

Target Mean Encoding replaces each category with the **mean of the target variable for that category**.

- For a classification problem with a binary target (0/1), it represents the historical positive-target rate for each category.
- For a regression problem, it represents the average target value for each category.

### When to Use

- **High Cardinality:** When a nominal categorical feature has many unique categories and One-Hot Encoding would create too many columns.
- **Strong Target Relationship:** When categories have different historical target outcomes that may provide useful predictive information.
- **Dimensionality Reduction:** When we want to represent a categorical feature using one numerical column instead of creating many columns through One-Hot Encoding.
- **Features with Many Categories:** Examples include city, product ID, merchant ID, and customer segment, provided their historical target patterns are useful.

### Advantages

- Reduces dimensionality compared with One-Hot Encoding for high-cardinality features.
- Captures the historical relationship between categories and the target.
- Can work well with high-cardinality categorical features when implemented correctly.
- Produces a numerical representation that many ML models can use.

**Important:** Target Mean Encoding does not guarantee better performance. Compare it with alternative encodings using proper validation.

---

## 2. Drawbacks and Their Solutions

### 1. Data Leakage Within Training Data

**Problem:** A training row's own target can contribute to the encoded value assigned to that row. This can make the training representation overly dependent on the target.

**Example:**

Suppose we want to predict whether a customer purchases a product.

| City | Purchased |
|---|---:|
| Mumbai | 1 |
| Mumbai | 0 |
| Mumbai | 1 |

Mumbai's target mean = (1 + 0 + 1) / 3 = 0.667

All three Mumbai rows receive an encoded value of 0.667. Each row's own target contributes to this mean.

This is particularly problematic for rare categories. If a category occurs once and its target is 1, its mean is 1.0, directly reflecting that row's target.

**Solution: Out-of-Fold (OOF) Target Encoding**

1. Divide the training data into multiple folds.
2. For each fold, calculate category means using only the other folds.
3. Encode the held-out fold using those means.
4. Repeat until every training row has an OOF encoded value.
5. For test data, use a mapping learned from the full training data.

**Example:**

Suppose Mumbai appears in four training rows with targets 1, 0, 1, 1.

- If a row belongs to Fold 1, calculate Mumbai's mean using Mumbai rows outside Fold 1.
- If another row belongs to Fold 2, calculate the mean using Mumbai rows outside Fold 2.
- Therefore, different training rows from the same city can receive different encoded values.

The row's own target is excluded from the data used to calculate its OOF encoding.

### 2. Rare Categories

**Problem:** Categories with very few observations can have unreliable target means.

**Example:**

| Brand | Number of observations | Target Mean |
|---|---:|---:|
| Nike | 100 | 0.70 |
| Adidas | 80 | 0.65 |
| NewBrand | 1 | 1.00 |

NewBrand has only one observation, and its target is 1. Its mean becomes 1.00, but one observation is insufficient to establish a reliable historical rate.

**Solution: Smoothing**

Combine the category mean with the overall training target mean.

Formula:

Smoothed Mean = (n × category_mean + α × global_mean) / (n + α)

Where:
- n = number of observations for the category
- category_mean = mean target value for that category
- global_mean = overall target mean in the training data
- α = smoothing strength

**Example:**

Suppose:
- NewBrand category mean = 1.00
- n = 1
- Global training target mean = 0.60
- α = 10

Smoothed Mean = (1 × 1.00 + 10 × 0.60) / (1 + 10) = 7/11 ≈ 0.636

Instead of assigning 1.00, smoothing moves the estimate toward the global mean of 0.60.

**Benefit:** Rare categories receive less extreme and generally more stable encoded values.

### 3. Overfitting

**Problem:** When a feature has many unique categories, the model may learn noisy historical target means instead of patterns that generalize to unseen data.

**Example:**

Suppose a dataset contains 1,000 product IDs.

- Product P101 has 100 observations and a target mean of 0.70.
- Product P999 has one observation and a target mean of 1.00.

The model might treat P999 as a highly successful product even though its mean is based on only one observation.

**Solutions:**

- Use OOF Target Encoding for training features.
- Apply smoothing to reduce the influence of unreliable category means.
- Evaluate performance on a separate validation set.
- Consider grouping sufficiently rare categories when appropriate.

**Benefit:** These methods reduce the risk of fitting noisy category-level patterns.

### 4. Unseen Categories

**Problem:** A category may appear in test or production data but not in the training data. Its target mean is therefore unavailable.

**Example:**

Training mapping:

| City | Target Mean |
|---|---:|
| Mumbai | 0.70 |
| Pune | 0.60 |
| Delhi | 0.40 |

A new test row contains Jaipur, which was absent from training.

There is no learned target mean for Jaipur.

**Solution: Use a Fallback Value**

A common strategy is to use the overall training target mean.

Suppose the global training target mean is 0.58.

Jaipur's fallback encoded value = 0.58.

This provides a reasonable baseline without using the test target.

**Important:** The fallback is not Jaipur's actual purchase rate. It is an estimate based on the training data.

### 5. Train-Test Leakage

**Problem:** If target means are calculated using both training and test data, information from the test set can influence the encoding.

**Example:**

| Dataset | City | Purchased |
|---|---|---:|
| Training | Mumbai | 1 |
| Training | Mumbai | 0 |
| Test | Mumbai | 1 |

If all three rows are used, Mumbai's mean becomes (1 + 0 + 1) / 3 = 0.667.

Using only training rows, Mumbai's mean is (1 + 0) / 2 = 0.50.

The test target has influenced the mapping, making the evaluation unreliable.

**Solution:**

1. Split the data into training and test sets first.
2. Learn target-encoding mappings only from training data.
3. Use OOF encoding to create leakage-safe training features.
4. Apply the mapping learned from the full training data to test data.
5. During cross-validation, learn the mapping separately within each training fold.

**Benefit:** Test targets do not influence the learned encoding.

---
