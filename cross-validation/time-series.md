# Time Series Cross-Validation

## 1. What is Time Series Cross-Validation?

Time Series Cross-Validation is a cross-validation technique used for datasets where observations are recorded over time.

Examples:
- Daily stock prices
- Monthly sales
- Hourly temperature
- Daily website traffic
- Electricity demand
- Product demand forecasting

In time series data, the **order of observations is important**.

Unlike ordinary machine learning datasets, we cannot randomly shuffle time series observations while splitting them for training and validation.

The model must be trained on past data and evaluated on future data.

---

## 2. Why Do We Need Time Series Cross-Validation?

Consider a dataset containing monthly sales:

| Month | Sales |
|---|---:|
| January | 100 |
| February | 120 |
| March | 150 |
| April | 180 |
| May | 200 |
| June | 250 |

Suppose we want to predict sales for July.

The model should learn from January to June and predict July.

It should not use August or September data to predict July.

### Problem with Normal K-Fold

Normal K-Fold randomly divides the dataset into folds.

This may result in:
- Future observations appearing in the training set.
- Past observations appearing in the validation set.

This creates **data leakage** because the model gets access to information from the future.

For time series forecasting, this produces unrealistic evaluation results.

---

## 3. How Time Series Cross-Validation Works

Time Series Cross-Validation follows the chronological order of observations.

The model is trained on past observations and evaluated on later observations.

For example:

| Iteration | Training Data | Validation Data |
|---|---|---|
| 1 | January–March | April |
| 2 | January–April | May |
| 3 | January–May | June |

In every iteration:
- Training data comes before validation data.
- Validation data comes after training data.
- Future observations are not used to train the model for an earlier prediction.

This approach evaluates whether the model can predict future observations using historical data.

---

## 4. TimeSeriesSplit

The commonly used Scikit-learn class for Time Series Cross-Validation is:

`TimeSeriesSplit`

It generates chronological train-validation splits.

### Important Parameters

- **n_splits:** Number of splits.
- **max_train_size:** Maximum number of samples used in the training set.
- **test_size:** Number of samples in each validation set.
- **gap:** Number of samples excluded between the training and validation sets.

---

## 5. How TimeSeriesSplit Divides Data

Suppose we have 10 chronological observations and use 3 splits.

The folds may look like this:

### Split 1

Training: Observations 1, 2, 3, 4

Validation: Observations 5, 6

### Split 2

Training: Observations 1, 2, 3, 4, 5, 6

Validation: Observations 7, 8

### Split 3

Training: Observations 1, 2, 3, 4, 5, 6, 7, 8

Validation: Observations 9, 10

The training window expands after each iteration.

This is called **Expanding Window Cross-Validation**.

---

## 6. Expanding Window vs Sliding Window

### Expanding Window

The training set grows with every iteration.

Example:

| Split | Training Data | Validation |
|---|---|---|
| 1 | Jan–Mar | Apr |
| 2 | Jan–Apr | May |
| 3 | Jan–May | Jun |

Useful when older observations remain relevant to future predictions.

### Sliding Window

The training set has a fixed size and moves forward over time.

Example:

| Split | Training Data | Validation |
|---|---|---|
| 1 | Jan–Mar | Apr |
| 2 | Feb–Apr | May |
| 3 | Mar–May | Jun |

Useful when older observations may no longer represent current patterns.

TimeSeriesSplit supports limiting the training window using `max_train_size`.

---

## 7. Does TimeSeriesSplit Ensure Equal Samples?

By default, the validation folds have approximately equal sizes.

The training folds grow over time.

For example:

| Split | Training Samples | Validation Samples |
|---|---:|---:|
| 1 | 4 | 2 |
| 2 | 6 | 2 |
| 3 | 8 | 2 |

The exact sizes depend on the number of observations and the parameters.

---

## 8. Does TimeSeriesSplit Preserve Class Distribution?

No.

TimeSeriesSplit does not preserve class proportions.

It splits chronologically.

If a time series classification dataset contains different class distributions over time, the folds may have different class proportions.

This is expected because preserving chronological order is more important than stratification for ordinary time series forecasting.

---

## 9. What Is the Gap Parameter?

The `gap` parameter specifies the number of observations excluded between the end of the training set and the beginning of the validation set.

Example:

| Training | Gap | Validation |
|---|---|---|
| Jan–Mar | April | May |

The gap helps prevent leakage in situations where nearby observations may contain overlapping information.

For example, if features are constructed using a rolling window, training observations close to the validation period may share information with validation observations.

A gap can reduce this risk.

However, **gap alone does not prevent every form of data leakage**. Features must still be constructed using only information available at the prediction time.

---

## 10. Advantages

- Preserves chronological order.
- Prevents future observations from being used to train models evaluated on earlier observations.
- Evaluates performance on future observations.
- Useful for forecasting and time-dependent prediction.
- Supports expanding training windows.
- Can use a gap between training and validation.

---

## 11. Disadvantages

- Training and validation folds may have different sizes.
- The training set may be small in early splits.
- Scores can vary due to changes in trends or seasonality.
- Does not automatically handle multiple independent time series.
- Does not automatically prevent every form of time-dependent data leakage.

---

## 12. When Should We Use It?

| Dataset | CV Method |
|---|---|
| Stock prices | TimeSeriesSplit |
| Monthly sales forecasting | TimeSeriesSplit |
| Daily demand forecasting | TimeSeriesSplit |
| Hourly electricity demand | TimeSeriesSplit |
| Website traffic forecasting | TimeSeriesSplit |
| Weather forecasting | TimeSeriesSplit |

Use Time Series Cross-Validation whenever the model is trained on historical observations and must predict future observations.

---

## 13. Important Points

- Time Series Cross-Validation preserves chronological order.
- Training data must come before validation data.
- Normal K-Fold can cause future information leakage.
- `TimeSeriesSplit` is the Scikit-learn class.
- It does not shuffle observations.
- It does not preserve class proportions.
- It can use a gap between training and validation.
- The dataset must be sorted chronologically before splitting.
- It is primarily used for time-dependent prediction and forecasting.

