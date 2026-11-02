# Group K-Fold Cross-Validation

## 1. What is Group K-Fold Cross-Validation?

Group K-Fold Cross-Validation is a cross-validation technique used when multiple samples belong to the same group.

It ensures that samples belonging to the same group are not split between the training and validation sets.

The primary objective is to evaluate how well a model generalizes to **unseen groups**, while preventing group-level data leakage.

For example:
- Multiple records belonging to the same patient
- Multiple transactions belonging to the same customer
- Multiple images belonging to the same person
- Multiple readings belonging to the same device

In such cases, normal K-Fold may place samples from the same group in both training and validation sets. Group K-Fold prevents this.

---

## 2. Why Do We Need Group K-Fold?

Consider a hospital dataset containing multiple records for each patient.

| Patient ID | Age | Disease |
|---|---:|---|
| P1 | 25 | Yes |
| P1 | 26 | No |
| P1 | 27 | Yes |
| P2 | 35 | No |
| P2 | 36 | No |
| P3 | 45 | Yes |
| P3 | 46 | Yes |
| P4 | 50 | No |

Each patient has multiple records.

If normal K-Fold is used, it may split the records of the same patient across training and validation sets.

For example:

- Patient P1's first record may be in the training set.
- Patient P1's second record may be in the validation set.

This can cause data leakage because the model has already seen information related to the same patient during training.

As a result, the validation score may be artificially high and may not represent performance on genuinely unseen patients.

Group K-Fold prevents this by keeping all records belonging to a particular patient in the same fold.

---

## 3. How Group K-Fold Works

Suppose a dataset contains:

- 100 unique patients
- Multiple records per patient
- 5 folds

Group K-Fold divides the groups into 5 folds.

During each iteration:

- 4 folds are used for training.
- 1 fold is used for validation.

The process is repeated 5 times.

Each group appears in the validation set exactly once across the 5 folds.

The samples belonging to the same group remain together in the same fold.

### Important

Group K-Fold splits the dataset based on groups rather than treating every row as an independent sample.

---

## 4. Python Implementation

### Scikit-learn class

`GroupKFold`

### Parameters

- `n_splits`: Number of folds.
- `groups`: Group ID associated with each sample.

The `groups` parameter identifies which samples belong to the same group.

For example, in a patient dataset, the patient ID can be used as the group.

Group K-Fold uses this information to ensure that a patient's samples are not distributed across multiple folds.

### Important distinction

Group K-Fold generates training and validation indices.

It does not train the machine learning model itself.

The model must be trained and evaluated separately for each split.

---

## 5. Does Group K-Fold Ensure Equal Samples in Every Fold?

No. Group K-Fold does not guarantee exactly equal numbers of samples in every fold.

It attempts to distribute groups across folds, but the number of samples in each group may be different.

Consider the following groups:

| Group | Number of Samples |
|---|---:|
| P1 | 100 |
| P2 | 10 |
| P3 | 20 |
| P4 | 50 |
| P5 | 5 |

Since groups must remain intact, folds may contain different numbers of samples.

The primary objective is group separation, not equal sample counts.

---

## 6. Does Group K-Fold Preserve Class Distribution?

No. Group K-Fold does not guarantee that the class distribution will be the same across all folds.

For example, one fold may contain a higher proportion of positive samples, while another may contain more negative samples.

This can happen when groups themselves have different class distributions.

### For classification datasets

If class distribution also needs to be approximately preserved, use:

`StratifiedGroupKFold`

It attempts to:

1. Keep samples from the same group together.
2. Preserve class proportions across folds as much as possible.

However, exact class proportions are not guaranteed, especially when groups are large or the dataset is highly imbalanced.

---

## 7. Advantages of Group K-Fold

- Prevents the same group from appearing in both training and validation sets.
- Helps avoid group-level data leakage.
- Evaluates generalization to unseen groups.
- Useful when samples are correlated because they belong to the same entity.
- Provides a more realistic evaluation when the deployment scenario involves new groups.

---

## 8. Disadvantages of Group K-Fold

- Does not guarantee equal sample counts in every fold.
- Does not guarantee equal class distributions.
- Large or imbalanced groups may produce uneven folds.
- Some folds may contain very few samples from a particular class.
- The number of unique groups limits the number of possible folds.

---

## 9. When Should We Use Group K-Fold?

| Dataset | Group |
|---|---|
| Hospital records | Patient ID |
| Customer transactions | Customer ID |
| Multiple images per person | Person ID |
| Sensor readings | Device ID |
| Multiple reviews per user | User ID |
| Multiple records per company | Company ID |

Use Group K-Fold when the same entity has multiple samples and the model should be evaluated on unseen entities.

---

## 10. Group K-Fold vs K-Fold

| Feature | K-Fold | Group K-Fold |
|---|---|---|
| Splitting basis | Individual samples | Groups |
| Same group across train and validation | Possible | Prevented |
| Equal sample count per fold | Approximately | Not guaranteed |
| Class distribution preserved | No | No |
| Suitable for repeated entities | Not always | Yes |
| Main objective | General cross-validation | Evaluate on unseen groups |

---

## 11. Important Points

- Group K-Fold is useful when multiple samples belong to the same entity.
- It keeps the samples belonging to a group together.
- It prevents group-level leakage between training and validation sets.
- It does not guarantee equal samples per fold.
- It does not guarantee equal class distributions.
- `GroupKFold` is the Scikit-learn class.
- `StratifiedGroupKFold` can be used when both group separation and approximate class stratification are required.

---
# Stratified Group K-Fold Cross-Validation

## 1. What is Stratified Group K-Fold?

Stratified Group K-Fold is a cross-validation technique used when:

- Multiple samples belong to the same group.
- The dataset is a classification dataset.
- We want to keep samples from the same group together.
- We also want to preserve the class distribution across folds as much as possible.

It combines the ideas of:

- **Group K-Fold:** Keeps samples belonging to the same group together.
- **Stratified K-Fold:** Attempts to preserve class proportions across folds.

The primary objective is to evaluate model performance on unseen groups while reducing group-level data leakage and maintaining approximately similar class distributions.

---

## 2. Why Do We Need Stratified Group K-Fold?

Consider a hospital dataset containing multiple records for each patient.

| Patient ID | Record | Disease |
|---|---|---|
| P1 | R1 | Yes |
| P1 | R2 | Yes |
| P1 | R3 | No |
| P2 | R1 | No |
| P2 | R2 | No |
| P3 | R1 | Yes |
| P3 | R2 | Yes |
| P4 | R1 | No |

There are two important requirements:

### Requirement 1: Group separation

Records belonging to the same patient should not appear in both training and validation sets.

Otherwise, the model may learn patient-specific patterns during training and perform artificially well during validation.

### Requirement 2: Class distribution

The proportion of positive and negative samples should be approximately similar across folds.

For example, if the overall dataset contains 70% positive and 30% negative samples, we want each fold to have approximately the same proportion, where feasible.

Stratified Group K-Fold attempts to satisfy both requirements.

---

## 3. How Does It Work?

Suppose a dataset contains:

- 100 unique patients
- Multiple records per patient
- 5 folds

Stratified Group K-Fold attempts to divide the patient groups into 5 folds such that:

1. All records belonging to the same patient remain in one fold.
2. Class proportions are approximately preserved across folds.
3. Each fold is used as the validation set once.

During each iteration:

- 4 folds are used for training.
- 1 fold is used for validation.

Total iterations = 5.

### Important

It is not always possible to perfectly preserve class proportions because groups cannot be split.

The class distribution in each fold depends on the groups and their class labels.

---

## 4. Does Stratified Group K-Fold Ensure Equal Samples?

No. It does not guarantee exactly equal numbers of samples in every fold.

It attempts to preserve class proportions while keeping groups intact.

If groups have different sizes, the number of samples per fold may vary.

For example:

| Group | Samples |
|---|---:|
| P1 | 100 |
| P2 | 10 |
| P3 | 20 |
| P4 | 50 |
| P5 | 5 |

Since the groups cannot be split, folds may contain different numbers of samples.

---

## 5. Does It Ensure the Same Class Distribution?

It attempts to preserve class proportions as closely as possible.

However, **it does not guarantee exactly the same class distribution in every fold.**

For example, suppose the dataset contains:

- 80 positive samples
- 20 negative samples

The overall positive-to-negative ratio is 80:20.

Stratified Group K-Fold tries to keep approximately similar ratios across folds, but exact ratios may not be achievable because samples belonging to the same group must remain together.

The distribution also depends on how the classes are distributed among groups.

---

## 6. Advantages

- Prevents the same group from appearing in both training and validation sets.
- Helps avoid group-level data leakage.
- Attempts to preserve class proportions across folds.
- Evaluates model performance on unseen groups.
- Useful for classification datasets with repeated entities and imbalanced classes.

---

## 7. Disadvantages

- Does not guarantee equal sample counts in every fold.
- Does not guarantee identical class distributions.
- Large or imbalanced groups can produce uneven folds.
- Perfect stratification may not be possible when groups must remain intact.
- The number of unique groups limits the number of possible folds.

---

## 8. When Should We Use It?

| Dataset | Group |
|---|---|
| Hospital records | Patient ID |
| Customer transactions | Customer ID |
| Multiple images per person | Person ID |
| Sensor readings | Device ID |
| Multiple reviews per user | User ID |
| Multiple records per company | Company ID |

Use Stratified Group K-Fold when:

- The task is classification.
- Multiple samples belong to the same entity.
- The model must generalize to unseen entities.
- Maintaining approximate class proportions across folds is important.

---

## 9. Group K-Fold vs Stratified Group K-Fold

| Feature | Group K-Fold | Stratified Group K-Fold |
|---|---|---|
| Group separation | Yes | Yes |
| Class distribution preservation | No | Attempts to preserve |
| Equal sample count guaranteed | No | No |
| Suitable for classification | Yes | Yes |
| Suitable for imbalanced classification | May produce uneven class proportions | Often more suitable |
| Main objective | Keep groups separate | Keep groups separate and approximately preserve class proportions |

---

## 10. Important Points

- Stratified Group K-Fold combines group separation with class stratification.
- It keeps samples belonging to the same group together.
- It attempts to preserve class proportions across folds.
- It does not guarantee equal sample counts.
- It does not guarantee identical class distributions.
- It is mainly useful for classification datasets.
- `StratifiedGroupKFold` is the Scikit-learn class.



