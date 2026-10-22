# Types of Machine Learning

Machine Learning systems can be classified into three different categories based on how they learn, how they receive data, and how they generalize to unseen examples.

## 1. Classification Based on Supervision

Based on the availability of labelled training data, Machine Learning is classified into four types:

1. Supervised Learning
2. Unsupervised Learning
3. Semi-Supervised Learning
4. Reinforcement Learning

## 1.1 Supervised Learning

### Definition

**Supervised Learning** is a type of Machine Learning in which the model is trained using labelled data. Each training example contains input features and the corresponding correct answer, known as the target.

The model learns the relationship between the input features and the target so that it can predict the target for new, unseen examples.

### Important Terms

- **Features (`X`):** Input variables used by the model to make predictions.
- **Target (`y`):** The output or correct answer that the model needs to predict.
- **Training:** The process of learning patterns from the training data.

### Example: House Price Prediction

Suppose we have the following dataset:

| House Size | Bedrooms | House Price |
|---:|---:|---:|
| 800 sq. ft. | 1 | ₹50 lakh |
| 1,200 sq. ft. | 2 | ₹75 lakh |
| 1,600 sq. ft. | 3 | ₹1 crore |

Here:

- Features (`X`) = House size and number of bedrooms.
- Target (`y`) = House price.

The model learns from houses whose prices are already known. After training, it can estimate the price of a house that was not present in the training dataset.

### Types of Supervised Learning

#### A. Regression

**Regression** is used when the target variable is numerical.

Examples:
- Predicting house prices.
- Predicting monthly sales.
- Estimating delivery time.
- Predicting product prices.

Common algorithms:
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor

#### B. Classification

**Classification** is used when the target variable represents a category or class.

Examples:
- Spam vs. Not Spam
- Fraudulent vs. Legitimate Transaction
- Defective vs. Non-defective Product
- Customer Churn vs. No Churn

Common algorithms:
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Decision Tree Classifier
- Random Forest Classifier

### Advantages

- The model learns using known target values.
- Performance can be evaluated by comparing predictions with actual answers.
- It can solve regression and classification problems.

### Limitations

- Labelled data can be expensive and time-consuming to collect.
- Poor-quality labels can negatively affect model performance.
- A model may overfit the training data and perform poorly on unseen examples.

**Key Point:** Supervised Learning uses labelled examples, but the objective is to generalize to new data rather than simply memorize the training examples.

---

## 1.2 Unsupervised Learning

### Definition

**Unsupervised Learning** is a type of Machine Learning in which the training data does not contain supplied target labels for the learning task.

The algorithm tries to discover useful patterns, structures, groups, or relationships in the data without being given the correct answer for each example.

### Example: Customer Segmentation

Suppose an e-commerce company has information about thousands of customers, including their spending, purchase frequency, and product preferences.

However, the company has not assigned predefined customer categories.

An unsupervised algorithm can group customers with similar purchasing behaviour. The company can then analyze these groups to understand its customers better.

### Types of Unsupervised Learning

#### A. Clustering

**Clustering** groups similar data points together based on their characteristics.

Examples:
- Grouping customers by purchasing behaviour.
- Grouping similar products.
- Identifying groups of users with similar activity patterns.

Common algorithms:
- K-Means
- DBSCAN
- Hierarchical Clustering

The algorithm discovers the groups from the data. These groups do not automatically represent predefined categories such as good or bad customers.

#### B. Anomaly and Novelty Detection

These techniques identify examples that differ significantly from expected or normal patterns.

Examples:
- Detecting unusual financial transactions.
- Identifying abnormal sensor readings.
- Finding unusual network activity.

Common algorithms:
- Isolation Forest
- One-Class SVM

The precise setup depends on the algorithm and whether the task is to identify outliers within a dataset or detect new examples that differ from learned normal behaviour.

#### C. Dimensionality Reduction

**Dimensionality Reduction** reduces the number of features or dimensions used to represent data while attempting to preserve useful information or structure.

For example, a dataset containing 100 features may be transformed into a smaller set of components that is easier to visualize or process.

Common techniques:
- Principal Component Analysis (PCA)
- Kernel PCA
- Locally Linear Embedding (LLE)
- t-SNE

**Important:** PCA does not directly predict a target such as house price. It learns directions of variance in the input features and transforms the data into a new representation.

#### D. Association Rule Learning

Association Rule Learning discovers relationships between items or events that frequently occur together.

Example:

A supermarket may discover that customers who purchase one product frequently purchase another product as well.

Common algorithms:
- Apriori
- Eclat

### Important Doubt: Can Unsupervised Learning Make Predictions on New Data?

**Yes, some unsupervised algorithms can process new examples after training.** Unsupervised learning means that target labels are not supplied during training; it does not mean that the model cannot produce outputs.

Examples:

- **K-Means:** After fitting, it can assign a new data point to its nearest learned cluster using `predict()`.
- **PCA:** After fitting, it can transform new data using `transform()`.
- **DBSCAN:** The scikit-learn implementation does not provide a general built-in `predict()` method for assigning arbitrary future examples to clusters in the same way as K-Means.

Therefore, the ability to process new examples depends on the particular algorithm and its implementation.

### Clustering vs. PCA

- **Clustering:** Groups similar data points together.
- **PCA:** Transforms features into a lower-dimensional representation.
- Both can be used without target labels.
- PCA components are not predictions of a target variable.

### Advantages

- Does not require labelled target values.
- Can reveal hidden patterns and structures.
- Can help with data exploration and feature representation.

### Limitations

- Discovered patterns may be difficult to interpret.
- There may be no single correct answer against which to evaluate the results.
- The usefulness of the discovered groups or representations depends on the problem.

**Key Point:** Unsupervised Learning discovers patterns or structure without supplied target labels.

---

## 1.3 Semi-Supervised Learning

### Definition

**Semi-Supervised Learning** uses a combination of a relatively small amount of labelled data and a larger amount of unlabelled data.

It is useful when collecting data is easy but manually labelling every example is expensive or time-consuming.

### Example: Product Image Classification

Suppose an e-commerce company has 10,000 product images, but only 500 images have been labelled as headphones, earbuds, or speakers.

A semi-supervised approach can use the labelled images along with the unlabelled images to help train a classification model.

### Common Approaches

#### A. Pseudo-Labelling

A model predicts labels for unlabelled examples. Predictions with sufficient confidence may be treated as provisional labels and used in further training.

These predicted labels can be incorrect, so pseudo-labelling needs appropriate confidence thresholds and validation.

#### B. Label Propagation

Labels are propagated between related examples using relationships or similarities within the dataset, under suitable assumptions.

#### C. Unsupervised Pretraining Followed by Supervised Fine-Tuning

A model first learns useful representations from unlabelled data. It is then trained or fine-tuned using labelled examples.

#### D. Clustering-Assisted Approaches

Clustering may help identify structure among examples and support the use of labelled data. However, clustering is not required by every semi-supervised method.

### Important Doubt: Is Semi-Supervised Learning Only Clustering?

**No.** Semi-Supervised Learning is a family of learning approaches, not a single algorithm and not simply another name for clustering.

It can be used to train classification models and, in suitable settings, regression models. Clustering may be part of an approach, but it is not necessary in every case.

### Advantages

- Can reduce the amount of manual labelling required.
- Makes use of available unlabelled data.
- Can be useful when labelled examples are limited.

### Limitations

- Incorrect pseudo-labels can negatively affect training.
- Unlabelled data does not automatically improve performance.
- The method's assumptions about the relationship between labelled and unlabelled examples must be appropriate.

**Key Point:** Semi-Supervised Learning combines labelled and unlabelled data.

---

## 1.4 Reinforcement Learning

### Definition

**Reinforcement Learning (RL)** is a type of Machine Learning in which an agent learns how to make decisions by interacting with an environment.

The agent takes actions and receives rewards or penalties. Through experience, it learns a strategy that aims to maximize cumulative, long-term reward.

### Important Terms

- **Agent:** The learner or decision-maker.
- **Environment:** The system or world with which the agent interacts.
- **State:** Information describing the current situation.
- **Action:** A decision made by the agent.
- **Reward:** Feedback received after an action.
- **Policy:** The strategy used by the agent to select actions.

### Example: Robot Learning to Walk

A robot tries different movements while interacting with its environment.

- It may receive a positive reward for moving forward while maintaining balance.
- It may receive a negative reward for falling.
- Over time, it learns a strategy that helps it move more effectively.

Other examples include game-playing agents, robotics, and certain resource-allocation problems.

### Reinforcement Learning vs. Supervised Learning

| Supervised Learning | Reinforcement Learning |
|---|---|
| Learns from labelled training examples. | Learns through interaction and reward feedback. |
| Training examples contain desired target answers. | A reward does not necessarily identify the correct action for every state. |
| Often minimizes a prediction loss. | Often aims to maximize cumulative reward. |

### Advantages

- Can learn strategies through interaction.
- Useful for sequential decision-making problems.
- Can optimize long-term outcomes rather than only immediate predictions.

### Limitations

- Learning may require many interactions.
- Designing a suitable reward function can be difficult.
- Poorly designed rewards may encourage undesirable behaviour.

**Key Point:** Reinforcement Learning learns decision-making strategies through actions and rewards.

---

# 2. Classification Based on How the Model Learns from Incoming Data

There are two common approaches:

1. Batch Learning
2. Online Learning

## 2.1 Batch Learning

### Definition

**Batch Learning** is an approach in which the model is trained using a prepared collection of training data. When new data arrives, a common strategy is to retrain the model using an updated dataset.

### Example

An e-commerce company trains a product-price prediction model every month.

1. It collects historical product data.
2. It prepares the training dataset.
3. It trains and evaluates the model.
4. It deploys the model for predictions.
5. During the next training cycle, it includes newer data and trains an updated model.

### Advantages

- Training can be performed as a controlled job.
- Historical data can be included in a training run.
- Training and evaluation can be recorded for reproducibility.

### Limitations

- Retraining may require significant time and computing resources.
- The model may become outdated if the underlying patterns change and retraining is infrequent.

### Important Doubt: Does Batch Learning Mean the Model Must Go Offline?

**No.** Batch Learning describes the model-training and update strategy, not the availability of the prediction service.

A new model can be trained separately while the old model continues making predictions. Once the new model has been evaluated and approved, the system can switch to it.

Therefore, retraining does not necessarily cause downtime.

The term *offline learning* is often associated with training outside the live prediction process, but actual service availability depends on the deployment architecture.

---

## 2.2 Online Learning

### Definition

**Online Learning** is an approach in which the model is updated incrementally as new training examples or small batches of examples become available.

Instead of depending entirely on a complete retraining run for every update, the model can learn progressively.

### Example

Suppose a fraud-detection system needs to adapt to changing transaction patterns.

As newly labelled transactions become available, the model can update incrementally, provided the algorithm supports this type of learning.

### Advantages

- Can adapt more frequently to changing patterns.
- May avoid retraining from scratch on the entire dataset for every update.
- Can be useful when data arrives continuously or resources are limited.

### Limitations

- Poor-quality incoming data may negatively affect the model.
- Some approaches can suffer from catastrophic forgetting, where performance on older patterns deteriorates.
- Updates require appropriate monitoring and validation.

### Important Doubt: Does Online Learning Use Only New Data?

Not necessarily.

Online learning commonly incorporates incoming examples incrementally, but a system may also replay older examples or combine old and new data.

The essential idea is **incremental updating**, not a strict rule that old data can never be reused.

### Important Doubt: Does Online Learning Require an Internet Connection?

No.

The term *online* refers to the model's incremental learning strategy. It does not necessarily mean that the model requires the internet or must learn in real time.

An online-learning system can receive data in a stream or through periodic small batches.

### Example Using Scikit-Learn

Some scikit-learn estimators support incremental learning through the `partial_fit()` method.

For example:

```python
from sklearn.linear_model import SGDRegressor

model = SGDRegressor()

# Update the model using the first batch.
model.partial_fit(X_batch_1, y_batch_1)

# Update the same model using another batch.
model.partial_fit(X_batch_2, y_batch_2)
```

Here, `X_batch_1`, `y_batch_1`, `X_batch_2`, and `y_batch_2` represent prepared feature batches and their corresponding targets.

This is an illustrative example; the data must satisfy the estimator's requirements.

### Important Doubt: Online Learning vs. Mini-Batch Training

These terms describe different concepts.

- **Online Learning:** Describes the overall strategy used to update the model incrementally as data becomes available.
- **Mini-Batch Training:** Describes processing a subset of training examples at a time during optimization.

A batch-learning model may also use mini-batches internally. It can still be considered batch learning if the overall strategy is to train on a prepared dataset in a training run and retrain when required.

### Batch Learning vs. Online Learning

| Aspect | Batch Learning | Online Learning |
|---|---|---|
| Update strategy | Training runs on a prepared dataset | Incremental updates as examples or small batches arrive |
| New data | Often included in a later retraining run | Can be incorporated through incremental updates |
| Adaptation | Depends on retraining frequency | Can adapt more frequently |
| Computing pattern | May require substantial resources during a training run | Updates may be smaller; total cost depends on the setup |
| Requires internet? | No | No |
| Prediction service must go offline? | No | No |

**Key Point:** Batch vs. Online Learning describes how the model is updated, not whether its prediction service is available.

---

# 3. Classification Based on How the Model Generalizes

There are two common approaches:

1. Instance-Based Learning
2. Model-Based Learning

## 3.1 Instance-Based Learning

### Definition

**Instance-Based Learning** is an approach in which the system retains training examples, or a useful representation of them, and makes predictions for new examples by comparing them with similar stored examples.

Instead of relying only on a general formula, the system uses similarity between a new example and examples it has already seen.

### Example: K-Nearest Neighbors (KNN)

Suppose we have the following headphone data:

| Playback Time | Price |
|---:|---:|
| 30 hours | ₹1,500 |
| 40 hours | ₹2,000 |

A new headphone has a playback time of 35 hours.

If we use both existing examples as its two nearest neighbours and consider only playback time, the simple average of their prices is:

`(1500 + 2000) / 2 = ₹1,750`

So the predicted price is ₹1,750 in this simplified example.

This is an illustrative calculation, not a universal pricing rule. Actual KNN predictions depend on the number of neighbours (`k`), the distance metric, selected features, feature scaling, and the training data.

### How KNN Works

1. Store the training examples.
2. Receive a new example.
3. Calculate its distance from stored examples.
4. Select the `k` nearest neighbours.
5. For classification, use their labels to determine the predicted class.
6. For regression, combine their target values, commonly by taking an average.

### Advantages

- The concept is relatively easy to understand.
- It can capture local patterns without assuming one simple global equation.

### Limitations

- Predictions may be slow when many examples need to be searched.
- Storing training data can require substantial memory.
- Features with very different scales can distort distance calculations if preprocessing is not appropriate.

### Important Doubt: Can Instance-Based Learning Predict Unseen Data?

Yes. It can generalize to unseen examples by comparing them with stored training examples.

Generalization does not require the system to learn a compact equation.

**Key Point:** Instance-Based Learning uses similarity to stored examples when making predictions.

---

## 3.2 Model-Based Learning

### Definition

**Model-Based Learning** is an approach in which the system fits a model to the training data and uses that model to make predictions for new examples.

The model captures a relationship, pattern, or structure learned from the data.

### Example: A Simplified Price Model

Imagine that a model has learned the following hypothetical relationship:

`Predicted Price = 50 × Playback Hours`

For a headphone with 35 hours of playback:

`Predicted Price = 50 × 35 = ₹1,750`

The model uses its learned relationship to make the prediction rather than searching the original examples and averaging their prices each time.

This is a hypothetical equation for understanding the concept, not a real headphone-pricing formula.

### Common Algorithms

- Linear Regression
- Logistic Regression
- Decision Trees
- Random Forests
- Neural Networks

These algorithms fit different types of models and are suitable for different tasks.

### Advantages

- Prediction can be efficient after the model has been fitted.
- The model can summarize patterns found in the training data.
- Some models, such as linear models and small decision trees, can be relatively interpretable.

### Limitations

- An overly simple model may miss important patterns.
- An overly complex model may overfit the training data.
- Performance depends on the data, features, model assumptions, and evaluation process.

### Important Doubt: Do Instance-Based and Model-Based Learning Both Generalize?

Yes. Both approaches aim to make useful predictions for unseen examples.

- **Instance-Based Learning:** Uses similarity to stored training examples.
- **Model-Based Learning:** Uses a fitted model learned from the training data.

The main difference is how each system uses what it learned to make predictions.

### Instance-Based vs. Model-Based Learning

| Aspect | Instance-Based Learning | Model-Based Learning |
|---|---|---|
| Main idea | Compare new examples with stored examples | Fit a model to the training data |
| Prediction uses | Similarity or distance | Learned parameters, rules, or model structure |
| Example | KNN | Linear Regression |
| Can handle unseen examples? | Yes | Yes |
| Main consideration | Storage and prediction-time distance calculations | Model choice, fitting, assumptions, and overfitting |

**Key Point:** Instance-Based and Model-Based Learning are different approaches to generalization, not a distinction between methods that can and cannot predict new data.

---

# 4. How These Categories Work Together

The three classification criteria describe different aspects of a Machine Learning system.

| Classification Criterion | Main Question | Categories |
|---|---|---|
| Supervision | What kind of labels or feedback are available? | Supervised, Unsupervised, Semi-Supervised, Reinforcement |
| Learning from incoming data | How is the model updated? | Batch, Online |
| Generalization | How does the system use what it learned for new examples? | Instance-Based, Model-Based |

These categories are **not mutually exclusive across the three criteria**. One system can belong to one category from each group.

### Example: An E-Commerce Spam Detector

Suppose a company builds a spam detector that:

- Learns from emails labelled as spam or legitimate.
- Updates incrementally as new labelled examples become available.
- Uses a fitted classification model to classify incoming emails.

The system can be described as:

- **Supervised Learning:** It trains using labelled emails.
- **Online Learning:** It updates incrementally as new training examples arrive.
- **Model-Based Learning:** It uses a fitted model for predictions.

Another spam detector might use batch learning instead. Its supervision type does not determine its update strategy.

---

# 5. Quick Revision

| Type | Main Idea | Example |
|---|---|---|
| Supervised Learning | Learns from labelled examples | House-price prediction |
| Unsupervised Learning | Discovers structure without supplied target labels | Customer clustering |
| Semi-Supervised Learning | Uses labelled and unlabelled data | Product image classification with limited labels |
| Reinforcement Learning | Learns actions through rewards and interaction | Robot learning to walk |
| Batch Learning | Trains or updates in runs using a prepared dataset | Monthly model retraining |
| Online Learning | Updates incrementally as data becomes available | Incremental fraud-model updates |
| Instance-Based Learning | Uses similarity to stored examples | KNN |
| Model-Based Learning | Uses a fitted model | Linear Regression |

## Remember These Three Contrasts

1. **Supervised vs. Unsupervised:** Are target labels supplied during training?
2. **Batch vs. Online:** How is the model updated as data arrives?
3. **Instance-Based vs. Model-Based:** How does the system use what it learned to handle new examples?

---

# 6. Self-Test Questions

Try answering these questions without looking at the notes first.

1. What are `X` and `y` in supervised learning?
2. How is regression different from classification?
3. Does unsupervised learning mean a model cannot process new examples?
4. What is the difference between clustering and PCA?
5. Why can semi-supervised learning be useful when labels are expensive?
6. How does reinforcement learning differ from supervised learning?
7. Does batch learning necessarily mean the prediction service must go offline during retraining?
8. Does online learning necessarily require an internet connection?
9. What is the difference between online learning and mini-batch training?
10. Can an online-learning system use old examples as well as new ones?
11. How does KNN make a prediction for a new example?
12. How does model-based learning differ from instance-based learning?
13. Can a system be supervised, online, and model-based at the same time?

## Self-Test Answers

1. `X` represents the input features; `y` represents the target or desired answer.
2. Regression predicts numerical values, whereas classification predicts categories.
3. No. Some unsupervised algorithms can transform or assign new examples after fitting; this depends on the algorithm.
4. Clustering groups examples, whereas PCA transforms features into a lower-dimensional representation.
5. It can use a small labelled dataset alongside a larger unlabelled dataset.
6. Supervised learning uses examples with desired target answers; reinforcement learning learns through interaction and reward feedback.
7. No. A new model can be trained separately while the current model continues serving predictions.
8. No. Online learning refers to incremental updates, not internet access.
9. Online learning is an update strategy; mini-batch training is a way of processing examples during training.
10. Yes. Some workflows replay old examples or combine old and new data.
11. KNN finds nearby stored examples and uses their labels or target values to make a prediction.
12. Instance-based methods compare with stored examples; model-based methods use a fitted model.
13. Yes. These categories describe separate aspects of a learning system.

---

# 7. Important Takeaways

- These categories describe different aspects of Machine Learning; they are not one list of mutually exclusive algorithms.
- The capabilities of a particular algorithm depend on its implementation and training setup.
- Examples in these notes are illustrative; real results depend on data quality, feature selection, preprocessing, and evaluation.
- A model should be evaluated on suitable held-out data rather than assumed to generalize simply because it performs well on training data.
