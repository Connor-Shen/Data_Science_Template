---
jupyter:
  title: XGBoost examples
  dataset: penguins
  difficulty: Middle
  module: XGBoost
  idx: 3
---

### Step 1. Import necessary libraries.
```python

```
### Step 2. Load the dataset from {Dataset Link} and Convert the data into a pandas dataframe and assign it to penguins_df.
```python

```
### Step 3. Split the data into features and target "Species", then into training and testing sets with test size being 0.3.
```python

```
### Step 4. Convert the pandas dataframes into DMatrix, which is a data format that XGBoost uses.
```python

```
### Step 5. Define a XGBoost model "xgb_model" with max_depth=4, n_estimators=100，learning_rate=0.01.
```python

```
### Step 6. Train the model with the given parameters Early Stopping Rounds=10, Evaluation Metric='mlogloss'.
```python

```
### Step 7. Predict the target for the test set and Evaluate the model using the test set.
```python

```
### Step 8. Report the confusion matrix and corresponding accuracy, precision, and recall.
```python

```
### Step 9. Use a histogram to show the feature importance of each feature，using title = 'Feature Importance', X label = 'Features', Y label = 'Importance'.
```python

```
### Step 10. Conduct model parameter tuning for max_depth, learning_rate, n_estimators, select three alternative values of each parameter and output the optimal values of the parameters.
```python

```