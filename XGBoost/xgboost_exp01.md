---
jupyter:
  title: XGBoost examples
  dataset: iris dataset
  difficulty: EASY
  module: XGBoost
  idx: 1
---

### Step 1. Import necessary libraries.
```python
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
from xgboost import plot_importance
import matplotlib.pyplot as plt
```
### Step 2. Load the dataset from {Dataset Link} and Convert the data into a pandas dataframe and assign it to iris_df.
```python
URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
iris_df = pd.read_csv(URL, header=None, names=column_names)
```
### Step 3. Split the data into features and target "species", then into training and testing sets with test size being 0.2.
```python
X, y = iris_df.iloc[:,:-1], iris_df.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
```
### Step 4. Convert the pandas dataframes into DMatrix, which is a data format that XGBoost uses. And use this DMatrix to train the model.
```python

```
### Step 5. Define a XGBoost model "xgb_model" with max_depth=3, n_estimators=100，learning_rate=0.05.
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