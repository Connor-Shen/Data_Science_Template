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
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
```
### Step 2. Load the dataset and Convert the data into a pandas dataframe and assign it to penguins_df.
```python
URL = "XGBoost\xgboost_dataset03.csv"
penguins_df = pd.read_csv(URL)
```
### Step 3. Split the data into features and target "Species", then into training and testing sets with test size being 0.3.
```python
X = penguins_df.drop('species', axis=1)
y = penguins_df['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```
### Step 4. Convert the pandas dataframes into DMatrix, which is a data format that XGBoost uses.
```python
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)
```
### Step 5. Define a XGBoost model "xgb_model" with max_depth=4, n_estimators=100，learning_rate=0.01.
```python
xgb_model = xgb.XGBClassifier(max_depth=4, n_estimators=100, learning_rate=0.01)
```
### Step 6. Train the model with the given parameters Early Stopping Rounds=10, Evaluation Metric='mlogloss'.
```python
eval_set = [(X_train, y_train), (X_test, y_test)]
xgb_model.fit(X_train, y_train, early_stopping_rounds=10, eval_metric="mlogloss", eval_set=eval_set, verbose=True)
```
### Step 7. Predict the target for the test set and Evaluate the model using the test set.
```python
y_pred = xgb_model.predict(X_test)
```
### Step 8. Report the confusion matrix and corresponding accuracy, precision, and recall.
```python
conf_matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
print(conf_matrix, accuracy, precision, recall)
```
### Step 9. Use a histogram to show the feature importance of each feature，using title = 'Feature Importance', X label = 'Features', Y label = 'Importance'.
```python
xgb.plot_importance(xgb_model)
plt.title('Feature Importance')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.show()
```
### Step 10. Conduct model parameter tuning for max_depth, learning_rate, n_estimators, select three alternative values of each parameter and output the optimal values of the parameters.
```python
param_grid = {
    'max_depth': [3, 4, 6],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 150, 200],
}
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, scoring='accuracy', cv=3, verbose=1)
grid_search.fit(X_train, y_train)
print(grid_search.best_params_)
```
