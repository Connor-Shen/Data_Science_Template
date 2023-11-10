---
jupyter:
  title: LightGBM examples
  dataset: iris dataset
  difficulty: EASY
  module: LightGBM
  idx: 1
---

### Step 1. Import necessary libraries.
```python
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
```

### Step 2. Load the dataset from https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data and Convert the data into a pandas dataframe and assign it to iris_df.
```python
URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
iris_df = pd.read_csv(URL, header=None, names=column_names)
```

### Step 3. Split the data into features and target "species", then into training and testing sets with test size being 0.2.
```python
X = iris_df.drop('species', axis=1)
y = iris_df['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### Step 4. Prepare LightGBM Dataset for training and testing.
```python
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
```

### Step 5. Define a LightGBM model lgbm_model with max_depth=4, n_estimators=120, learning_rate=0.01 and num_leaves=31.
```python
lgbm_model = lgb.LGBMClassifier(max_depth=4, n_estimators=120, learning_rate=0.01, num_leaves=31)
```

### Step 6. Train the model with the given parameters Early Stopping Rounds=10, Evaluation Metric='logloss'.
```python
lgbm_model.fit(X_train, y_train, early_stopping_rounds=10, eval_metric='logloss', eval_set=[(X_test, y_test)], verbose=True)
```

### Step 7. Predict the target for the test set and Evaluate the model using the test set.
```python
y_pred = lgbm_model.predict(X_test)
```

### Step 8. Report the confusion matrix and corresponding accuracy, precision, and recall.
```python
conf_matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
print(conf_matrix, accuracy, precision, recall)
```

### Step 9. Use a histogram to show the feature importance of each feature, using title = 'Feature Importance', X label = 'Features', Y label = 'Importance'.
```python
lgb.plot_importance(lgbm_model, max_num_features=30, title='Feature Importance', xlabel='Features', ylabel='Importance')
plt.show()
```

### Step 10. Conduct model parameter tuning for max_depth, learning_rate, n_estimators, num_leaves. select three alternative values of each parameter and output the optimal values of the parameters.
```python
param_grid = {
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 120, 150],
    'num_leaves': [20, 31, 40],
}
grid_search = GridSearchCV(estimator=lgbm_model, param_grid=param_grid, scoring='accuracy', cv=3, verbose=1)
grid_search.fit(X_train, y_train)
print(grid_search.best_params_)
```
