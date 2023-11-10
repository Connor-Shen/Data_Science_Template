---
jupyter:
  title: LightGBM examples
  dataset: wine quality dataset
  difficulty: EASY
  module: LightGBM
  idx: 2
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

### Step 2. Load the dataset from https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv and Convert the data into a pandas dataframe and assign it to wine_df.
```python
URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
wine_df = pd.read_csv(URL, sep=';')
```

### Step 3. Split the data into features and target "quality", then into training and testing sets with test size being 0.25.
```python
X = wine_df.drop('quality', axis=1)
y = wine_df['quality']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
```

### Step 4. Prepare LightGBM Dataset for training and testing.
```python
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
```

### Step 5. Define a LightGBM model wine_lgbm with max_depth=30, n_estimators=150, learning_rate=0.05 and num_leaves=40.
```python
wine_lgbm = lgb.LGBMClassifier(max_depth=30, n_estimators=150, learning_rate=0.05, num_leaves=40)
```

### Step 6. Train the model with the given parameters Early Stopping Rounds=15, Evaluation Metric='rmse'.
```python
wine_lgbm.fit(X_train, y_train, early_stopping_rounds=15, eval_metric='rmse', eval_set=[(X_test, y_test)], verbose=True)
```

### Step 7. Predict the target for the test set and Evaluate the model using the test set.
```python
y_pred = wine_lgbm.predict(X_test)
```

### Step 8. Report the confusion matrix and corresponding accuracy, precision, and recall.
```python
conf_matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
print(conf_matrix, accuracy, precision, recall)
```

### Step 9. Use a histogram to show the feature importance of each feature, using title = 'Wine Quality Feature Importance', X label = 'Features', Y label = 'Score'.
```python
lgb.plot_importance(wine_lgbm, max_num_features=30, title='Wine Quality Feature Importance', xlabel='Features', ylabel='Score')
plt.show()
```

### Step 10. Conduct model parameter tuning for max_depth, learning_rate, n_estimators, num_leaves. select three alternative values of each parameter and output the optimal values of the parameters.
```python
param_grid = {
    'max_depth': [20, 30, 40],
    'learning_rate': [0.03, 0.05, 0.07],
    'n_estimators': [120, 150, 180],
    'num_leaves': [30, 40, 50],
}
grid_search = GridSearchCV(estimator=wine_lgbm, param_grid=param_grid, scoring='accuracy', cv=3, verbose=1)
grid_search.fit(X_train, y_train)
print(grid_search.best_params_)
```
