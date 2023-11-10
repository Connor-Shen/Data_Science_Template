---
jupyter:
  title: LightGBM examples
  dataset: boston houseprice dataset
  difficulty: Middle
  module: LightGBM
  idx: 7
---

### Step 1. Import necessary libraries.
```python
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_boston
import numpy as np
```

### Step 2. Load the dataset and Convert the data into a pandas dataframe and assign it to 'boston_df'.
```python
boston = load_boston()
boston_df = pd.DataFrame(boston.data, columns=boston.feature_names)
boston_df['PRICE'] = boston.target
```

### Step 3. Split the data into features and target, then into training and testing sets with test size being 0.3.
```python
X = boston_df.drop('PRICE', axis=1)
y = boston_df['PRICE']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

### Step 4. Convert the pandas dataframes into LightGBM Dataset format.
```python
dtrain = lgb.Dataset(X_train, label=y_train)
dtest = lgb.Dataset(X_test, label=y_test)
```

### Step 5. Define a LightGBM model 'lgb_boston_model' for regression with max_depth=25, n_estimators=120, learning_rate=0.05 and num_leaves=4.
```python
lgb_boston_model = lgb.LGBMRegressor(max_depth=25, n_estimators=120, learning_rate=0.05, num_leaves=4)
```

### Step 6. Train the model with the given parameters: Early Stopping Rounds=10
```python
lgb_boston_model.fit(X_train, y_train, early_stopping_rounds=10, eval_set=[(X_test, y_test)], verbose=True)
```

### Step 7. Predict the target for the test set and Evaluate the model using the test set.
```python
y_pred = lgb_boston_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}
R2 Score: {r2}')
```

### Step 8. Report the mean squared error and R2 score.
```python
# This is covered in Step 7
```

### Step 9. Use a histogram to show the feature importance of each feature, using 'Boston Housing Feature Importance' as Plot Title, 'Features' as X-axis Label, 'Importance' as Y-axis Label.
```python
lgb.plot_importance(lgb_boston_model)
plt.title('Boston Housing Feature Importance')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.show()
```

### Step 10. Conduct model parameter tuning for max_depth, learning_rate, n_estimators, num_leaves. select three alternative values of each parameter and output the optimal values of the parameters.
```python
param_grid = {
    'max_depth': [15, 25, 35],
    'learning_rate': [0.03, 0.05, 0.07],
    'n_estimators': [80, 120, 160],
    'num_leaves': [2, 4, 6],
}
grid_search = GridSearchCV(estimator=lgb_boston_model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=3, verbose=1)
grid_search.fit(X_train, y_train)
print(grid_search.best_params_)
```
