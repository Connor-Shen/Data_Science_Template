---
jupyter:
  title: LightGBM examples
  dataset: abalone dataset
  difficulty: Middle
  module: LightGBM
  idx: 8
---

### Step 1. Import necessary libraries.
```python
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
import numpy as np
```

### Step 2. Load the dataset from a URL and Convert the data into a pandas dataframe and assign it to 'abalone_df'.
```python
dataset_url = "lightgbm\lightgbm_dataset08.csv"
column_names = ['Sex', 'Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight', 'Rings']
abalone_df = pd.read_csv(dataset_url, names=column_names)
abalone_df = pd.get_dummies(abalone_df, columns=['Sex'])  # One-hot encoding for categorical variable 'Sex'
```

### Step 3. Split the data into features and target, then into training and testing sets with test size being 0.25.
```python
X = abalone_df.drop('Rings', axis=1)
y = abalone_df['Rings']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
```

### Step 4. Convert the pandas dataframes into LightGBM Dataset format.
```python
dtrain = lgb.Dataset(X_train, label=y_train)
dtest = lgb.Dataset(X_test, label=y_test)
```

### Step 5. Define a LightGBM model 'lgb_abalone_model' for regression with max_depth=22, n_estimators=100, learning_rate=0.06 and num_leaves=5.
```python
lgb_abalone_model = lgb.LGBMRegressor(max_depth=22, n_estimators=100, learning_rate=0.06, num_leaves=5)
```

### Step 6. Train the model with the given parameters: Early Stopping Rounds=10
```python
lgb_abalone_model.fit(X_train, y_train, early_stopping_rounds=10, eval_set=[(X_test, y_test)], verbose=True)
```

### Step 7. Predict the target for the test set and Evaluate the model using the test set.
```python
y_pred = lgb_abalone_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}
R2 Score: {r2}')
```

### Step 8. Report the mean squared error and R2 score.
```python
# This is covered in Step 7
```

### Step 9. Use a histogram to show the feature importance of each feature, using 'Abalone Feature Importance' as Plot Title, 'Features' as X-axis Label, 'Importance' as Y-axis Label.
```python
lgb.plot_importance(lgb_abalone_model)
plt.title('Abalone Feature Importance')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.show()
```

### Step 10. Conduct model parameter tuning for max_depth, learning_rate, n_estimators, num_leaves. select three alternative values of each parameter and output the optimal values of the parameters.
```python
param_grid = {
    'max_depth': [12, 22, 32],
    'learning_rate': [0.03, 0.06, 0.09],
    'n_estimators': [50, 100, 150],
    'num_leaves': [3, 5, 7],
}
grid_search = GridSearchCV(estimator=lgb_abalone_model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=3, verbose=1)
grid_search.fit(X_train, y_train)
print(grid_search.best_params_)
```
