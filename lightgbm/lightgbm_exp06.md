---
jupyter:
  title: LightGBM examples
  dataset: auto mobile dataset
  difficulty: Middle
  module: LightGBM
  idx: 6
---

### Step 1. Import necessary libraries.
```python
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
import numpy as np
```

### Step 2. Load the dataset from a URL and Convert the data into a pandas dataframe and assign it to 'auto_df'.
```python
dataset_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"
column_names = ['symboling', 'normalized_losses', 'make', 'fuel_type', 'aspiration', 'num_doors', 'body_style', 'drive_wheels', 'engine_location', 'wheel_base', 'length', 'width', 'height', 'curb_weight', 'engine_type', 'num_cylinders', 'engine_size', 'fuel_system', 'bore', 'stroke', 'compression_ratio', 'horsepower', 'peak_rpm', 'city_mpg', 'highway_mpg', 'price']
auto_df = pd.read_csv(dataset_url, names=column_names)
auto_df = auto_df.replace('?', np.nan).dropna()  # Replace '?' with NaN and drop rows with NaN values
auto_df['price'] = pd.to_numeric(auto_df['price'])  # Convert 'price' to numeric
auto_df['high_price'] = auto_df['price'].apply(lambda x: 1 if x > auto_df['price'].median() else 0)  # Create a binary target variable
```

### Step 3. Split the data into features and target, then into training and testing sets with test size being 0.25.
```python
X = auto_df.drop(['price', 'high_price'], axis=1)
y = auto_df['high_price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
```

### Step 4. Convert the pandas dataframes into LightGBM Dataset format.
```python
dtrain = lgb.Dataset(X_train, label=y_train)
dtest = lgb.Dataset(X_test, label=y_test)
```

### Step 5. Define a LightGBM model 'lgb_auto_model' with max_depth=28, n_estimators=110, learning_rate=0.08 and num_leaves=6.
```python
lgb_auto_model = lgb.LGBMClassifier(max_depth=28, n_estimators=110, learning_rate=0.08, num_leaves=6)
```

### Step 6. Train the model with the given parameters: Early Stopping Rounds=14, Evaluation Metric='binary_logloss'
```python
eval_set = [(X_train, y_train), (X_test, y_test)]
lgb_auto_model.fit(X_train, y_train, early_stopping_rounds=14, eval_metric='binary_logloss', eval_set=eval_set, verbose=True)
```

### Step 7. Predict the target for the test set and Evaluate the model using the test set.
```python
y_pred = lgb_auto_model.predict(X_test)
```

### Step 8. Report the confusion matrix and corresponding accuracy, precision, and recall.
```python
conf_matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
print(f'Confusion Matrix:
{conf_matrix}
Accuracy: {accuracy}
Precision: {precision}
Recall: {recall}')
```

### Step 9. Use a histogram to show the feature importance of each feature, using 'Automobile Feature Importance' as Plot Title, 'Features' as X-axis Label, 'Importance' as Y-axis Label.
```python
lgb.plot_importance(lgb_auto_model)
plt.title('Automobile Feature Importance')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.show()
```

### Step 10. Conduct model parameter tuning for max_depth, learning_rate, n_estimators, num_leaves. select three alternative values of each parameter and output the optimal values of the parameters.
```python
param_grid = {
    'max_depth': [18, 28, 38],
    'learning_rate': [0.04, 0.08, 0.12],
    'n_estimators': [70, 110, 150],
    'num_leaves': [4, 6, 8],
}
grid_search = GridSearchCV(estimator=lgb_auto_model, param_grid=param_grid, scoring='accuracy', cv=3, verbose=1)
grid_search.fit(X_train, y_train)
print(grid_search.best_params_)
```
