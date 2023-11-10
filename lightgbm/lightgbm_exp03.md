---
jupyter:
  title: LightGBM examples
  dataset: breast cancer dataset
  difficulty: Middle
  module: LightGBM
  idx: 3
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

### Step 2. Load the dataset from a URL and Convert the data into a pandas dataframe and assign it to 'cancer_df'.
```python
dataset_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
column_names = ['ID', 'Diagnosis', 'Radius_mean', 'Texture_mean', 'Perimeter_mean', 'Area_mean', 'Smoothness_mean', 'Compactness_mean', 'Concavity_mean', 'Concave_points_mean', 'Symmetry_mean', 'Fractal_dimension_mean', 'Radius_se', 'Texture_se', 'Perimeter_se', 'Area_se', 'Smoothness_se', 'Compactness_se', 'Concavity_se', 'Concave_points_se', 'Symmetry_se', 'Fractal_dimension_se', 'Radius_worst', 'Texture_worst', 'Perimeter_worst', 'Area_worst', 'Smoothness_worst', 'Compactness_worst', 'Concavity_worst', 'Concave_points_worst', 'Symmetry_worst', 'Fractal_dimension_worst']
cancer_df = pd.read_csv(dataset_url, names=column_names)
cancer_df = cancer_df.drop('ID', axis=1)  # Drop the ID column as it's not a feature
cancer_df['Diagnosis'] = cancer_df['Diagnosis'].map({'M': 1, 'B': 0})  # Convert the 'Diagnosis' column to binary
```

### Step 3. Split the data into features and target, then into training and testing sets with test size being 0.25.
```python
X = cancer_df.drop('Diagnosis', axis=1)
y = cancer_df['Diagnosis']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
```

### Step 4. Convert the pandas dataframes into LightGBM Dataset format.
```python
dtrain = lgb.Dataset(X_train, label=y_train)
dtest = lgb.Dataset(X_test, label=y_test)
```

### Step 5. Define a LightGBM model 'lgb_cancer_model' with max_depth=40, n_estimators=150, learning_rate=0.05 and num_leaves=4.
```python
lgb_cancer_model = lgb.LGBMClassifier(max_depth=40, n_estimators=150, learning_rate=0.05, num_leaves=4)
```

### Step 6. Train the model with the given parameters: Early Stopping Rounds=10, Evaluation Metric='binary_logloss'
```python
eval_set = [(X_train, y_train), (X_test, y_test)]
lgb_cancer_model.fit(X_train, y_train, early_stopping_rounds=10, eval_metric='binary_logloss', eval_set=eval_set, verbose=True)
```

### Step 7. Predict the target for the test set and Evaluate the model using the test set.
```python
y_pred = lgb_cancer_model.predict(X_test)
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

### Step 9. Use a histogram to show the feature importance of each feature, using 'Cancer Feature Importance' as Plot Title, 'Features' as X-axis Label, 'Importance' as Y-axis Label.
```python
lgb.plot_importance(lgb_cancer_model)
plt.title('Cancer Feature Importance')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.show()
```

### Step 10. Conduct model parameter tuning for max_depth, learning_rate, n_estimators, num_leaves. select three alternative values of each parameter and output the optimal values of the parameters.
```python
param_grid = {
    'max_depth': [20, 40, 60],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 150, 200],
    'num_leaves': [2, 4, 6],
}
grid_search = GridSearchCV(estimator=lgb_cancer_model, param_grid=param_grid, scoring='accuracy', cv=3, verbose=1)
grid_search.fit(X_train, y_train)
print(grid_search.best_params_)
```
