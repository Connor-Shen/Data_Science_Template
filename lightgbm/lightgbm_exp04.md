---
jupyter:
  title: LightGBM examples
  dataset: diabetes dataset
  difficulty: Middle
  module: LightGBM
  idx: 4
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

### Step 2. Load the dataset from a URL and Convert the data into a pandas dataframe and assign it to 'diabetes_df'.
```python
dataset_url = "lightgbm\lightgbm_dataset04.csv"
column_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
diabetes_df = pd.read_csv(dataset_url, names=column_names)
```

### Step 3. Split the data into features and target, then into training and testing sets with test size being 0.3.
```python
X = diabetes_df.drop('Outcome', axis=1)
y = diabetes_df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

### Step 4. Convert the pandas dataframes into LightGBM Dataset format.
```python
dtrain = lgb.Dataset(X_train, label=y_train)
dtest = lgb.Dataset(X_test, label=y_test)
```

### Step 5. Define a LightGBM model 'lgb_diabetes_model' with max_depth=35, n_estimators=120, learning_rate=0.07 and num_leaves=3.
```python
lgb_diabetes_model = lgb.LGBMClassifier(max_depth=35, n_estimators=120, learning_rate=0.07, num_leaves=3)
```

### Step 6. Train the model with the given parameters: Early Stopping Rounds=15, Evaluation Metric='binary_error'
```python
eval_set = [(X_train, y_train), (X_test, y_test)]
lgb_diabetes_model.fit(X_train, y_train, early_stopping_rounds=15, eval_metric='binary_error', eval_set=eval_set, verbose=True)
```

### Step 7. Predict the target for the test set and Evaluate the model using the test set.
```python
y_pred = lgb_diabetes_model.predict(X_test)
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

### Step 9. Use a histogram to show the feature importance of each feature, using 'Diabetes Feature Importance' as Plot Title, 'Features' as X-axis Label, 'Importance' as Y-axis Label.
```python
lgb.plot_importance(lgb_diabetes_model)
plt.title('Diabetes Feature Importance')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.show()
```

### Step 10. Conduct model parameter tuning for max_depth, learning_rate, n_estimators, num_leaves. select three alternative values of each parameter and output the optimal values of the parameters.
```python
param_grid = {
    'max_depth': [25, 35, 45],
    'learning_rate': [0.03, 0.07, 0.1],
    'n_estimators': [80, 120, 160],
    'num_leaves': [2, 3, 5],
}
grid_search = GridSearchCV(estimator=lgb_diabetes_model, param_grid=param_grid, scoring='accuracy', cv=3, verbose=1)
grid_search.fit(X_train, y_train)
print(grid_search.best_params_)
```
