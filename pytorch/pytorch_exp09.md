---
jupyter:
  title: Classification tasks in pytorch
  dataset: Bike Sharing Dataset
  difficulty: Hard
  model: RandomForest
  module: pytorch
  idx: 9
---
# Steps and Python Code for Bike Rental Prediction using the Bike Sharing Dataset

## Step 1. Import necessary libraries.

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.preprocessing import MinMaxScaler
```

## Step 2. Load the dataset from the provided link.

Dataset Link: [Bike Sharing Dataset](https://archive.ics.uci.edu/ml/datasets/bike+sharing+dataset)

```python
bike_sharing_data = pd.read_csv('pytorch_dataset09.csv')
```

## Step 3. Convert the data into a pandas dataframe named `BikeRentalsDF`.

```python
BikeRentalsDF = bike_sharing_data
```

## Step 4. Normalize the `cnt` column using Min-Max Scaling and split it into training and testing sets.

```python
scaler = MinMaxScaler()
BikeRentalsDF['cnt_scaled'] = scaler.fit_transform(BikeRentalsDF[['cnt']])
TestSize = 0.2
X_train, X_test, y_train, y_test = train_test_split(BikeRentalsDF.drop(['cnt', 'cnt_scaled'], axis=1), 
                                                    BikeRentalsDF['cnt_scaled'], 
                                                    test_size=TestSize)
```

## Step 5. Since RandomForestRegressor works with numpy arrays, there's no need to convert them into PyTorch tensors.

(Step omitted since we are using scikit-learn's RandomForestRegressor)

## Step 6. Define a `RandomForest` model with appropriate inputs.

```python
ModelName = RandomForestRegressor
model = ModelName(n_estimators=100, random_state=42)  # Using default hyperparameters
```

## Step 7. There's no need to define a loss function and optimizer as RandomForestRegressor has them inherently defined.

(Step omitted since we are using scikit-learn's RandomForestRegressor)

## Step 8. Train the model for a specified number of epochs.

```python
# In scikit-learn, we do not train for epochs, we fit the model to the training data.
model.fit(X_train, y_train)
```

## Step 9. Evaluate the model using the test set and compute the Mean Squared Error.

```python
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')
```

## Step 10. Report the regression metrics.

Since this is a regression task, we won't have a confusion matrix, accuracy, precision, and recall. Instead, we use regression metrics such as R^2 score, Mean Absolute Error, etc.

```python
from sklearn.metrics import r2_score, mean_absolute_error

r2 = r2_score(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
print(f'R^2 Score: {r2}')
print(f'Mean Absolute Error: {mae}')
```

## Step 11. Visualize the predicted vs actual values.

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.plot(y_test.values, label='Actual Count')
plt.plot(predictions, label='Predicted Count', alpha=0.7)
plt.title('Actual vs Predicted Bike Rental Count')
plt.xlabel('Hours')
plt.ylabel('Normalized Count of Rentals')
plt.legend()
plt.show()
```
