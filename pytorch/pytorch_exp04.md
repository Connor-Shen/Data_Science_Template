---
jupyter:
  title: Classification tasks in pytorch
  dataset: Forest Fires Dataset
  difficulty: Middle
  model: linear regression
  module: pytorch
  idx: 4
---

## Step 1. Import necessary libraries.

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
```

## Step 2. Load the dataset.

Dataset Link: [Forest Fires Dataset Download](https://archive.ics.uci.edu/ml/datasets/forest+fires)

```python
# Load the dataset directly using the UCI ML Repository URL
forest_fires_df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/forest-fires/forestfires.csv')
```

## Step 3. Convert the data into a pandas dataframe and preprocess it for the regression model.

```python
# Assuming the dataset includes features and 'area' as the target variable
ForestFiresDF = forest_fires_df
# Convert categorical features to dummy variables
ForestFiresDF = pd.get_dummies(ForestFiresDF)
```

## Step 4. Split the dataset into features and target variable, then into training and testing sets.

```python
X = ForestFiresDF.drop('area', axis=1)
y = ForestFiresDF['area']
TestSize = 0.2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TestSize)
```

## Step 5. Standardize the features and convert the data into PyTorch tensors.

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

train_features = torch.tensor(X_train_scaled, dtype=torch.float32)
test_features = torch.tensor(X_test_scaled, dtype=torch.float32)
train_targets = torch.tensor(y_train.values, dtype=torch.float32)
test_targets = torch.tensor(y_test.values, dtype=torch.float32)
```

## Step 6. Define a neural network for regression.

```python
class FireAreaPredictor(nn.Module):
    def __init__(self, input_size):
        super(FireAreaPredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Initialize the model
input_size = X_train.shape[1]
model = FireAreaPredictor(input_size)
```

## Step 7. Define the loss function and optimizer.

```python
criterion = nn.MSELoss()  # Mean Squared Error Loss for regression tasks
optimizer = optim.Adam(model.parameters(), lr=0.01)
```

## Step 8. Train the model.

```python
Epochs = 30

for epoch in range(Epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(train_features)
    loss = criterion(outputs.squeeze(), train_targets)
    loss.backward()
    optimizer.step()
    print(f'Epoch {epoch+1}/{Epochs}, Loss: {loss.item()}')
```

## Step 9. Evaluate the model using the test set and compute the loss.

```python
model.eval()
with torch.no_grad():
    predictions = model(test_features)
    test_loss = criterion(predictions.squeeze(), test_targets)
print(f'Test Loss: {test_loss.item()}')
```

## Step 10. Report the confusion matrix and corresponding accuracy, precision, and recall.
```python
with torch.no_grad():
    predictions = model(test_features).squeeze()
    predicted_classes = predictions.round()
    cm = confusion_matrix(test_targets, predicted_classes)
    accuracy = accuracy_score(test_targets, predicted_classes)
    precision = precision_score(test_targets, predicted_classes)
    recall = recall_score(test_targets, predicted_classes)

print(f'Confusion Matrix:\n{cm}')
print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
```

## Step 11. Visualize the actual vs predicted values.

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.scatter(range(len(test_targets)), test_targets, label='Actual Area')
plt.scatter(range(len(predictions)), predictions, label='Predicted Area', color='r')
plt.legend()
plt.title('Actual vs Predicted Area of Forest Fires')
plt.xlabel('Sample Index')
plt.ylabel('Burned Area (hectares)')
plt.show()
```
