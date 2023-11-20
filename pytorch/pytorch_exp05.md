---
jupyter:
  title: Classification tasks in pytorch
  dataset: pima-indians-diabetes
  difficulty: Middle
  model: MLP
  module: pytorch
  idx: 5
---

## Step 1. Import necessary libraries.

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
```

## Step 2. Load the dataset named "diabetes_df".

```python
# Replace 'path_to_dataset.csv' with the actual path of the downloaded dataset
Dataset Link: "https://www.kaggle.com/uciml/pima-indians-diabetes-database" 
diabetes_df = pd.read_csv('pytorch_dataset05.csv')
```

## Step 3. Convert the data into a pandas dataframe named "diabetes_df" and add the target variable.

```python
# Assuming the dataset has a 'Outcome' column as the target
# No additional code is needed here since the dataset already includes the target variable
```

## Step 4. Normalize the data of "glucose" using Min-Max Scaling and split it into training and testing sets with a test size being 0.2.

```python
scaler = MinMaxScaler()
diabetes_df['glucose_normalized'] = scaler.fit_transform(diabetes_df[['Glucose']])

X = diabetes_df.drop('Outcome', axis=1)
y = diabetes_df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

## Step 5. Convert the numpy arrays into PyTorch tensors.

```python
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)
```

## Step 6. Define a "DiabetesPredictor" model with appropriate inputs and 1 output (label).

```python
class DiabetesPredictor(nn.Module):
    def __init__(self):
        super(DiabetesPredictor, self).__init__()
        self.fc1 = nn.Linear(X_train.shape[1], 12)
        self.fc2 = nn.Linear(12, 8)
        self.fc3 = nn.Linear(8, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

model = DiabetesPredictor()
```

## Step 7. Define "BinaryCrossEntropy" as the loss function and "Adam" as the optimizer with a learning rate of 0.01.

```python
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
```

## Step 8. Train the model for 20 epochs.

```python
for epoch in range(20):
    optimizer.zero_grad()
    output = model(X_train_tensor)
    loss = criterion(output.squeeze(), y_train_tensor)
    loss.backward()
    optimizer.step()

    if epoch % 5 == 0:
        print(f'Epoch {epoch+1}/20 - Loss: {loss.item()}')
```

## Step 9. Evaluate the model using the test set and compute the "BinaryCrossEntropy".

```python
with torch.no_grad():
    y_pred = model(X_test_tensor)
    y_pred = y_pred.squeeze().round()
    loss = criterion(y_pred, y_test_tensor)
    print(f'Test Loss: {loss.item()}')
```

## Step 10. Report the confusion matrix and corresponding accuracy, precision, and recall.

```python
y_pred_np = y_pred.numpy()
y_test_np = y_test_tensor.numpy()
cm = confusion_matrix(y_test_np, y_pred_np)
accuracy = accuracy_score(y_test_np, y_pred_np)
precision = precision_score(y_test_np, y_pred_np)
recall = recall_score(y_test_np, y_pred_np)

print('Confusion Matrix:')
print(cm)
print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
```

## Step 11. Visualize the predicted vs actual values.

```python
plt.figure(figsize=(10, 6))
plt.plot(y_test_np, label='Actual')
plt.plot(y_pred_np, label='Predicted')
plt.title('Predicted vs Actual Values')
plt.xlabel('Sample index')
plt.ylabel('Diabetes Outcome')
plt.legend()
plt.show()
```
