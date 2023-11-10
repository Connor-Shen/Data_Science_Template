---
jupyter:
  title: Classification tasks in pytorch
  dataset: COVID_19_Severity
  difficulty: Middle
  model: linear regression
  module: pytorch
  idx: 3
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

## Step 2. Load the dataset.

```python
# Replace 'path_to_covid_dataset.csv' with the actual path of the downloaded dataset
Dataset Link: "https://data.cdc.gov/Public-Health-Surveillance/COVID-19-Case-Surveillance-Public-Use-Data/ynhu-f2s2" 
covid_df = pd.read_csv('path_to_covid_dataset.csv')
```

## Step 3. Convert the data into a pandas dataframe named "covid_df" and add the target variable.

```python
# Assuming the dataset has a 'Severity' column as the target
# No additional code is needed here since the dataset already includes the target variable
```

## Step 4. Normalize the data of "age" using Min-Max Scaling and split it into training and testing sets with a test size being 0.2.

```python
scaler = MinMaxScaler()
covid_df['age_normalized'] = scaler.fit_transform(covid_df[['Age']])

X = covid_df.drop('Severity', axis=1)
y = covid_df['Severity']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

## Step 5. Convert the numpy arrays into PyTorch tensors.

```python
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)
```

## Step 6. Define a "CovidSeverityPredictor" model with appropriate inputs and 1 output (label).

```python
class CovidSeverityPredictor(nn.Module):
    def __init__(self):
        super(CovidSeverityPredictor, self).__init__()
        self.fc1 = nn.Linear(X_train.shape[1], 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

model = CovidSeverityPredictor()
```

## Step 7. Define "CrossEntropyLoss" as the loss function and "Adam" as the optimizer with a learning rate of 0.01.

```python
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
```

## Step 8. Train the model for 20 epochs.

```python
for epoch in range(20):
    optimizer.zero_grad()
    output = model(X_train_tensor)
    loss = criterion(output, y_train_tensor.long())
    loss.backward()
    optimizer.step()

    if epoch % 5 == 0:
        print(f'Epoch {epoch+1}/20 - Loss: {loss.item()}')
```

## Step 9. Evaluate the model using the test set and compute the "CrossEntropyLoss".

```python
with torch.no_grad():
    y_pred = model(X_test_tensor)
    y_pred_classes = y_pred.argmax(dim=1)
    loss = criterion(y_pred, y_test_tensor.long())
    print(f'Test Loss: {loss.item()}')
```

## Step 10. Report the confusion matrix and corresponding accuracy, precision, and recall.

```python
y_pred_np = y_pred_classes.numpy()
y_test_np = y_test_tensor.numpy()
cm = confusion_matrix(y_test_np, y_pred_np)
accuracy = accuracy_score(y_test_np, y_pred_np)
precision = precision_score(y_test_np, y_pred_np, average='weighted')
recall = recall_score(y_test_np, y_pred_np, average='weighted')

print('Confusion Matrix:')
print(cm)
print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
```

## Step 11. Visualize the predicted vs actual values.

```python
plt.figure(figsize=(10, 6))
plt.scatter(range(len(y_test_np)), y_test_np, label='Actual')
plt.scatter(range(len(y_pred_np)), y_pred_np, label='Predicted', color='red')
plt.title('Predicted vs Actual Values')
plt.xlabel('Sample index')
plt.ylabel('COVID-19 Severity Outcome')
plt.legend()
plt.show()
```
