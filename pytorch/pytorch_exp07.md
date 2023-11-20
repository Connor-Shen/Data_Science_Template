---
jupyter:
  title: Classification tasks in pytorch
  dataset: Adult Income
  difficulty: Middle
  model: MLP
  module: pytorch
  idx: 7
---

## Step 1. Import necessary libraries.

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
```

## Step 2. Load the dataset from path named "AdultIncomeDF".

```python

path = "pytorch/pytorch_dataset07.csv"
AdultIncomeDF = pd.read_csv(path)
```

## Step 3. Convert the data into a pandas dataframe named "diabetes_df" and convert the variables using Label Encoding.

```python
# Define column names
columns = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation', 
           'relationship', 'race', 'gender', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income']

# Assign column names to the dataframe
AdultIncomeDF.columns = columns

# Convert categorical columns to numerical using Label Encoding
label_encoders = {}
for categorical_col in ['workclass', 'education', 'marital_status', 'occupation', 'relationship', 'race', 'gender', 'native_country', 'income']:
    label_encoders[categorical_col] = LabelEncoder()
    AdultIncomeDF[categorical_col] = label_encoders[categorical_col].fit_transform(AdultIncomeDF[categorical_col])

```

## Step 4. Normalize the data using Standard Scaling. Split the dataset into features and target variable, then into training and testing sets.

```python
X = AdultIncomeDF.drop('income', axis=1)
y = AdultIncomeDF['income']
TestSize = 0.2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TestSize)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

## Step 5. Convert the numpy arrays into PyTorch tensors.

```python

train_features = torch.tensor(X_train_scaled, dtype=torch.float32)
test_features = torch.tensor(X_test_scaled, dtype=torch.float32)
train_targets = torch.tensor(y_train.values, dtype=torch.float32)
test_targets = torch.tensor(y_test.values, dtype=torch.float32)
```

## Step 6. Define a PyTorch MLP for classification.

```python
class AdultIncomeClassifier(nn.Module):
    def __init__(self, input_size):
        super(AdultIncomeClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x))

# Initialize the model
input_size = X_train.shape[1]
model = AdultIncomeClassifier(input_size)

```

## Step 7. Define the BCELoss as the loss function and Adam as the optimizer with learning rate 0.01.

```python
criterion = nn.BCELoss()  # Binary Cross Entropy Loss for binary classification tasks
optimizer = optim.Adam(model.parameters(), lr=0.01)

```

## Step 8. Train the model for 50 epochs.

```python
Epochs = 50

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
# First, we need to convert predictions to binary (0 or 1)
predictions_binary = (predictions > 0.5).float()

# Then we can calculate the metrics
confusion_matrix = confusion_matrix(test_targets.numpy(), predictions_binary.numpy())
accuracy = accuracy_score(test_targets.numpy(), predictions_binary.numpy())
precision = precision_score(test_targets.numpy(), predictions_binary.numpy())
recall = recall_score(test_targets.numpy(), predictions_binary.numpy())

print(f'Confusion Matrix:\n{confusion_matrix}')
print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')

```

## Step 11. Visualize the actual vs predicted values.

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.scatter(range(len(test_targets)), test_targets, label='Actual Income')
plt.scatter(range(len(predictions)), predictions, label='Predicted Income', color='r')
plt.legend()
plt.title('Actual vs Predicted Income')
plt.xlabel('Sample Index')
plt.ylabel('Income (<=50K: 0, >50K: 1)')
plt.show()

```
