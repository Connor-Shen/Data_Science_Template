---
jupyter:
  title: Classification tasks in pytorch
  dataset: Wine quality
  difficulty: Middle
  model: CNN
  module: pytorch
  idx: 3
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

## Step 2. Load the dataset from path named "WineQualityDF".
```python
url = "pytorch/pytorch_dataset03.csv"
WineQualityDF = pd.read_csv(url, sep=';')
WineQualityDF.dropna(inplace=True)  # Removing rows with missing values for simplicity
```

## Step 3. Convert the data into a pandas dataframe and add the target variable to the dataframe named "WineQualityDF". Convert quality >= 7 to 1 and quality < 7 to 0.
```python
# Convert 'quality' to a binary classification problem (good quality: 1, bad quality: 0)
WineQualityDF['quality'] = WineQualityDF['quality'].apply(lambda x: 1 if x >= 7 else 0)

```

## Step 4. Normalize the data using Standard Scaling. Split the dataset into features and target variable, then into training and testing sets.
```python
X = WineQualityDF.drop('quality', axis=1)
y = WineQualityDF['quality']
TestSize = 0.2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TestSize)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

## Step 5. Convert the numpy arrays into PyTorch tensors.
```python

train_features = torch.tensor(X_train_scaled, dtype=torch.float32).unsqueeze(1)
test_features = torch.tensor(X_test_scaled, dtype=torch.float32).unsqueeze(1)
train_targets = torch.tensor(y_train.values, dtype=torch.float32)
test_targets = torch.tensor(y_test.values, dtype=torch.float32)

```

## Step 6. Define a CNN for classification with appropriate inputs and 1 output.
```python
class WineQualityCNN(nn.Module):
    def __init__(self):
        super(WineQualityCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, 3, 1)  # Convolutional layer
        self.fc1 = nn.Linear(16 * (X_train.shape[1] - 2), 64)
        self.fc2 = nn.Linear(64, 1)
        
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))

model = WineQualityCNN()

```

## Step 7. Define the CrossEntropyLoss as the loss function and Adam as the optimizer with learning rate 0.01.
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

## Step 10. Report the confusion matrix and corresponding accuracy, precision and recall.
```python
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

with torch.no_grad():
    predictions = model(test_features).squeeze()
    predicted_classes = (predictions > 0.5).float()
    cm = confusion_matrix(test_targets, predicted_classes)
    accuracy = accuracy_score(test_targets, predicted_classes)
    precision = precision_score(test_targets, predicted_classes)
    recall = recall_score(test_targets, predicted_classes)

print(f'Confusion Matrix:\n{cm}')
print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')

```

## Step 11. Visualize the predicted vs actual values.
```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.scatter(range(len(test_targets)), test_targets, label='Actual Quality')
plt.scatter(range(len(predictions)), predictions, label='Predicted Quality', color='r')
plt.legend()
plt.title('Actual vs Predicted Wine Quality')
plt.xlabel('Sample Index')
plt.ylabel('Quality (Good: 1, Bad: 0)')
plt.show()

```