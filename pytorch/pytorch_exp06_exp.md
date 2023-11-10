---
jupyter:
  title: Classification tasks in pytorch
  dataset: Human Activity Recognition
  difficulty: Hard
  model: CNN
  module: pytorch
  idx: 6
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

## Step 2. Load the dataset.

Dataset Link: [Human Activity Recognition Using Smartphones Dataset](https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones)

```python
# Data loading code will go here
# Assuming the data is split into features (X) and labels (y) files
X = pd.read_csv('features.csv')
y = pd.read_csv('labels.csv')

```

## Step 3. Convert the data into a pandas dataframe and preprocess it for the model.

```python
# Data preprocessing code will go here
HARDataFrame = pd.concat([X, y], axis=1)
```

## Step 4. Normalize the sensor data using StandardScaler and split it into training and testing sets.

```python
# Normalization and train-test split code will go here
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
y_encoded = LabelEncoder().fit_transform(y.values.ravel())
TestSize = 0.2
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=TestSize)

```

## Step 5. Reshape the data for CNN input and one-hot encode the labels.

```python
# Data reshaping and one-hot encoding code will go here
train_features = torch.tensor(X_train, dtype=torch.float32)
test_features = torch.tensor(X_test, dtype=torch.float32)
train_labels = torch.tensor(y_train, dtype=torch.long)
test_labels = torch.tensor(y_test, dtype=torch.long)

```

## Step 6. Define a CNN model with appropriate layers for time-series classification.

```python
# CNN model definition code will go here
class HAR_CNN(nn.Module):
    def __init__(self):
        super(HAR_CNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(128, 6)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc1(x)
        return x

# Instantiate the model
model = HAR_CNN()

```

## Step 7. Compile the CNN with a loss function and an optimizer.

```python
# Model compilation code will go here
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

```

## Step 8. Train the CNN model.

```python
# Model training code will go here
Epochs = 25

for epoch in range(Epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(train_features)
    loss = criterion(outputs, train_labels)
    loss.backward()
    optimizer.step()
    print(f'Epoch {epoch+1}/{Epochs}, Loss: {loss.item()}')

```

## Step 9. Evaluate the CNN model using the test set.

```python
# Model evaluation code will go here
model.eval()
with torch.no_grad():
    predicted_labels = model(test_features)
    test_loss = criterion(predicted_labels, test_labels)
print(f'Test Loss: {test_loss.item()}')

```

## Step 10. Report the confusion matrix and corresponding accuracy, precision, and recall.

```python
# Classification metrics reporting code will go here
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

_, predicted_classes = torch.max(predicted_labels, 1)
cm = confusion_matrix(test_labels, predicted_classes)
accuracy = accuracy_score(test_labels, predicted_classes)
precision = precision_score(test_labels, predicted_classes, average='weighted')
recall = recall_score(test_labels, predicted_classes, average='weighted')

print(f'Confusion Matrix:\n{cm}')
print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')

```

## Step 11. Visualize the predicted vs actual labels.

```python
# Visualization code will go here
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.plot(test_labels, label='Actual Labels')
plt.plot(predicted_classes, label='Predicted Labels', alpha=0.7)
plt.title('Actual vs Predicted Labels for Human Activity Recognition')
plt.xlabel('Sample index')
plt.ylabel('Activity')
plt.legend()
plt.show()

```
