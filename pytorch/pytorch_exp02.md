---
jupyter:
  title: Classification tasks in pytorch
  dataset: Breast Cancer Wisconsin
  difficulty: EASY
  model: linear regression
  module: pytorch
  idx: 2
---

## Step 1. Import necessary libraries.
```python
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
```

## Step 2. Load the dataset from "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data".
```python
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"
column_names = ["id", "clump_thickness", "cell_size", "cell_shape", "marginal_adhesion", "epithelial_cell_size", "bare_nuclei", "bland_chromatin", "normal_nucleoli", "mitoses", "class"]
cancer_df = pd.read_csv(url, names=column_names, na_values="?")
cancer_df.dropna(inplace=True)  # Removing rows with missing values for simplicity
```

## Step 3. Convert the data into a pandas dataframe and add the target variable to the dataframe named "cancer_df". Convert benign to 0 and malignant to 1.
```python
cancer_df["class"] = cancer_df["class"].map({2: 0, 4: 1})
```

## Step 4. Normalize the data using Min-Max Scaling and split it into training and testing sets with test size being 0.15.
```python
X = cancer_df.drop(["id", "class"], axis=1).values
y = cancer_df["class"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

## Step 5. Convert the numpy arrays into PyTorch tensors.
```python
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.LongTensor(y_train)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.LongTensor(y_test)
```

## Step 6. Define a simple neural network model with appropriate inputs and 1 output.
```python
class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 10)
        self.fc2 = nn.Linear(10, 2)  # 2 classes in the output

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleNN(X_train_tensor.shape[1])
```

## Step 7. Define the CrossEntropyLoss as the loss function and Stochastic Gradient Descent (SGD) as the optimizer with learning rate 0.05.
```python
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.05)
```

## Step 8. Train the model for 20 epochs.
```python
epochs = 20
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
```

## Step 9. Evaluate the model using the test set and compute the loss.
```python
with torch.no_grad():
    y_pred_tensor = model(X_test_tensor)
    loss = criterion(y_pred_tensor, y_test_tensor)
print(f"Cross Entropy Loss: {loss.item()}")
```

## Step 10. Report the confusion matrix and corresponding accuracy, precision and recall.
```python
conf_matrix = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
print(f"Confusion Matrix:
{conf_matrix}")
print(f"Accuracy: {acc}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
```

## Step 11. Visualize the predicted vs actual values (Note: For simplicity, you can just print them side by side).
```python
print("Predicted values:", y_pred)
print("Actual values   :", y_test)
```