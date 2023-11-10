---
jupyter:
  title: Classification tasks in pytorch
  dataset: iris
  difficulty: EASY
  model: linear regression
  module: pytorch
  idx: 1
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

## Step 2. Load the dataset from "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data".
```python
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
column_names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
iris_df = pd.read_csv(url, names=column_names)
```

## Step 3. Convert the data into a pandas dataframe and add the target variable to the dataframe named "iris_df".
```python
# The dataset is already in a pandas dataframe, so we'll convert the 'class' column to numerical values
class_mapping = {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}
iris_df["class"] = iris_df["class"].map(class_mapping)
```

## Step 4. Normalize the data using Min-Max Scaling and split it into training and testing sets with test size being 0.15.
```python
X = iris_df.drop("class", axis=1).values
y = iris_df["class"].values
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

## Step 6. Define a simple linear regression model with appropriate inputs and 1 output.
```python
class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 10)
        self.fc2 = nn.Linear(10, 3)  # 3 classes in the output

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
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
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