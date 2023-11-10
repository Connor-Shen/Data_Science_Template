---
jupyter:
  title: Classification tasks in pytorch
  dataset: COVIDx CXR-2
  difficulty: Hard
  model: CNN
  module: pytorch
  idx: 10
---
# COVID-19 Image Data Collection - Model Training with PyTorch

## Step 1. Import necessary libraries.
```python
import os
import pandas as pd
import numpy as np
import torch
from torch import nn, optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
```

## Step 2. Load the dataset from the Dataset Link.
```python
# The dataset can be downloaded from its official GitHub repository or a similar source
dataset_path = 'path_to_covid_dataset'  # Replace with the actual path
url = "https://paperswithcode.com/dataset/covidx"
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])
dataset = datasets.ImageFolder(root=dataset_path, transform=transform)

```

## Step 3. Convert the data into a pandas dataframe and add the target variable to the dataframe named "covid_df".
```python
covid_df = pd.DataFrame(data)
```

## Step 4. Normalize the data of Selected column name using Min-Max Scaling and split it into training and testing sets with test size being "0.2".
```python
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
```

## Step 5. Convert the numpy arrays into PyTorch tensors.
```python
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
```

## Step 6. Define a Model name with appropriate inputs and 1 output(label).
```python
class CovidCNN(nn.Module):
    def __init__(self):
        super(CovidCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 56 * 56, 512)
        self.fc2 = nn.Linear(512, 2)  # Assuming binary classification (COVID vs. No-COVID)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 56 * 56)  # Flatten the tensor
        x = torch.relu(self.fc1(self.dropout(x)))
        x = self.fc2(self.dropout(x))
        return x

model = CovidCNN()

```

## Step 7. Define CrossEntropyLoss as the loss function and Optimizer as the optimizer with learning rate "0.001".
```python
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

```

## Step 8. Train the model for "10" epochs.
```python
epochs = 10
for epoch in range(epochs):
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch {epoch + 1}/{epochs} - Loss: {running_loss / len(train_loader)}')

```

## Step 9. Evaluate the model using the test set and compute the Loss function.
```python
model.eval()
all_predictions = []
all_targets = []
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        all_predictions.extend(predicted.numpy())
        all_targets.extend(labels.numpy())
accuracy = accuracy_score(all_targets, all_predictions)
print(f'Accuracy: {accuracy}')
```

## Step 10. Report the confusion matrix and corresponding accuracy, precision, and recall.
```python
cm = confusion_matrix(all_targets, all_predictions)
precision = precision_score(all_targets, all_predictions)
recall = recall_score(all_targets, all_predictions)
print(f'Confusion Matrix:\n{cm}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
```

## Step 11. Visualize the predicted vs actual values.
```python
def visualize_predictions(dataset, all_predictions, num_images=10):
    fig = plt.figure(figsize=(25, 4))
    for idx in np.arange(num_images):
        ax = fig.add_subplot(2, num_images//2, idx+1, xticks=[], yticks=[])
        plt.imshow(np.transpose(dataset[idx][0].numpy(), (1, 2, 0)))
        ax.set_title(f"Predicted Class: {all_predictions[idx]}")
visualize_predictions(test_dataset, all_predictions)
```
