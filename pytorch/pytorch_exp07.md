---
jupyter:
  title: Classification tasks in pytorch
  dataset: Amazon Product Reviews Dataset
  difficulty: Hard
  model: LSTM
  module: pytorch
  idx: 7
---

# Steps and Python Code for Sentiment Analysis using Amazon Product Reviews Dataset

## Step 1. Import necessary libraries.

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
```

## Step 2. Load the dataset.

You can download the dataset from the following URL:

Dataset Link: [Amazon Product Reviews Dataset Download](https://nijianmo.github.io/amazon/index.html)

```python
# Load a sample dataset for demonstration purposes
reviews_df = pd.read_json('pytorch_dataset07.json.gz', lines=True)
```

## Step 3. Convert the data into a pandas dataframe and name it `ProductReviewsDF`.

```python
ProductReviewsDF = reviews_df[['reviewText', 'overall']]  # 'reviewText' contains the text, 'overall' contains the ratings
```

## Step 4. Preprocess the data by converting ratings to sentiment labels and split the data into training and testing sets.

```python
# Convert ratings to binary sentiment labels, where ratings <= 2 are negative (0) and > 2 are positive (1)
ProductReviewsDF['sentiment'] = np.where(ProductReviewsDF['overall'] > 2, 1, 0)

TestSize = 0.2
X_train, X_test, y_train, y_test = train_test_split(ProductReviewsDF['reviewText'], ProductReviewsDF['sentiment'], test_size=TestSize)
```

## Step 5. Prepare the data for training by tokenizing the text and converting it into PyTorch tensors.

```python
# Assume 'tokenize_and_pad' is a function that tokenizes and pads the text data
X_train_tokens = tokenize_and_pad(X_train)
X_test_tokens = tokenize_and_pad(X_test)

# Convert the tokenized reviews to tensors
train_tensors = torch.tensor(X_train_tokens)
test_tensors = torch.tensor(X_test_tokens)
train_labels = torch.tensor(y_train.values)
test_labels = torch.tensor(y_test.values)
```


## Step 6. Define a neural network model for sentiment classification.

```python
class SentimentClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(SentimentClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (ht, ct) = self.lstm(embedded)
        out = self.fc(ht[-1])
        return self.sigmoid(out)

# Instantiate the model with the appropriate vocabulary size and dimensions
vocab_size = 10000  # This is just an example value
model = SentimentClassifier(vocab_size, embedding_dim=400, hidden_dim=256)
```

## Step 7. Define the loss function and optimizer.

```python
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
```

## Step 8. Train the model.

```python
Epochs = 10

for epoch in range(Epochs):
    model.train()
    running_loss = 0.0
    
    # Assuming 'train_loader' is a DataLoader object that loads batches from 'train_tensors' and 'train_labels'
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), labels.float())
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f'Epoch {epoch+1}/{Epochs} - Loss: {running_loss/len(train_loader)}')
```

## Step 9. Evaluate the model using the test set and compute the loss.

```python
model.eval()
test_loss = 0.0
correct_predictions = 0

# Assuming 'test_loader' is a DataLoader object that loads batches from 'test_tensors' and 'test_labels'
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), labels.float())
        test_loss += loss.item()
        predicted = outputs.round()
        correct_predictions += (predicted.squeeze() == labels).sum().item()

average_loss = test_loss / len(test_loader)
accuracy = correct_predictions / len(test_tensors)
print(f'Loss: {average_loss}')
print(f'Accuracy: {accuracy}')
```

### Step 10. Report the confusion matrix and corresponding accuracy, precision, and recall.

```python
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

# Generate predictions
predictions = []
with torch.no_grad():
    for inputs in test_loader:
        outputs = model(inputs[0])
        predictions.append(outputs.round().squeeze())

predictions = torch.cat(predictions).numpy()
truths = test_labels.numpy()

# Calculate metrics
cm = confusion_matrix(truths, predictions)
acc = accuracy_score(truths, predictions)
precision = precision_score(truths, predictions)
recall = recall_score(truths, predictions)

print(f'Confusion Matrix:\n{cm}')
print(f'Accuracy: {acc}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
```

Step 11. Visualize the predicted vs actual values.

```python
import matplotlib.pyplot as plt

def plot_predictions(truths, predictions):
    plt.figure(figsize=(10, 5))
    plt.plot(truths, 'o', label='True values')
    plt.plot(predictions, '.', label='Predicted values')
    plt.title('Predicted vs Actual Sentiments')
    plt.xlabel('Sample index')
    plt.ylabel('Sentiment')
    plt.legend()
    plt.show()

plot_predictions(truths, predictions)
```
