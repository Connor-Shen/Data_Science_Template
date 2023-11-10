---
jupyter:
  title: Classification tasks in pytorch
  dataset: Sentiment140 Dataset
  difficulty: Hard
  model: LSTM
  module: pytorch
  idx: 8
---
# Steps and Python Code for Sentiment Analysis using Sentiment140 Dataset

## Step 1. Import necessary libraries.

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
```

## Step 2. Load the dataset.

You can download the dataset from its hosting URL:

Dataset Link: [Sentiment140 Dataset Download](http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip)

```python
# After downloading and unzipping, load the dataset
tweets_df = pd.read_csv('pytorch_dataset08.csv', encoding='latin1', usecols=[0,5], names=['sentiment', 'text'])
```

## Step 3. Preprocess the data and prepare it for training.

```python
# Convert sentiments to binary
tweets_df['sentiment'] = tweets_df['sentiment'].replace(4, 1)

# Preprocess the tweets text - this would include cleaning and tokenization
# For demonstration, we'll assume there's a function called 'preprocess_text' available
tweets_df['text'] = tweets_df['text'].apply(preprocess_text)

# Encode the text using a tokenizer - this typically involves converting text to sequences of integers
# For demonstration, we'll assume there's a function called 'tokenize' available
tweets_df['text'] = tokenize(tweets_df['text'])
```

## Step 4. Split the dataset into training and testing sets.

```python
X_train, X_test, y_train, y_test = train_test_split(tweets_df['text'], tweets_df['sentiment'], test_size=0.2, random_state=42)
```

## Step 5. Convert the data into PyTorch DataLoader for batch processing.

```python
train_data = TensorDataset(torch.from_numpy(X_train.values), torch.from_numpy(y_train.values))
test_data = TensorDataset(torch.from_numpy(X_test.values), torch.from_numpy(y_test.values))

batch_size = 50
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)
```

## Step 6. Define a "SentimentAnalysisLSTM" model with appropriate layers for NLP.

```python
class SentimentAnalysisLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, drop_prob=0.5):
        super(SentimentAnalysisLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=drop_prob, batch_first=True)
        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, hidden = self.lstm(embedded)
        out = self.dropout(lstm_out)
        out = self.fc(out[:, -1])
        return self.sigmoid(out)

# We need to define vocab_size and other hyperparameters based on the dataset
model = SentimentAnalysisLSTM(vocab_size, embedding_dim=400, hidden_dim=256, output_dim=1, n_layers=2)
```

## Step 7. Define the BCELoss loss function and the Adam optimizer.

```python
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

## Step 8. Train the model for 20 epochs.

```python
epochs = 20
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), labels.float())
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f'Epoch {epoch+1}/{epochs} - Loss: {running_loss/len(train_loader)}')
```

## Step 9. Evaluate the model performance on the test set.

```python
model.eval()
predictions, truths = [], []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        outputs = model(inputs)
        predictions.append(outputs.squeeze())
        truths.append(labels)
        
predictions = torch.cat(predictions).cpu().numpy()
truths = torch.cat(truths).cpu().numpy()
predictions = np.round(predictions)
```
 
## Step 10. Report the model's performance metrics, such as accuracy.

```python
from sklearn.metrics import accuracy_score, classification_report
accuracy = accuracy_score(truths, predictions)
report = classification_report(truths, predictions)
print(f'Accuracy: {accuracy}')
print(report)
```

## Step 11. Visualize some predictions.

```python
import matplotlib.pyplot as plt

def plot_predictions(truths, predictions, n=20):
    plt.figure(figsize=(10, 5))
    plt.plot(truths[:n], label='True labels')
    plt.plot(predictions[:n], label='Predicted labels')
    plt.xlabel('Sample index')
    plt.ylabel('Sentiment')
    plt.legend()
    plt.show()

plot_predictions(truths, predictions)
```
