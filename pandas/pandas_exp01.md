---
jupyter:
  title: Data set preprocessing using pandas in Python
  dataset: Wine Quality
  difficulty: EASY
  module: pandas
  idx: 1
---

### Step 1. Import required python packages.
```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```

### Step 2. Load the dataset from the path into a pandas DataFrame named `wine_df`.
```python
path = "pandas\pandas_dataset01.csv"
wine_df = pd.read_csv(path, sep=';')
```

### Step 3. Calculate and display the number of missing values in each column of the DataFrame `wine_df`.
```python
missing_values = wine_df.isnull().sum()
print(missing_values)
```

### Step 4. Replace missing values in the `pH` column with the median value of the same column.
```python
wine_df['pH'].fillna(wine_df['pH'].median(), inplace=True)
```

### Step 5. Remove any remaining rows with missing values from the DataFrame `wine_df`.
```python
wine_df.dropna(inplace=True)
```

### Step 6. Perform square root transformation to `alcohol` and give the data distribution of the column.
```python
wine_df['alcohol'] = wine_df['alcohol']**0.5
plt.hist(wine_df['alcohol'], bins=30)
plt.xlabel('Square Root Transformed Alcohol')
plt.ylabel('Frequency')
plt.show()
```

### Step 7. Detect whether there are outliers in `fixed acidity`, set Outlier_degree = 3.
```python
Q1 = wine_df['fixed acidity'].quantile(0.25)
Q3 = wine_df['fixed acidity'].quantile(0.75)
IQR = Q3 - Q1
outliers = wine_df[(wine_df['fixed acidity'] < Q1 - 3*IQR) | (wine_df['fixed acidity'] > Q3 + 3*IQR)]
print(outliers)
```

### Step 8. Visualize Outliers Using Box Plots for `volatile acidity`, using Plot_Title = "Boxplot for Volatile Acidity in Wines", X_axis_Label = "Volatile Acidity", Y_axis_Label = "Frequency", Figure_size = (10, 6), Color = "green".
```python
plt.figure(figsize=(10, 6))
sns.boxplot(y=wine_df['volatile acidity'], color="green")
plt.title("Boxplot for Volatile Acidity in Wines")
plt.xlabel("Volatile Acidity")
plt.ylabel("Frequency")
plt.show()
```

### Step 9. Convert values in the `quality` column to numerical values using label encoding.
```python
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
wine_df['quality'] = encoder.fit_transform(wine_df['quality'])
```

### Step 10. Group and Aggregate Data by `quality` and calculate the min, max, average of each group.
```python
grouped = wine_df.groupby('quality').agg(['min', 'max', 'mean'])
print(grouped)
```

### Step 11. Display the correlation matrix of three related columns: `alcohol`, `pH`, and `fixed acidity`.
```python
correlation_matrix = wine_df[['alcohol', 'pH', 'fixed acidity']].corr()
print(correlation_matrix)
```

### Step 12. Plot a heatmap of the correlation matrix calculated in Step 11.
```python
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.show()
```
