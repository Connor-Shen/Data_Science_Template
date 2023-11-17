---
jupyter:
  title: Data set preprocessing using pandas in Python
  dataset: Iris
  difficulty: EASY
  module: pandas
  idx: 3
---

### Step 1. Import required python packages.
```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import numpy as np
```

### Step 2. Load the dataset from the URL into a pandas DataFrame named "df".
```python
URL = "pandas\pandas_dataset03.csv"
df = pd.read_csv(URL, sep=',')
```


### Step 3. Calculate and display the number of missing values in each column of the DataFrame "df".
```python
missing_values = df.isnull().sum()
print(missing_values)
```

### Step 4. Replace missing values in the "petal_width" column with the median value of the same column. Make sure to apply this change to the DataFrame "df" itself.
```python
df['petal_width'].fillna(df['petal_width'].median(), inplace=True)
```

### Step 5. Remove any remaining rows with missing values from the DataFrame "df". Ensure that changes are made inplace.
```python
df.dropna(inplace=True)
```

### Step 6. Perform log transformation to "petal_length" and give the data distribution of the column.
```python
df['petal_length'] = np.log(df['petal_length'])
plt.hist(df['petal_length'], bins=30)
plt.xlabel('Log Transformed Petal Length')
plt.ylabel('Frequency')
plt.show()
```

### Step 7. Detect whether there are outliers in "sepal_width", set Outlier_degree = 2.5.
```python
Q1 = df['sepal_width'].quantile(0.25)
Q3 = df['sepal_width'].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df['sepal_width'] < Q1 - 2.5*IQR) | (df['sepal_width'] > Q3 + 2.5*IQR)]
print(outliers)
```

### Step 8. Visualize Outliers Using Box Plots for "sepal_width" column with the title "Sepal Width Outliers", x-axis label "Species", y-axis label "Sepal Width (cm)", figure size of (10, 5), and color set to "red".
```python
plt.figure(figsize=(10, 5))
sns.boxplot(x=df['species'], y=df['sepal_width'], color="red")
plt.title("Sepal Width Outliers")
plt.xlabel("Species")
plt.ylabel("Sepal Width (cm)")
plt.show()
```

### Step 9. Convert values in the "species" column to numerical values using label encoding.
```python
encoder = LabelEncoder()
df['species'] = encoder.fit_transform(df['species'])
```

### Step 10. Group and Aggregate Data by "species" and calculate the min, max average of each group.
```python
grouped = df.groupby('species').agg(['min', 'max', 'mean'])
print(grouped)
```

### Step 11. Display the correlation matrix of "sepal_length", "petal_length", and "petal_width".
```python
correlation_matrix = df[['sepal_length', 'petal_length', 'petal_width']].corr()
print(correlation_matrix)
```

### Step 12. Plot a heatmap of the correlation matrix calculated in Step 11. Display the heatmap with annotations for each cell.
```python
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.show()
```
