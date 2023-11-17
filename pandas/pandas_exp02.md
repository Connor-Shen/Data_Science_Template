---
jupyter:
  title: Data set preprocessing using pandas in Python
  dataset: Auto MPG
  difficulty: EASY
  module: pandas
  idx: 2
---


### Step 1. Import required python packages.
```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import numpy as np
```

### Step 2. Load the dataset from the URL into a pandas DataFrame named `mpg_df`.
```python
Dataset_Link = "pandas\pandas_dataset02.csv"
mpg_df = pd.read_csv(Dataset_Link, sep=',')
```

### Step 3. Calculate and display the number of missing values in each column of the DataFrame `mpg_df`.
```python
mpg_df.replace('?', np.nan, inplace=True)
missing_values = mpg_df.isnull().sum()
print(missing_values)
```

### Step 4. Replace missing values in the `horsepower` column with the median value of the same column.
```python
mpg_df['horsepower'] = mpg_df['horsepower'].astype(float)
mpg_df['horsepower'].fillna(mpg_df['horsepower'].median(), inplace=True)
```

### Step 5. Remove any remaining rows with missing values from the DataFrame `mpg_df`.
```python
mpg_df.dropna(inplace=True)
```

### Step 6. Perform log transformation to `weight` and give the data distribution of the column.
```python
mpg_df['weight'] = np.log(mpg_df['weight'])
plt.hist(mpg_df['weight'], bins=30)
plt.xlabel('Log Transformed Weight')
plt.ylabel('Frequency')
plt.show()
```

### Step 7. Detect whether there are outliers in `displacement`， set Outlier_degree = 2.5.
```python
Q1 = mpg_df['displacement'].quantile(0.25)
Q3 = mpg_df['displacement'].quantile(0.75)
IQR = Q3 - Q1
outliers = mpg_df[(mpg_df['displacement'] < Q1 - 2.5*IQR) | (mpg_df['displacement'] > Q3 + 2.5*IQR)]
print(outliers)
```

### Step 8. Visualize Outliers Using Box Plots for `cylinders`, using Plot_Title = "Boxplot for Number of Cylinders in Cars", X_axis_Label = "Number of Cylinders", Y_axis_Label = "Frequency", Figure_size = (10, 6), Color = "red".
```python
plt.figure(figsize=(10, 6))
sns.boxplot(y=mpg_df['cylinders'], color="red")
plt.title("Boxplot for Number of Cylinders in Cars")
plt.xlabel("Number of Cylinders")
plt.ylabel("Frequency")
plt.show()
```

### Step 9. Convert values in the `car name` column to numerical values using label encoding.
```python
encoder = LabelEncoder()
mpg_df['car name'] = encoder.fit_transform(mpg_df['car name'])
```

### Step 10. Group and Aggregate Data by `origin` and calculate the min, max, average of each group.
```python
grouped = mpg_df.groupby('origin').agg(['min', 'max', 'mean'])
print(grouped)
```

### Step 11. Display the correlation matrix of three related columns: `mpg`, `horsepower`, and `weight`.
```python
correlation_matrix = mpg_df[['mpg', 'horsepower', 'weight']].corr()
print(correlation_matrix)
```

### Step 12. Plot a heatmap of the correlation matrix calculated in Step 11.
```python
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.show()
```
