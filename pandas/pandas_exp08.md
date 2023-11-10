---
jupyter:
  title: Data set preprocessing using pandas in Python
  dataset: 携程订单数据
  difficulty: Hard
  module: pandas
  idx: 8
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
(Note: You didn't provide a specific URL for this dataset, so I'm using a placeholder. Please replace it with the actual URL.)
```python
URL = "YOUR_DATASET_URL_HERE"
df = pd.read_csv("pandas\pandas_dataset08.csv")
```

### Step 3. Calculate and display the number of missing values in each column of the DataFrame "df".
```python
missing_values = df.isnull().sum()
print(missing_values)
```

### Step 4. Replace missing values in the "starprefer" column with the median value of the same column. Make sure to apply this change to the DataFrame "df" itself.
```python
df['starprefer'].fillna(df['starprefer'].median(), inplace=True)
```

### Step 5. Remove any remaining rows with missing values from the DataFrame "ordercanncelednum". Ensure that changes are made inplace.
```python
df.dropna(subset=['ordercanncelednum'], inplace=True)
```

### Step 6. Perform square root transformation to "avgprice" and give the data distribution of the column.
```python
df['avgprice'] = np.sqrt(df['avgprice'])
plt.hist(df['avgprice'], bins=30)
plt.xlabel('Square Root Transformed avgprice')
plt.ylabel('Frequency')
plt.show()
```

### Step 7. Detect whether there are outliers in "landhalfhours", set Outlier_degree = 2.5.
```python
Q1 = df['landhalfhours'].quantile(0.25)
Q3 = df['landhalfhours'].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df['landhalfhours'] < Q1 - 2.5*IQR) | (df['landhalfhours'] > Q3 + 2.5*IQR)]
print(outliers)
```

### Step 8. Visualize Outliers Using Box Plots for "landhalfhours", using Plot_Title = "Boxplot for landhalfhours", X_axis_Label = "landhalfhours", Y_axis_Label = "Frequency", Figure_size = (10, 6), Color = "green".
```python
plt.figure(figsize=(10, 6))
sns.boxplot(y=df['landhalfhours'], color="green")
plt.title("Boxplot for landhalfhours")
plt.xlabel("landhalfhours")
plt.ylabel("Frequency")
plt.show()
```

### Step 9. Convert values in the "label" column, 1 to "ordered" and 0 to "not ordered".
```python
df['label'] = df['label'].replace({1: "ordered", 0: "not ordered"})
```

### Step 10. Group and Aggregate Data by "weekday" and calculate the min, max, average of each numerical group.
```python
grouped = df.groupby('weekday').agg(['min', 'max', 'mean'])
print(grouped)
```

### Step 11. Display the correlation matrix of "avgprice", "lowestprice", and "customereval_pre2".
```python
correlation_matrix = df[['avgprice', 'lowestprice', 'customereval_pre2']].corr()
print(correlation_matrix)
```

### Step 12. Plot a heatmap of the correlation matrix calculated in Step 11. Display the heatmap with annotations for each cell.
```python
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.show()
```
