
### Step 1. Import Required Python Packages
```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

```

### Step 2. Load the Dataset into a pandas DataFrame named 'cancer_df'.
```python
cancer_df = pd.read_csv('path_to_dataset.csv')  # Replace 'path_to_dataset.csv' with the actual file path

```

### Step 3. Display the First 5 Rows of the DataFrame 'cancer_df'
```python
print(cancer_df.head())

```

### Step 4. Create a line plot of the 'mean radius' column, using title = "Line Plot of Mean Radius", xlabel = "Sample Index", ylabel = "Mean Radius", figsize=(10,6), color='blue'.
```python
plt.figure(figsize=(10, 6))
plt.plot(cancer_df['mean radius'], color='blue')
plt.title("Line Plot of Mean Radius")
plt.xlabel("Sample Index")
plt.ylabel("Mean Radius")
plt.show()

```

### Step 5. Create a histogram of the 'mean texture' column, using title = "Histogram of Mean Texture", xlabel = "Mean Texture", ylabel = "Frequency", figsize=(10,6), bins=30, color='green', alpha=0.7.
```python
plt.figure(figsize=(10, 6))
plt.hist(cancer_df['mean texture'], bins=30, color='green', alpha=0.7)
plt.title("Histogram of Mean Texture")
plt.xlabel("Mean Texture")
plt.ylabel("Frequency")
plt.show()

```

### Step 6. Create a bar chart using the unique values of the 'diagnosis' column, using title = "Bar Chart of Diagnosis", xlabel = "Diagnosis", ylabel = "Count", figsize=(10,6), color='purple', alpha=0.7.
```python
diagnosis_counts = cancer_df['diagnosis'].value_counts()
plt.figure(figsize=(10, 6))
diagnosis_counts.plot(kind='bar', color='purple', alpha=0.7)
plt.title("Bar Chart of Diagnosis")
plt.xlabel("Diagnosis")
plt.ylabel("Count")
plt.show()

```

### Step 7. Create a scatter plot of 'mean area' vs 'mean smoothness', using title = "Scatter Plot of Mean Area vs Mean Smoothness", xlabel = "Mean Area", ylabel = "Mean Smoothness", figsize=(10,6).
```python
plt.figure(figsize=(10, 6))
plt.scatter(cancer_df['mean area'], cancer_df['mean smoothness'])
plt.title("Scatter Plot of Mean Area vs Mean Smoothness")
plt.xlabel("Mean Area")
plt.ylabel("Mean Smoothness")
plt.show()

```

### Step 8. Create a box plot for the 'mean perimeter' column, using title = "Box Plot of Mean Perimeter", xlabel = "Patients", ylabel = "Mean Perimeter", figsize=(10,6).
```python
plt.figure(figsize=(10, 6))
sns.boxplot(y=cancer_df['mean perimeter'])
plt.title("Box Plot of Mean Perimeter")
plt.xlabel("Patients")
plt.ylabel("Mean Perimeter")
plt.show()

```

### Step 9. Create a pie chart of the unique values of the 'diagnosis' column, using title = "Pie Chart of Diagnosis Categories", figsize=(8,8).
```python
diagnosis_counts = cancer_df['diagnosis'].value_counts()
plt.figure(figsize=(8, 8))
diagnosis_counts.plot(kind='pie', autopct='%1.1f%%')
plt.title("Pie Chart of Diagnosis Categories")
plt.ylabel('')  # Hide the y-label
plt.show()

```

### Step 10. Create a stacked bar chart using 'mean radius' and 'mean texture', using title = "Stacked Bar Chart of Mean Radius and Mean Texture", xlabel = "Sample Index", ylabel = "Values", figsize=(10,6).
```python
plt.figure(figsize=(10, 6))
cancer_df[['mean radius', 'mean texture']].plot(kind='bar', stacked=True)
plt.title("Stacked Bar Chart of Mean Radius and Mean Texture")
plt.xlabel("Sample Index")
plt.ylabel("Values")
plt.show()

```

### Step 11. Create an area plot for 'mean compactness' and 'mean concavity', using title = "Area Plot of Mean Compactness and Mean Concavity", xlabel = "Sample Index", ylabel = "Values", figsize=(10,6), alpha=0.4.
```python
plt.figure(figsize=(10, 6))
cancer_df[['mean compactness', 'mean concavity']].plot(kind='area', alpha=0.4)
plt.title("Area Plot of Mean Compactness and Mean Concavity")
plt.xlabel("Sample Index")
plt.ylabel("Values")
plt.show()

```

### Step 12. Create a step plot for the 'mean symmetry' column, using title = "Step Plot of Mean Symmetry", xlabel = "Sample Index", ylabel = "Mean Symmetry", figsize=(10,6), color='cyan'.
```python
plt.figure(figsize=(10, 6))
plt.step(cancer_df.index, cancer_df['mean symmetry'], color='cyan')
plt.title("Step Plot of Mean Symmetry")
plt.xlabel("Sample Index")
plt.ylabel("Mean Symmetry")
plt.show()
```

### Step 13. Create two subplots. The first subplot should plot the values of 'mean fractal dimension' on a linear scale and the second subplot should plot the values of 'mean fractal dimension' on a logarithmic scale, using title = "Linear Scale Plot of Mean Fractal Dimension" for the first subplot and "Logarithmic Scale Plot of Mean Fractal Dimension" for the second subplot, xlabel = "Sample Index", ylabel = "Mean Fractal Dimension", figsize=(10,10).
```python
fig, axs = plt.subplots(2, figsize=(10, 10))

# Linear Scale Plot
axs[0].plot(cancer_df['mean fractal dimension'])
axs[0].set_title("Linear Scale Plot of Mean Fractal Dimension")
axs[0].set_xlabel("Sample Index")
axs[0].set_ylabel("Mean Fractal Dimension")

# Logarithmic Scale Plot
axs[1].plot(cancer_df['mean fractal dimension'])
axs[1].set_yscale('log')
axs[1].set_title("Logarithmic Scale Plot of Mean Fractal Dimension")
axs[1].set_xlabel("Sample Index")
axs[1].set_ylabel("Mean Fractal Dimension")

plt.tight_layout()
plt.show()
```


