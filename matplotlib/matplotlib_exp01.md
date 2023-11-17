---
jupyter:
  title: Plotting tasks using matplotlib
  dataset: auto-mpg dataset
  difficulty: EASY
  module: matplotlib
  idx: 1
---

### Step 1. Import required python packages.
 ```python
import pandas as pd
import matplotlib.pyplot as plt
```

### Step 2. Load the dataset from path into a pandas DataFrame named "auto_mpg_df".
```python
url = "matplotlib\matplotlib_dataset01.csv"
auto_mpg_df = pd.read_csv(url)
```

### Step 3. Display the first 5 rows of the DataFrame "auto_mpg_df".
```python
print(auto_mpg_df.head(5))
```

### Step 4. Create a line plot of "acceleration", using title = "Line plot of Acceleration", xlabel = "Car Index", ylabel = "Acceleration", figsize=(10,6), color='blue'.
```python
plt.figure(figsize=(10,6))
plt.plot(auto_mpg_df['acceleration'], color='blue')
plt.title("Line plot of Acceleration")
plt.xlabel("Car Index")
plt.ylabel("Acceleration")
plt.show()
```

### Step 5. Create a histogram of the "weight", using title = "Histogram of Weight", xlabel = "Weight", ylabel = "Frequency", figsize=(10,6), bins=30, color='green', alpha=0.7.
```python
plt.figure(figsize=(10,6))
plt.hist(auto_mpg_df['weight'], bins=30, color='green', alpha=0.7)
plt.title("Histogram of Weight")
plt.xlabel("Weight")
plt.ylabel("Frequency")
plt.show()
```

### Step 6. Create a bar chart using the unique values of "cylinders", using title = "Bar chart of Cylinders", xlabel = "Number of Cylinders", ylabel = "Count", figsize=(10,6), color='purple', alpha=0.7
```python
bar_data = auto_mpg_df['cylinders'].value_counts()
bar_data.plot(kind='bar', color='purple', alpha=0.7)
plt.title("Bar chart of Cylinders")
plt.xlabel("Number of Cylinders")
plt.ylabel("Count")
plt.show()
```

### Step 7. Create a scatter plot of "mpg" vs "displacement", using title = "Scatter plot of MPG vs Displacement", xlabel = "Displacement", ylabel = "MPG", figsize=(10,6).
```python
plt.figure(figsize=(10,6))
plt.scatter(auto_mpg_df["displacement"], auto_mpg_df["mpg"], color='red')
plt.title("Scatter plot of MPG vs Displacement")
plt.xlabel("Displacement")
plt.ylabel("MPG")
plt.show()
```

### Step 8. Create a box plot for "horsepower", using title = "Box plot of Horsepower", ylabel = "Horsepower", figsize=(10,6).
```python

plt.figure(figsize=(10,6))
plt.boxplot(auto_mpg_df['horsepower'])
plt.title("Box plot of Horsepower")
plt.ylabel("Horsepower")
plt.show()
```

### Step 9. Create a pie chart of the unique values of "origin", using title = "Pie chart of Origin", figsize=(8,8).
```python
pie_data = auto_mpg_df['origin'].value_counts()
plt.figure(figsize=(8,8))
plt.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%')
plt.title("Pie chart of Origin")
plt.show()
```

### Step 10. Create a stacked bar chart using "mpg" and "displacement", using title = "Stacked bar chart of MPG and Displacement", xlabel = "Car Index", ylabel = "Value", figsize=(10,6).
```python
stacked_data = auto_mpg_df[['mpg', 'displacement']]
stacked_data.plot(kind='bar', stacked=True, figsize=(10,6))
plt.title("Stacked bar chart of MPG and Displacement")
plt.xlabel("Car Index")
plt.ylabel("Value")
plt.show()
```

### Step 11. Create an area plot for "mpg" and "weight", using title = "Area plot of MPG and Weight", xlabel = "Car Index", ylabel = "Value", figsize=(10,6), alpha=0.4.
```python
auto_mpg_df[["mpg", "weight"]].plot(kind='area', alpha=0.4, figsize=(10,6))
plt.title("Area plot of MPG and Weight")
plt.xlabel("Car Index")
plt.ylabel("Value")
plt.show()
```

### Step 12. Create a step plot for "acceleration", using title = "Step plot of Acceleration", xlabel = "Car Index", ylabel = "Acceleration", figsize=(10,6), color='cyan'.### 
```python
plt.figure(figsize=(10,6))
plt.step(auto_mpg_df.index, auto_mpg_df['acceleration'], color='cyan')
plt.title("Step plot of Acceleration")
plt.xlabel### ("Car Index")
plt.ylabel("Acceleration")
plt.show()
```

### Step 13. Create two subplots. The first subplot should plot the values of "mpg" on a linear scale and the second subplot should plot the values of "mpg" on a logarithmic scale. Use title = "Linear scale plot of MPG" for the first subplot and title = "Logarithmic scale plot of MPG" for the second subplot, xlabel = "Car Index", ylabel = "MPG", figsize=(10,10).
```python
fig, axs = plt.subplots(2, 1, figsize=(10,10))
axs[0].plot(auto_mpg_df["mpg"], color='blue')
axs[0].set_title("Linear scale plot of MPG")
axs[0].set_xlabel("Car Index")
axs[0].set_ylabel("MPG")
axs[1].plot(auto_mpg_df["mpg"], color='green')
axs[1].set_yscale("log")
axs[1].set_title("Logarithmic scale plot of MPG")
axs[1].set_xlabel("Car Index")
axs[1].set_ylabel("MPG (Log scale)")
plt.tight_layout()
plt.show()
```