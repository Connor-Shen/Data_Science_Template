---
jupyter:
  title: Plotting tasks using seaborn
  dataset: auto mpg dataset
  difficulty: EASY
  module: seaborn
  idx: 3
---

### Step 1. Import necessary libraries.
```python
import pandas as pd
import seaborn as sns

```

### Step 2. Load the dataset from the URL into a pandas DataFrame named `mpg_df`.
```python
Dataset_Link = "seaborn\seaborn_dataset03.csv"
mpg_df = pd.read_csv(Dataset_Link)
```

### Step 3. Display the first 5 rows of mpg_df.
```python
print(mpg_df.head())

```

### Step 4. Create a scatter plot for 'weight' against 'mpg', using title = "Weight vs MPG", X_label = "Weight", Y_label = "MPG", figure size = (8, 6), color = "blue".
```python
sns.scatterplot(data=mpg_df, x="weight", y="mpg", color="blue").set(title="Weight vs MPG", xlabel="Weight", ylabel="MPG")

```

### Step 5. Generate a joint plot for 'horsepower' and 'acceleration', using title = "Horsepower vs Acceleration", X_label = "Horsepower", Y_label = "Acceleration", figure size = (8, 6), color = "green".
```python
sns.jointplot(data=mpg_df, x="horsepower", y="acceleration", color="green", height=6).set_axis_labels("Horsepower", "Acceleration").fig.suptitle("Horsepower vs Acceleration")

```

### Step 6. Display a regression plot for 'displacement' against 'mpg', using title = "Displacement vs MPG", X_label = "Displacement", Y_label = "MPG", figure size = (8, 6), color = "red".
```python
sns.regplot(data=mpg_df, x="displacement", y="mpg", color="red").set(title="Displacement vs MPG", xlabel="Displacement", ylabel="MPG")

```

### Step 7. Generate a pair plot for the entire dataset mpg_df, using title = "Auto MPG Dataset Pairplot", X_label = "Feature", Y_label = "Feature", figure size = (10, 10), color = "pastel".
```python
sns.pairplot(mpg_df, palette="pastel").fig.suptitle("Auto MPG Dataset Pairplot")

```

### Step 8. Visualize a box plot for 'cylinders', using title = "Box Plot of Cylinders", X_label = "Cylinders", Y_label = "Frequency", figure size = (8, 6), color = "purple".
```python
sns.boxplot(data=mpg_df, x="cylinders", color="purple").set(title="Box Plot of Cylinders", xlabel="Cylinders", ylabel="Frequency")

```

### Step 9. Generate a violin plot for 'model year', using title = "Violin Plot of Model Year", X_label = "Model Year", Y_label = "Density", figure size = (8, 6), color = "orange".
```python
sns.violinplot(data=mpg_df, x="model year", color="orange").set(title="Violin Plot of Model Year", xlabel="Model Year", ylabel="Density")

```

### Step 10. Display a histogram for 'acceleration', using title = "Histogram of Acceleration", X_label = "Acceleration", Y_label = "Frequency", figure size = (8, 6), color = "skyblue".
```python
sns.histplot(data=mpg_df, x="acceleration", color="skyblue", bins=10).set(title="Histogram of Acceleration", xlabel="Acceleration", ylabel="Frequency")

```

### Step 11. Visualize the distribution of 'horsepower', using title = "Distribution of Horsepower", X_label = "Horsepower", Y_label = "Density", figure size = (8, 6), color = "pink".
```python
sns.kdeplot(data=mpg_df, x="horsepower", color="pink", fill=True).set(title="Distribution of Horsepower", xlabel="Horsepower", ylabel="Density")

```

### Step 12. Generate a heatmap of correlations for mpg_df, using title = "Heatmap of Auto MPG Dataset Correlations", X_label = "Feature", Y_label = "Feature", figure size = (8, 8), color = "coolwarm".
```python
sns.heatmap(mpg_df.corr(), annot=True, cmap="coolwarm").set(title="Heatmap of Auto MPG Dataset Correlations")

```

### Step 13. Create a scatterplot that colors points based on the 'cylinders' column, using title = "Cylinder-based Scatterplot", X_label = "Displacement", Y_label = "MPG", figure size = (8, 6), color = "bright".
```python
sns.scatterplot(data=mpg_df, x="displacement", y="mpg", hue="cylinders", palette="bright").set(title="Cylinder-based Scatterplot", xlabel="Displacement", ylabel="MPG")

```

### Step 14. Create a scatterplot with a regression line to visualize the relationship between 'weight' and 'mpg', using title = "Weight vs MPG", X_label = "Weight", Y_label = "MPG", figure size = (8, 6), color = "plum".
```python
sns.regplot(data=mpg_df, x="weight", y="mpg", color="plum").set(title="Weight vs MPG", xlabel="Weight", ylabel="MPG")

```