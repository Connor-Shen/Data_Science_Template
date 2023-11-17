## PyOD Library Template
“”“
Args:
{Dataset Path} -> "str": The path to the dataset.
{DataFrame Name} -> "str": Name of the DataFrame to be used.
{Feature Columns} -> "list[str]": List of feature column names used for outlier detection.
{Test Size} -> "float": The proportion of the dataset to include in the test split.
{Random State} -> "int": A seed for random number generation to ensure reproducibility.
{Detectors Dictionary} -> "dictionary": List of PyOD outlier detection models to be explored.
{Outliers Fraction} -> "float": The fraction of the Outliers
{Selected column name} -> "str": Name of the selected column，can be changed according to the needs of different steps
”“”

### Step 1. Import required python packages.
```python

```

### Step 2. Load the dataset from the {Dataset Path} into a pandas DataFrame named {DataFrame Name}.
```python

```

### Step 3. Normalize the data and store the values in a NumPy array for later use in our model
```python

```

### Step 4. Choose {Selected column name} and {Selected column name} for outlier detection. Split the two columns' data into features and training/test sets as {Test Size}. 
```python

```

### Step 5. Randomly choose three detectors and create a classifiers dictionary {Detectors Dictionary}, using {Outliers Fraction}, {Random State}
```python

```

### Step 6. Initialize and fit the three detectors from {Detectors Dictionary} respectively.
```python

```

### Step 7. Predict outliers on the test set using the three detectors respectively.
```python

```

### Step 8. Evaluate the prediction using ROC and Precision on training data and test data
```python

```

### Step 9. Visualize the outlier scores by visualize function for the three detectors respectively.
```python

```