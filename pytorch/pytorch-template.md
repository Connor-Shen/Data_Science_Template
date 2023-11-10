"""
Args:
{Dataset Link} -> "str": The link to access the dataset
{Dataset Description} -> "str": Description of the dataset
{DataFrame Name} -> "str": Name of the DataFrame
{Model name} -> "str": Name of the model used in this task
{Selected column name} -> "str": Name of the selected column，can be changed according to the needs of different steps
{Test Size} -> "float": Proportion of the dataset to include in the test split, range from 0.1-0.2
{Epochs} -> "int": Number of epochs for training, range from 10-30
{Learning Rate} -> "float": Learning rate for the optimizer, range from 0.01-0.3
{Loss function} -> "str": The loss function of model.
{Optimizer} -> "str": The optimizer of model.
"""

Step 1. Import necessary libraries.
```python

```

Step 2. Load the dataset from {Dataset Link}.
```python

```

Step 3. Convert the data into a pandas dataframe and add the target variable to the dataframe named {DataFrame Name}.
```python

```

Step 4. Normalize the data of {Selected column name} using Min-Max Scaling and split it into training and testing sets with test size being {Test Size}.
```python

```

Step 5. Convert the numpy arrays into PyTorch tensors.
```python

```

Step 6. Define a {Model name} with appropriate inputs and 1 output(label).
```python

```

Step 7. Define {Loss function} as the loss function and {Optimizer} as the optimizer with learning rate {Learning Rate}.
```python

```

Step 8. Train the model for {Epochs} epochs.
```python

```

Step 9. Evaluate the model using the test set and compute the {Loss function}.
```python

```

Step 10. Report the confusion matrix and corresponding accuracy, precision and recall.
```python

```

Step 11. Visualize the predicted vs actual values.
```python

```