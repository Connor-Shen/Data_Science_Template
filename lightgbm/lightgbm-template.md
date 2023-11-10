"""
Args:
{Dataset Link} -> "str": The link to access the dataset
{Dataset Description} -> "str": Description of the dataset
{DataFrame Name} -> "str": Name of the DataFrame
{Model name} -> "str": Name of the model used in this task
{Target column name} -> "str": Name of the target column for prediction
{Test Size} -> "float": Proportion of the dataset to include in the test split, range from 0.1-0.3
{Evaluation Metric} -> "str": Evaluation metric used for validating the model
{num_leaves} -> "int": Max number of leaves, range from 2-5
{Early Stopping Rounds} -> "int": Number of early stopping rounds to avoid overfitting
{max_depth} -> "int": Max depth of decision tree, range from 20-40
{learning_rate} -> "float": Learning rate for the classifier, range from 0.01-0.3
{n_estimators} -> "int": Number of weak learners， range from 50-150
{Plot Title} -> "str": Title for the plot，can be changed according to the needs of different steps
{X-axis Label} -> "str": Label for the x-axis，can be changed according to the needs of different steps
{Y-axis Label} -> "str": Label for the y-axis，can be changed according to the needs of different steps
"""

Step 1. Import necessary libraries.
```python

```
Step 2. Load the dataset from {Dataset Link} and Convert the data into a pandas dataframe and assign it to {DataFrame Name}.
```python

```
Step 3. Split the data into features and target, then into training and testing sets with test size being {Test Size}.
```python

```
Step 4. Convert the pandas dataframes into LightGBM Dataset format.
```python

```
Step 5. Define a LightGBM model {Model name} with {max_depth}, {n_estimators}，{learning_rate} and {num_leaves}.
```python

```
Step 6. Train the model with the given parameters:{Early Stopping Rounds}, {Evaluation Metric}
```python

```
Step 7. Predict the target for the test set and Evaluate the model using the test set.
```python

```
Step 8. Report the confusion matrix and corresponding accuracy, precision, and recall.
```python

```
Step 9. Use a histogram to show the feature importance of each feature，using {Plot Title}, {X-axis Label}, {Y-axis Label}.
```python

```
Step 10. Conduct model parameter tuning for {max_depth}, {learning_rate}, {n_estimators}, {num_leaves}. select three alternative values of each parameter and output the optimal values of the parameters.
```python

```