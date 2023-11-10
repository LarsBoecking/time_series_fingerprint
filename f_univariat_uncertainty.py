# %%
# import required functions and classes
import os

import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
import seaborn as sns
from src.utils_performance import (
    _all_algorithms_all_datasets_performance,
)

from src.utils_visualization import NotebookFigureSaver
from src.utils_data_centric import _get_all_data_set_characteristics

# %%
# get the characteristic statistics on all data sets
normalized_data_set_characteristics = _get_all_data_set_characteristics(
    multivariate=False, number_data_sets=50
)
X = pd.DataFrame(normalized_data_set_characteristics)

# get the mean performance of all algorithms on all data sets
algorithm_data_set_performance = _all_algorithms_all_datasets_performance(
    performance_of_interest="$\\hat{\\mu}$", multivariate=False
)
Y = pd.DataFrame(algorithm_data_set_performance).T

# %%
X=X.drop(columns=["dim_count"])



# %%
# inner join x and y on index
matched_data_sets = X.join(Y, how="inner")
# document how many rows were not matched
num_rows_not_matched = len(X) - len(matched_data_sets)
print(f"Number of rows not matched: {num_rows_not_matched} from total of {len(X)}")

# %%
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


for target_algorithm in Y.columns:
    print(f"\n Predicting {target_algorithm}")
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        matched_data_sets[X.columns],
        matched_data_sets[target_algorithm],     
        test_size=0.3,                          
        random_state=42                
    )

    # Initialize the Random Forest Regressor
    rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)

    # Fit the model on the training data
    rf_regressor.fit(X_train, y_train)

    # Predict on the test set
    y_pred = rf_regressor.predict(X_test)

    # Calculate the mean squared error of the predictions
    mse_forest = mean_absolute_error(y_test, y_pred)
    print(f"Mean Absolute Error of Forest: {mse_forest}")
    
    # Calculate the naive baseline performance
    naive_baseline = y_train.mean()
    mse_naive = mean_absolute_error(np.repeat(naive_baseline, len(y_pred)), y_pred)
    print(f"Mean Absolute Error of Naive Baseline: {mse_naive}")

    # calculate the relative improvement
    relative_improvement = (mse_forest- mse_naive) / mse_naive
    print(f"Relative Improvement: {relative_improvement*100:.2f}%")


# %%
# visualise the predictions vs the ground truth 
import matplotlib.pyplot as plt

# Visualize the predictions vs the ground truth
plt.scatter(y_test, y_pred)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k--', lw=2)
plt.xlabel("Ground Truth")
plt.ylabel("Predictions")
plt.title("Predictions vs Ground Truth")
plt.show()

# %%
import numpy as np
import matplotlib.pyplot as plt

# Extract the individual tree predictions for the test set
tree_predictions = np.array([tree.predict(X_test) for tree in rf_regressor.estimators_])

# Flatten the array of tree predictions to use in the scatter plot
all_tree_predictions = tree_predictions.flatten()

# Repeat the ground truth values for each tree prediction
ground_truth_repeated = np.repeat(y_test, len(rf_regressor.estimators_))

# Create the scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(ground_truth_repeated, all_tree_predictions,marker='x',alpha=0.1, color='blue')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k--', lw=2)
plt.title("Scatter Plot of Ground Truth vs. Tree Predictions")
plt.xlabel("Ground Truth")
plt.ylabel("Predictions of Individual Trees")
plt.grid(True)
plt.show()
# %%
