# %%
# import required functions and classes
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from src.utils_performance import (
    _all_algorithms_all_datasets_performance,
    _calculate_descriptive_performance,
    _get_algorithm_performance_all_data_set,
    _get_performance_master_dict,
)

from src.utils_visualization import NotebookFigureSaver
from src.utils_data_centric import _get_all_data_set_characteristics

# Where to save the figures
CHAPTER_ID = "01d_univariate_uncertainty"
fig_saver = NotebookFigureSaver(CHAPTER_ID)


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
# inner join x and y on index
matched_data_sets = X.join(Y, how="inner")
# document how many rows were not matched
num_rows_not_matched = len(X) - len(matched_data_sets)
print(f"Number of rows not matched: {num_rows_not_matched} from total of {len(X)}")


# %%
fig, axes = plt.subplots(
    1, 2, figsize=(12, 10), sharey=False, gridspec_kw={"width_ratios": [5, 3]}
)
input_columns = X.columns
target_columns = Y.columns

# Plot the input columns as a heat map
input_heatmap = sns.heatmap(
    matched_data_sets[input_columns],
    ax=axes[0],
    cmap="coolwarm",
    cbar_kws={
        "use_gridspec": True,
        "location": "top",
        "shrink": 0.5,
        "pad": 0.1,
    },  # Adjusted color bar
    # vmax=1,
    # vmin=-1,
    annot=False,
    xticklabels=True,
    yticklabels=True,  # Ensure all y-tick labels are shown
)
axes[0].set_title("Input Descriptive Characteristics")
axes[0].set_xlabel("Descriptive Characteristics")
axes[0].set_ylabel("Data Set")

# Set consistent formatting for x and y tick labels
plt.setp(axes[0].get_xticklabels(), rotation=90, ha="right", fontsize=8)
plt.setp(axes[0].get_yticklabels(), fontsize=8)

# Plot the target columns as a heat map
target_heatmap = sns.heatmap(
    matched_data_sets[target_columns],
    ax=axes[1],
    cmap="coolwarm",
    cbar_kws={
        "use_gridspec": True,
        "location": "top",
        "shrink": 0.5,
        "pad": 0.1,
    },  # Adjusted color bar
    annot=False,
    xticklabels=True,
    yticklabels=False,  # Hide y-tick labels for the second plot (shared y-axis)
)
axes[1].set_title("Target Performance")
axes[1].set_xlabel("Target Columns")

# Set consistent formatting for x and y tick labels
algorithm_names = [algorithm_name.strip("_ACC") for algorithm_name in Y.columns]
axes[1].set_xticklabels(algorithm_names, rotation=90, ha="right", fontsize=8)

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the plot
plt.show()


# %%
