# %%
# import required functions and classes
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from src.utils_performance import (
    _get_algorithm_performance_all_data_set,
)

from src.utils_visualization import NotebookFigureSaver
from src.utils_data_centric import _get_all_data_set_characteristics

# Where to save the figures
CHAPTER_ID = "01d_univariate_uncertainty"
fig_saver = NotebookFigureSaver(CHAPTER_ID)


# %%
normalized_data_set_characteristics = _get_all_data_set_characteristics(multivariate=False, number_data_sets=10)
model_performance_descriptive = _get_algorithm_performance_all_data_set(
    algorithm_name="Arsenal_ACC.csv", multivariate=False
)

#%%
X = pd.DataFrame(normalized_data_set_characteristics)
Y = pd.DataFrame(model_performance_descriptive).T

#%%
X

#%%
Y

# %%
# inner join x and y on index
matched_data_sets = X.join(Y, how="inner")
# document how many rows were not matched
num_rows_not_matched = len(X) - len(matched_data_sets)
print(f"Number of rows not matched: {num_rows_not_matched} from total of {len(X)}")

# %%

input_columns = X.columns

target_columns = [
    "$\hat{\mu}$",
    #   '$\hat{\sigma}$',   #! check, leads to errors
    "$Q_{max}$",
    "$Q_{50}$",
    "$Q_{75}$",
    "$Q_{95}$",
    "$Q_{99}$",
]

# %%
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

fig, axes = plt.subplots(
    1, 2, figsize=(12, 6), sharey=True, gridspec_kw={"width_ratios": [5, 2]}
)

# Plot the input columns as a heat map with a log scale for the colors
input_heatmap = sns.heatmap(
    matched_data_sets[input_columns],
    ax=axes[0],
    cmap="coolwarm",
    cbar_kws=dict(use_gridspec=True, location="bottom"),
    annot=False,
    # min and max values for the colorbar selected by try and error
    # vmax=0.1,
    # vmin=-0.5,
    xticklabels=True,
)
axes[0].set_title("Input Descriptive Characteristics")
axes[0].set_xlabel("Descriptive Characteristics")
axes[0].set_ylabel("Data Set")

# Modify x and y tick labels font size
input_heatmap.tick_params(axis="both", labelsize=8)

# Rotate x-axis tick labels by 20 degrees
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=90, ha="right")

# Plot the target columns as a heat map with a log scale for the colors
target_heatmap = sns.heatmap(
    matched_data_sets[target_columns],
    ax=axes[1],
    cmap="coolwarm",
    cbar_kws=dict(use_gridspec=True, location="bottom"),
    annot=True,
    fmt=".2f",
    xticklabels=True,
)
axes[1].set_title("Target Performance")
axes[1].set_xlabel("Target Columns")

# Modify x and y tick labels font size
target_heatmap.tick_params(axis="both", labelsize=8)

axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0, ha="right")
plt.tight_layout()
plt.show()


# %%
