# %%
# import required functions and classes
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from src.utils import (_calculate_algorithm_descriptives,
                                    _calculate_descriptive_performance,
                                    _get_dataset_descriptives_master_table,
                                    _get_performance_master_dict,
                                    _load_algorithm_performance,
                                    _load_data_set)

from src.utils_visualization import NotebookFigureSaver
# Where to save the figures
CHAPTER_ID = "02b_multivariate_data_centric"
fig_saver = NotebookFigureSaver(CHAPTER_ID)

from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)



# %% [markdown]
# ### Define ML problem: mapping from data set descriptives to algorithm performance

# %%
model_performance = _load_algorithm_performance(algorithm_name="Arsenal_ACC.csv",multivariate=True)

model_performance_descriptive = {}
fold_columns = model_performance.columns.drop("folds:")

for index_data_set, row in model_performance.iterrows():
    data_set_name = row["folds:"]

    # get the raw performance for the algorithm csv and the data set row
    algorithm_data_set_performance_raw = model_performance.loc[
        model_performance["folds:"] == data_set_name
    ][fold_columns].values.flatten()

    descriptive_stats = _calculate_descriptive_performance(
        algorithm_data_set_performance_raw
    )

    model_performance_descriptive[data_set_name] = descriptive_stats

Y = pd.DataFrame(model_performance_descriptive).T
Y.head(3)

# %%
# Load dataset characteristics from CSV file
data_set_characteristics_path = os.path.join(
    "datasets", "material", "own_dataset_descriptives.csv"
)
dataset_characteristics = pd.read_csv(data_set_characteristics_path, index_col=[0])

X = dataset_characteristics.T
X.head(3)


# %%
# inner join x and y on index
matched_data_sets = X.join(Y, how="inner")

# document how many rows were not matched
num_rows_not_matched = len(X) - len(matched_data_sets)
print(f"Number of rows not matched: {num_rows_not_matched}")

input_columns = [
    "dim_count",
    "number_train_instances",
    "length_train_instance",
    "number_test_instances",
    "length_test_instance",
    "number_target_classes",
]

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
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

fig, axes = plt.subplots(
    1, 2, figsize=(12, 6), sharey=True, gridspec_kw={"width_ratios": [3, 2]}
)

# Plot the input columns as a heat map with a log scale for the colors
input_heatmap = sns.heatmap(
    matched_data_sets[input_columns],
    ax=axes[0],
    cmap="coolwarm",
    cbar_kws=dict(use_gridspec=True, location="bottom"),
    annot=True,
    fmt=".2f",
    xticklabels=True,
    norm=LogNorm(
        vmin=matched_data_sets[input_columns].min().min(),
        vmax=matched_data_sets[input_columns].max().max(),
    ),
)
axes[0].set_title("Input Variables")
axes[0].set_xlabel("Input Columns")
axes[0].set_ylabel("Data Set")

# Modify x and y tick labels font size
input_heatmap.tick_params(axis="both", labelsize=8)

# Rotate x-axis tick labels by 20 degrees
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=10, ha="right")

vmin = (matched_data_sets[input_columns].min().min(),)
vmax = (matched_data_sets[input_columns].max().max(),)

print(vmin, vmax)

# Plot the target columns as a heat map with a log scale for the colors
target_heatmap = sns.heatmap(
    matched_data_sets[target_columns],
    ax=axes[1],
    cmap="coolwarm",
    cbar_kws=dict(use_gridspec=True, location="bottom"),
    annot=True,
    fmt=".2f",
    xticklabels=True,
    norm=LogNorm(
        vmin=matched_data_sets[target_columns].min().min(),
        vmax=matched_data_sets[target_columns].max().max(),
    ),
)
axes[1].set_title("Target Variables")
axes[1].set_xlabel("Target Columns")

# Modify x and y tick labels font size
target_heatmap.tick_params(axis="both", labelsize=8)

# Rotate x-axis tick labels by 20 degrees
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=10, ha="right")
plt.tight_layout()
fig_saver.save_fig(
    f"ml_problem_data_centric"
)
plt.show()

# %% [markdown]
# ## train a basic model

# %%
matched_data_sets.columns

# %%
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

input_columns = [
    "dim_count",
    "number_train_instances",
    "length_train_instance",
    "number_test_instances",
    "length_test_instance",
    "number_target_classes",
]

target_columns = [
    "$\hat{\mu}$",
    "$\hat{\sigma}$",
    "$Q_{min}$",
    "$Q_{max}$",
    "$Q_{1}$",
    "$Q_{5}$",
    "$Q_{25}$",
    "$Q_{50}$",
    "$Q_{75}$",
    "$Q_{95}$",
    "$Q_{99}$",
]

X = matched_data_sets[input_columns]
y = matched_data_sets[target_columns]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=42
)

# Create a dictionary to store model predictions
y_pred = {}

# Train a model for each target column
for column in target_columns:
    model = LinearRegression()
    model.fit(X_train, y_train[column])
    y_pred[column] = model.predict(X_test)

# Calculate and plot the predictions
plt.figure(figsize=(20, 6))

for i, column in enumerate(target_columns):
    real_values = y_test[column]
    predicted_values = y_pred[column]

    # Calculate the root mean squared error (RMSE)
    rmse = mean_squared_error(real_values, predicted_values, squared=False)

    plt.subplot(1, len(target_columns), i + 1)
    plt.scatter(real_values, predicted_values, color="b", alpha=0.5)
    plt.plot(
        [real_values.min(), real_values.max()],
        [real_values.min(), real_values.max()],
        "k--",
        lw=2,
    )
    plt.xlabel(f"Real {column}")
    plt.ylabel(f"Predicted {column}")
    plt.title(f"Real vs Predicted {column}\nRMSE: {rmse:.2f}")

plt.tight_layout()
plt.show()


# %% [markdown]
# # TODO based on the descriptives of the data sets, predict the performance of different algortihm
#
#
# - high level heatmap: datasets (y-axis), algorithm (x-axis), performance (color)
# - analysis descriptives data set: datasets (x-axis), descriptes (y-axis), value (color)
#
# - initial model: to predict performance (mean) based on descriptes of algorithm


# %%
