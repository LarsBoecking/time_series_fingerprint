# %%
# import required functions and classes
import os

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

from get_root import PROJECT_ROOT

from src.utils_performance import (
    _all_algorithms_all_datasets_performance,
)

from src.utils_visualization import NotebookFigureSaver
from src.utils_data_centric import _get_all_data_set_characteristics

# Where to save the figures
CHAPTER_ID = "g_uncertainty_estimation"
fig_saver = NotebookFigureSaver(CHAPTER_ID)

# %%
# get the characteristic statistics on all data sets
normalized_data_set_characteristics = _get_all_data_set_characteristics(
    multivariate=False, number_data_sets=None, normalize_each_characteristic=True
)
X = pd.DataFrame(normalized_data_set_characteristics)

# get the mean performance of all algorithms on all data sets
algorithm_data_set_performance = _all_algorithms_all_datasets_performance(
    performance_of_interest="$\\hat{\\mu}$", multivariate=False
)
Y = pd.DataFrame(algorithm_data_set_performance).T

# inner join x and y on index
matched_data_sets = X.join(Y, how="inner")
# document how many rows were not matched
num_rows_not_matched = len(X) - len(matched_data_sets)
print(f"Number of rows not matched: {num_rows_not_matched} from total of {len(X)}")

# %%
import pandas as pd
import numpy as np
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
)
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import random 

# Random Seed at file level
random_seed = 42

np.random.seed(random_seed)
random.seed(random_seed)

# Define the regression models to test
models = {
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
    # "LinearRegression": LinearRegression(),
    "Ridge": Ridge(),
    "DecisionTree": DecisionTreeRegressor(min_samples_leaf=5, random_state=42),
    "SVR": SVR(),
    "KNeighbors": KNeighborsRegressor(),
    "GradientBoosting": GradientBoostingRegressor(),
    "AdaBoost": AdaBoostRegressor(),
    # "Lasso": Lasso(),
    # "ElasticNet": ElasticNet(),
}

# Store results
results_dict = {}

for applied_algorithm in Y.columns:
    results_dict[applied_algorithm] = {}

    X_train, X_test, y_train, y_test = train_test_split(
        matched_data_sets[X.columns],
        matched_data_sets[applied_algorithm],
        test_size=0.3,
        random_state=42,
    )

    for model_name, model in models.items():
        results_dict[applied_algorithm][model_name] = {}
        # Fit the model
        model.fit(X_train, y_train)

        # Predict and calculate errors
        y_pred = model.predict(X_test)
        mae_model = mean_absolute_error(y_test, y_pred)

        # Naive baseline performance
        naive_baseline = y_train.mean()
        mae_naive = mean_absolute_error(np.repeat(naive_baseline, len(y_pred)), y_test)

        # Calculate relative improvement
        relative_improvement = (mae_model - mae_naive) / mae_naive

        # Store results

        results_dict[applied_algorithm][model_name] = {
            "GT Performance": y_test,
            "Estimated Performance": y_pred,
            "Naive Baseline": np.repeat(naive_baseline, len(y_test)),
            "Naive mae": mae_naive,
            "Model mae": mae_model,
            "Improvement": relative_improvement,
        }

        print(
            f"Model: {model_name}\n MAE data-Centric: {mae_model},MAE Naive Baseline: {mae_naive} Relative Improvement: {relative_improvement*100:.2f}% \n"
        )

# %%
# Initialize an empty dictionary to collect data
data = {}

# Iterating through the nested dictionary
for applied_algorithm, data_load in results_dict.items():
    for model_name, performance in data_load.items():
        # Check if model_name is already a key in data
        if model_name not in data:
            data[model_name] = {}

        # Assign the improvement value to the respective algorithm under the model_name
        data[model_name][applied_algorithm] = performance["Improvement"]

# Convert collected data to DataFrame
results_df = pd.DataFrame(data)
results_df = results_df.astype(float)

# %%
import seaborn as sns
import matplotlib.pyplot as plt

save_figure = True

# Convert the DataFrame values to percentage
results_df_percentage = results_df * 100

# Define the range for the color bar
color_bar_limit = [-25, 25]  # Adjust these limits as needed

# Generate heatmap for results_df
fig_size = (10, 12)
fig, ax = plt.subplots(1, 1, figsize=fig_size, sharex=False, sharey=True)
# Add labels and title
plt.xlabel(
    "Algorithm to estimate Performance based on data-centric characteristics",
    fontsize=18,
)
# Extract algorithm names
algorithm_names = [
    algorithm_name.replace("_ACC", "") if algorithm_name.endswith("_ACC") else algorithm_name 
    for algorithm_name in results_df_percentage.index
]
results_df_percentage.index = algorithm_names

# Calculate the positions and labels for x-ticks
y_tick_positions = [i + 0.5 for i in range(len(algorithm_names))]
y_tick_labels = algorithm_names

cbar_ax = fig.add_axes([0.92, 0.1, 0.02, .88])

results_df_percentage =pd.concat([results_df_percentage,pd.DataFrame(results_df_percentage.min(axis=1),columns=["Best"])],axis=1)

sns.heatmap(
    results_df_percentage,
    cmap="coolwarm",
    ax=ax,
    cbar_ax=cbar_ax,
    annot=True,
    cbar=True,
    fmt=".1f",
    vmin=color_bar_limit[0],
    vmax=color_bar_limit[1],
    annot_kws={"size": 18},
    cbar_kws={"label": "Relative Improvement (%)"},
)
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=18)
cbar.set_label('Relative Improvement (%)', fontsize=18)

ax.set_ylabel(r"Target algorithm $h(.)$", fontsize=18)  # Set the y-axis label
ax.tick_params(axis="x",labelsize=18)
ax.tick_params(axis="y", labelsize=18)
ax.grid(visible=True, linestyle="--", alpha=0.7)
ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right")  #.set_xticklabels
ax.set_yticklabels(algorithm_names, fontsize=18)
plt.subplots_adjust(hspace=.5, wspace=0.05, left=0.2, right=0.9, bottom=0.1, top=.98)

if save_figure:
    fig_saver.save_fig(f"performance_improvement_naive_baseline")

plt.show()

#%%
results_df_percentage = results_df_percentage.round(2).astype(str)
print(results_df_percentage.to_latex())

#%%
import matplotlib.pyplot as plt
import numpy as np

applied_algorithm = "Hydra_ACC"
applied_algorithm = "WEASEL_ACC"
applied_algorithm = "WEASEL-D_ACC"
applied_algorithm = "STSF_ACC"
applied_algorithm = "ROCKET_ACC"  # Replace with the desired algorithm choice
applied_algorithm = "BOSS_ACC" 

model_name = "AdaBoost"
model_name = "SVR"
model_name = "RandomForest"

save_figure = True


fig_size = (10, 4)
fig, ax = plt.subplots(1, 1, figsize=fig_size, sharex=False, sharey=True)
gt_performance = results_dict[applied_algorithm][model_name]["GT Performance"].values
estimated_performance = results_dict[applied_algorithm][model_name][
    "Estimated Performance"
]
naive_baseline = results_dict[applied_algorithm][model_name]["Naive Baseline"]

plt.scatter(
    gt_performance,
    estimated_performance,
    label="Predicted $\mathbb{E}_{h}^{d}$",
    s=100,
    alpha=0.7,
)
plt.plot(
    [
        results_dict[applied_algorithm][model_name]["GT Performance"].values.min(),
        results_dict[applied_algorithm][model_name]["GT Performance"].values.max(),
    ],
    [
        results_dict[applied_algorithm][model_name]["Naive Baseline"].max(),
        results_dict[applied_algorithm][model_name]["Naive Baseline"].min(),
    ],
    linestyle="--",
    color="orange",
    label="Naive Baseline $\mathbb{E}_{h}^{d}$",
)
# Plot the perfect prediction line
plt.plot(
    [gt_performance.min(), gt_performance.max()],
    [gt_performance.min(), gt_performance.max()],
    "k--",
    label="GT $\mathbb{E}_{h}^{d}$",
)

# Adding error bars
for i in range(len(gt_performance)):
    if np.abs(gt_performance[i] - estimated_performance[i]) < np.abs(
        gt_performance[i] - naive_baseline[i]
    ):
        # Model is better than naive baseline
        plt.vlines(
            gt_performance[i],
            ymin=estimated_performance[i],
            ymax=naive_baseline[i],
            color="green",
            linewidth=4,
            alpha=0.7,
            linestyles="dotted",
        )
    elif np.abs(gt_performance[i] - estimated_performance[i]) > np.abs(
        gt_performance[i] - naive_baseline[i]
    ):
        # Model is worse than naive baseline
        plt.vlines(
            gt_performance[i],
            ymin=naive_baseline[i],
            ymax=estimated_performance[i],
            color="red",
            alpha=0.7,
            linewidth=4,
            linestyles="dotted",
        )

plt.xlabel("Ground Truth", fontsize=18)
plt.ylabel("Estimated", fontsize=18)
ax.tick_params(axis="x",labelsize=18)
ax.tick_params(axis="y", labelsize=18)
plt.legend(fontsize=18)
plt.subplots_adjust(hspace=.5, wspace=0.05, left=0.08, right=0.99, bottom=0.15, top=.99)
if save_figure:
    fig_saver.save_fig(f"performance_comparison_{applied_algorithm}_{model_name}")
plt.show()

# %%
