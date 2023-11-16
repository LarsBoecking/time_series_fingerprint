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

# Where to save the figures
CHAPTER_ID = "f_uncertainty_estimation"
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

# Define the regression models to test
models = {
    "RandomForestRegressor": RandomForestRegressor(n_estimators=100, random_state=42),
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(),
    "DecisionTreeRegressor": DecisionTreeRegressor(),
    "SVR": SVR(),
    "KNeighborsRegressor": KNeighborsRegressor(),
    "GradientBoostingRegressor": GradientBoostingRegressor(),
    "AdaBoostRegressor": AdaBoostRegressor(),
    "Lasso": Lasso(),
    "ElasticNet": ElasticNet(),
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
plt.figure(figsize=(12, 8))
sns.heatmap(
    results_df_percentage,
    cmap="coolwarm",
    annot=True,
    cbar=True,
    fmt=".1f",
    vmin=color_bar_limit[0],
    vmax=color_bar_limit[1],
    cbar_kws={"label": "Relative Improvement (%)"},
)

# Add labels and title
plt.xlabel(
    "Algorithm to estimate Performance based on data-centric characteristics",
    fontsize=12,
)

# Extract algorithm names
algorithm_names = [
    algorithm_name.strip("_ACC") for algorithm_name in results_df_percentage.index
]
# Calculate the positions and labels for x-ticks
y_tick_positions = [i + 0.5 for i in range(len(algorithm_names))]
y_tick_labels = algorithm_names
# Set the x-ticks at the calculated positions and use the labels
plt.yticks(
    y_tick_positions,
    y_tick_labels,
)
plt.xticks(
    rotation=15,
    ha="right",
)

plt.ylabel("Target Algorithm Performance to estimate", fontsize=12)
plt.title(
    "Relative Improvement of data-centric approach vs. naive baseline [negative values better]",
    fontsize=14,
)

if save_figure:
    fig_saver.save_fig(f"performance_improvement_naive_baseline")

plt.show()


# %%
import matplotlib.pyplot as plt
import numpy as np

algorithm_choice = "ROCKET_ACC"  # Replace with the desired algorithm choice
algorithm_choice = "Hydra_ACC"
algorithm_choice = "WEASEL_ACC"

applied_algorithm = "ROCKET_ACC"
model_name = "GradientBoostingRegressor"

plt.figure(figsize=(10, 6))
gt_performance = results_dict[applied_algorithm][model_name]["GT Performance"].values
estimated_performance = results_dict[applied_algorithm][model_name][
    "Estimated Performance"
]
naive_baseline = results_dict[applied_algorithm][model_name]["Naive Baseline"]

plt.scatter(
    gt_performance,
    estimated_performance,
    label="Predicted Performance",
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
    label="Naive Baseline Performance",
)
# Plot the perfect prediction line
plt.plot(
    [gt_performance.min(), gt_performance.max()],
    [gt_performance.min(), gt_performance.max()],
    "k--",
    label="Perfect Prediction",
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
            linestyles="dotted",
        )

plt.xlabel("GT Performance")
plt.ylabel("Performance")
plt.legend()
plt.show()

# %%
