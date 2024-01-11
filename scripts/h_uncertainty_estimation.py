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
import random

# Random Seed at file level
random_seed = 42

np.random.seed(random_seed)
random.seed(random_seed)

# Where to save the figures
CHAPTER_ID = "h_uncertainty_estimation"
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

#%%
X

# %%
matched_data_sets

# %%
import pandas as pd
import numpy as np
from sklearn.ensemble import (
    RandomForestRegressor,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Define the regression models to test
model = RandomForestRegressor(n_estimators=100, random_state=42)

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

    results_dict[applied_algorithm] = {}
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

    # Store individual tree predictions
    individual_tree_predictions = []
    for i, tree in enumerate(model.estimators_):
        individual_tree_predictions.append(tree.predict(X_test))

    # Store results
    results_dict[applied_algorithm] = {
        "GT Performance": y_test,
        "Estimated Performance": y_pred,
        "Individual": individual_tree_predictions,
        "Naive Baseline": np.repeat(naive_baseline, len(y_test)),
        "Naive mae": mae_naive,
        "Model mae": mae_model,
        "Improvement": relative_improvement,
    }

# %%
results_dict["ResNet_ACC"]

# %%

import matplotlib.pyplot as plt
import seaborn as sns


def plot_performance_distributions(
    results_dict,
    data_set_index,
    n_algorithms,
    bw_adjust=1.0,
    algorithm_line_width=2,
    save_figure=False,
):
    # Create a figure with subplots
    fig, axes = plt.subplots(
        n_algorithms, 1, figsize=(10, n_algorithms * 2), sharex=True, sharey=False
    )

    # Ensure axes is an array even if there's only one plot
    if n_algorithms == 1:
        axes = [axes]

    # Loop through the first n_algorithms
    for i, algorithm_choice in enumerate(list(results_dict.keys())[:n_algorithms]):
        # Get the individual tree predictions, estimated performance, and GT performance
        individual_predictions = results_dict[algorithm_choice]["Individual"][
            data_set_index
        ]
        estimated_performance = results_dict[algorithm_choice]["Estimated Performance"][
            data_set_index
        ]
        gt_performance = results_dict[algorithm_choice]["GT Performance"][
            data_set_index
        ]
        naive_baseline_performance = results_dict[algorithm_choice]["Naive Baseline"][
            data_set_index
        ]
        performance_improvement = (
            abs(naive_baseline_performance - gt_performance)
            - abs(estimated_performance - gt_performance)
        ) * 100

        # Plotting the density curve for individual tree predictions
        sns.kdeplot(
            individual_predictions,
            bw_adjust=bw_adjust,  # the higher the smoother, the lower the more accurate, default 1.
            shade=True,
            color="blue",
            alpha=0.1,
            linewidth=1,
            label="Tree $\mathbb{E}_{h}^{d}$",
            ax=axes[i],
        )

        # Plotting vertical lines for estimated performance and GT performance
        axes[i].axvline(
            gt_performance,
            color="green",
            linestyle="--",
            linewidth=algorithm_line_width,
            label="GT $\mathbb{E}_{h}^{d}$",
        )
        axes[i].axvline(
            naive_baseline_performance,
            color="grey",
            linestyle="-",
            linewidth=2,
            label="Naive Baseline $\mathbb{E}_{h}^{d}$",
        )
        axes[i].axvline(
            estimated_performance,
            color="blue",
            linestyle="dotted",
            linewidth=algorithm_line_width,
            label="Estimated ",
        )

        # Setting labels and title for each subplot
        axes[i].set_xlabel(r"Performance (Accuracy $\%$)", fontsize=18)
        # axes[i].set_xlim([0.3,1.2])
        axes[i].tick_params(axis="x", labelsize=18)
        axes[i].tick_params(axis="y", labelsize=18)
        algorithm_name = algorithm_choice.replace("_ACC", "")
        axes[i].set_title(
            f"{algorithm_name} "
            + r"$\mathbb{E}_{h}^{d}$ [$\Delta$"
            + f" {performance_improvement:.2f}%pt]",
            fontsize=18,
        )
        axes[i].set_ylabel("")
        axes[i].set_yticklabels([""])
        if i == 0:
            axes[i].legend(loc="best", fontsize=18, ncol=2)

    plt.subplots_adjust(
        hspace=0.3, wspace=0.0, left=0.02, right=0.99, bottom=0.12, top=0.94
    )

    if save_figure:
        fig_saver.save_fig(
            f"performance_distributions_{data_set_index}_algorithms_{n_algorithms}"
        )
    plt.show()


# %%
n_algorithms = 3  # Number of algorithms to plot
data_set_index = 0  # Index of the dataset
plot_performance_distributions(
    results_dict,
    data_set_index,
    n_algorithms,
    bw_adjust=0.2,
    algorithm_line_width=4,
    save_figure=True,
)


# %%
n_algorithms = 4  # Number of algorithms to plot
data_set_index = 10  # Index of the dataset
plot_performance_distributions(
    results_dict, data_set_index, n_algorithms, algorithm_line_width=4, save_figure=True
)


# %%
