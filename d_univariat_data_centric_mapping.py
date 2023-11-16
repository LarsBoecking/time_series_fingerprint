# %%
# import required functions and classes
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from src.utils_performance import (
    _all_algorithms_all_datasets_performance,
)

from src.utils_visualization import NotebookFigureSaver
from src.utils_data_centric import _get_all_data_set_characteristics

# Where to save the figures
CHAPTER_ID = "d_data_centric_mapping"
fig_saver = NotebookFigureSaver(CHAPTER_ID)


# %%
def _visualise_data_set_mapping(
    number_data_sets=None,
    multivariate=False,
    normalize_each_characteristic=True,
    save_figure=False,
):
    # get the characteristic statistics on all data sets
    normalized_data_set_characteristics = _get_all_data_set_characteristics(
        multivariate=multivariate, number_data_sets=number_data_sets,normalize_each_characteristic=normalize_each_characteristic
    )
    X = pd.DataFrame(normalized_data_set_characteristics)


    # get the mean performance of all algorithms on all data sets
    algorithm_data_set_performance = _all_algorithms_all_datasets_performance(
        performance_of_interest="$\\hat{\\mu}$", multivariate=False
    )
    Y = pd.DataFrame(algorithm_data_set_performance).T

    # inner join x and y on index, because of missing performance on AllGestureWii Data Sets
    matched_data_sets = X.join(Y, how="left")

    # document how many rows were not matched
    num_rows_not_matched = len(X) - len(matched_data_sets)
    print(f"Number of rows not matched: {num_rows_not_matched} from total of {len(X)}")

    # scaling the figure to a reasonable size
    if number_data_sets is None:
        fig_size = (15, int(len(matched_data_sets)/5)+ 2)
    else:
        fig_size = (15, int(len(matched_data_sets)/5)+ 2)

    fig, axes = plt.subplots(
        1, 2, figsize=fig_size, sharey=False, gridspec_kw={"width_ratios": [5, 2]}
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
            "shrink": 0.25,
            "pad": 0.1,
        },  # Adjusted color bar
        vmax=1,
        vmin=0,
        annot=False,
        xticklabels=True,
        yticklabels=True,  # Ensure all y-tick labels are shown
    )
    axes[0].set_title("Input Descriptive Characteristics")
    axes[0].set_xlabel("Descriptive Characteristics")
    axes[0].set_ylabel("Data Set")

    # Set consistent formatting for x and y tick labels
    plt.setp(axes[0].get_xticklabels(), rotation=90, ha="center", fontsize=8)
    plt.setp(axes[0].get_yticklabels(), fontsize=8)

    # Plot the target columns as a heat map
    target_heatmap = sns.heatmap(
        matched_data_sets[target_columns],
        ax=axes[1],
        cmap="coolwarm",
        cbar_kws={
            "use_gridspec": True,
            "location": "top",
            "shrink": 0.7,
            "pad": 0.1,
        },  # Adjusted color bar
        vmax=1,
        vmin=0,
        annot=False,
        xticklabels=True,
        yticklabels=False,  # Hide y-tick labels for the second plot (shared y-axis)
    )
    axes[1].set_title("Target Performance")
    axes[1].set_xlabel("Algorithm")

    # Set consistent formatting for x and y tick labels
    algorithm_names = [algorithm_name.strip("_ACC") for algorithm_name in Y.columns]
    axes[1].set_xticklabels(algorithm_names, rotation=90, ha="center", fontsize=8)

    # Adjust layout to prevent overlap
    plt.tight_layout()

    if save_figure:
        if number_data_sets is None:
            fig_saver.save_fig(f"data_set_mapping_all")
        else:    
            fig_saver.save_fig(f"data_set_mapping_{number_data_sets}")


    # Show the plot
    plt.show()

# %%
_visualise_data_set_mapping(
    number_data_sets=10,
    save_figure=False,
)
# %%

for number_data_sets in [10,20,30,50,None]:
    _visualise_data_set_mapping(
        number_data_sets=number_data_sets,
        save_figure=True,
    )
# %%
