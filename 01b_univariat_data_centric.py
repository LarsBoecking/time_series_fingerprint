# %%
# import required functions and classes
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from collections import defaultdict
from tqdm import tqdm
from scipy.stats import skew, kurtosis, pearsonr, entropy
from matplotlib.patches import Rectangle

from src.utils import _load_data_set, list_data_sets

from src.utils_visualization import NotebookFigureSaver

# Where to save the figures
CHAPTER_ID = "01b_univariate_data_centric"
fig_saver = NotebookFigureSaver(CHAPTER_ID)


# %%

def _visualize_descriptives_data_set(
    data_set_name = "ArrowHead",
    n_samples_per_class = 20,
    max_number_classes = 5,
    save_figure=False,
):
    # Load the data
    train_data, test_data = _load_data_set(data_set_name=data_set_name, multivariate=False)

    # get the individual classes
    classes = train_data["class_val"].unique()
    
    # If there are more classes than max_number_classes only take the first one
    if max_number_classes < len(classes):
        classes = classes[:max_number_classes]
    
    target_class_descriptives = {}

    # iterate over all classes and calculate the descriptive statistics
    n_samples_per_class_available = [n_samples_per_class]
    
    for i, target_class in enumerate(classes):
        
        # Filter the data for the current class
        class_data = train_data[train_data["class_val"] == target_class]

        # If there are fewer instances in the class than n_samples_per_class, adjust it
        if len(class_data) < n_samples_per_class:
            print(
                f"\n ATTENTION:\n len(class_data)= {len(class_data)} < n_samples_per_class {n_samples_per_class}"
            )
            n_samples_per_class_available.append(len(class_data))

    n_samples_per_class = min(n_samples_per_class_available)
    
    for i, target_class in enumerate(classes):
        class_data = train_data[train_data["class_val"] == target_class]

        # Select n random samples from the target class
        class_samples = class_data[:n_samples_per_class]

        # Calculate basic statistics for each time series
        statistics = (
            class_samples["dim_0"]
            .apply(
                lambda x: pd.Series(x).describe(
                    percentiles=[0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]
                )
            )
            .T
        )
        statistics = statistics.T.drop(columns=["count"])

        # Calculate additional statistics
        statistics["median"] = class_samples["dim_0"].apply(lambda x: np.median(x))
        statistics["iqr"] = class_samples["dim_0"].apply(
            lambda x: np.percentile(x, 75) - np.percentile(x, 25)
        )
        statistics["cv"] = class_samples["dim_0"].apply(
            lambda x: np.std(x) / np.mean(x) if np.mean(x) != 0 else 0
        )

        # Calculate statistics on the mean change in a time step and max change in a time step
        statistics["mean_change"] = class_samples["dim_0"].apply(
            lambda x: np.mean(np.diff(x))
        )
        statistics["max_change"] = class_samples["dim_0"].apply(
            lambda x: np.max(np.diff(x))
        )
        statistics["std_change"] = class_samples["dim_0"].apply(
            lambda x: np.std(np.diff(x))
        )

        # Calculate Skewness and Kurtosis
        statistics["skewness"] = class_samples["dim_0"].apply(lambda x: skew(x))
        statistics["kurtosis"] = class_samples["dim_0"].apply(lambda x: kurtosis(x))

        # Calculate Autocorrelation
        statistics["autocorrelation"] = class_samples["dim_0"].apply(
            lambda x: pearsonr(x[:-1], x[1:])[0]
        )

        target_class_descriptives[target_class] = statistics


    # identify the mean and max values for each statistic characteristic in order the scale them to [0, 1]
    min_values = pd.DataFrame()
    max_values = pd.DataFrame()
    # iterate all target_class keys and values in target_class_descriptives
    for target_class, statistics in target_class_descriptives.items():
        # store the max and min value in each column
        class_min_values = statistics.min(axis=0)
        class_max_values = statistics.max(axis=0)

        # check if the min_values and max_values need to be updated for each value
        min_values[target_class] = class_min_values
        max_values[target_class] = class_max_values

    # identify the min and max values for each statistic characteristic over all classes
    min_values_all_classes = min_values.min(axis=1)
    max_values_all_classes = max_values.max(axis=1)

    # Create subplots for each class
    fig, axes = plt.subplots(
        1, len(classes), figsize=(4 * len(classes), 6), sharex=False, sharey=True
    )
    cbar_ax = fig.add_axes([0.92, 0.1, 0.02, 0.78])

    for i, target_class in enumerate(classes):
        class_statistics = target_class_descriptives[target_class]

        # scale each column to the range [0, 1]
        class_statistics_scaled = (class_statistics - min_values_all_classes) / (
            max_values_all_classes - min_values_all_classes
        )

        # calculate the mean values over all class instances
        mean_df = pd.DataFrame(class_statistics_scaled.mean()).T
        mean_df.index = ["$\mu$"]
        class_statistics_scaled_new = pd.concat([class_statistics_scaled, mean_df], axis=0)

        # Create a heatmap to visualize the statistics
        sns.heatmap(
            class_statistics_scaled_new.T,
            cmap="coolwarm",
            center=0,
            ax=axes[i],
            cbar=i == len(classes) - 1,
            cbar_ax=cbar_ax,
            annot=False,
            fmt=".2f",
        )
        x, y, w, h = n_samples_per_class -0.1, -0.5, 1.2, class_statistics_scaled_new.shape[1] + 2
        axes[i].add_patch(
            Rectangle((x, y), w, h, fill=False, edgecolor="crimson", lw=2, clip_on=False)
        )

        if i == 0:
            axes[i].set_title(
                "Data set: "
                + r"$\bf{"
                + str(data_set_name)
                + "}$\n"
                + "Target class:"
                + r"$\bf{"
                + str(target_class)
                + "}$"
            )
        else:
            axes[i].set_title(f"Target class: " + r"$\bf{" + str(target_class) + "}$")
            
        # write y labels if subplot i == 0
        if i == 0:
            axes[i].set_ylabel("Statistics")
        axes[i].set_xlabel("Time Series Index")

    plt.subplots_adjust(hspace=0.5, wspace=0.1)
    if save_figure:
        fig_saver.save_fig(f"data_set_{data_set_name}_samples_{n_samples_per_class}_classes_{max_number_classes}")
    # Show the plot
    plt.show()


# %%
_visualize_descriptives_data_set(
    data_set_name = "GunPoint",
    n_samples_per_class = 10,
    max_number_classes = 5,
    save_figure = True
)
# %%
for data_set in tqdm(list_data_sets(multivariate=False)):
    # try catch block
    try:
        _visualize_descriptives_data_set(
            data_set_name = data_set,
            n_samples_per_class = 10,
            max_number_classes = 5,
            save_figure = True
        )
    except:
        pass
# %%
