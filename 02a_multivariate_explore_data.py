# %%
# import required functions and classes
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from src.utils import _load_data_set, list_data_sets
from src.utils_visualization import NotebookFigureSaver

# Where to save the figures
CHAPTER_ID = "02a_multivariate_explore_data"
fig_saver = NotebookFigureSaver(CHAPTER_ID)

from src.utils import (_calculate_descriptive_performance,
                       _get_data_set_descriptive_performance,
                       _get_performance_master_dict, _load_data_set)

# %% [markdown]
# ### Visualise the multivariate timeseries data


# %%
def visualize_time_series(
    data_set_name,
    n_samples_per_class=10,
    max_number_classes=99,
    max_num_dimensions=99,
    save_figure=False,
):
    # Load the data
    train_data, test_data = _load_data_set(
        data_set_name=data_set_name, multivariate=True
    )

    # Filter classes and dimensions
    classes = train_data["class_val"].unique()
    available_classes = len(classes)
    classes = (
        classes[:max_number_classes] if len(classes) > max_number_classes else classes
    )

    available_dimensions = sum(["dim" in col for col in train_data.columns])
    num_dimensions = min(max_num_dimensions, available_dimensions)

    # Create subplots for each class and dimension
    fig, axes = plt.subplots(
        len(classes),
        num_dimensions,
        figsize=(12, 3 * len(classes)),
        sharex=True,
        sharey=True,
    )

    for i, target_class in enumerate(sorted(classes)):
        # Filter the data for the current class
        class_data = train_data[train_data["class_val"] == target_class]

        if len(class_data) < n_samples_per_class:
            print(
                f"\n ATTENTION:\n len(class_data)= {len(class_data)} < n_samples_per_class {n_samples_per_class}"
            )
            n_samples_per_class = len(class_data)

        # Select n random samples from the target class
        class_samples = class_data.sample(n=n_samples_per_class)

        for dim in range(num_dimensions):
            for idx, row in class_samples.iterrows():
                # Plot the selected time series on the corresponding subplot with the same color
                axes[i][dim].plot(row[f"dim_{dim}"], color="b", alpha=0.3)

            # Calculate and plot the average line for the class (bold line)
            average_line = (
                class_samples[f"dim_{dim}"].apply(lambda x: pd.Series(x)).mean(axis=0)
            )
            axes[i][dim].plot(
                average_line, color="r", linewidth=2, label="Average", alpha=0.9
            )

            if i == 0 and dim == 0:
                axes[i][dim].set_title(
                    "Data set: "
                    + r"$\bf{"
                    + str(data_set_name)
                    + "}$"
                    + f", [#classes: "
                    + r"$\bf{"
                    + str(available_classes)
                    + "}$"
                    + f", #dim.: "
                    + r"$\bf{"
                    + str(available_dimensions)
                    + "}$] \n"
                    + "Target: "
                    + r"$\bf{"
                    + str(target_class)
                    + "}$\n"
                    + " Dim.: "
                    + r"$\bf{"
                    + str(dim)
                    + "}$"
                )
            else:
                axes[i][dim].set_title(
                    "Target: "
                    + r"$\bf{"
                    + str(target_class)
                    + "}$\n"
                    + " Dim.: "
                    + r"$\bf{"
                    + str(dim)
                    + "}$"
                )

    # Adjust layout spacing between subplots
    plt.tight_layout()

    if save_figure:
        fig_saver.save_fig(
            f"data_set_{data_set_name}_samples_{n_samples_per_class}_classes_{max_number_classes}_dim_{max_num_dimensions}"
        )
    # Show the plot
    plt.show()


# %%

# show the visualisation for one specific data set example
data_set_name = "AtrialFibrillation"

visualize_time_series(
    data_set_name=data_set_name,
    n_samples_per_class=10,
    max_number_classes=4,
    max_num_dimensions=5,
    save_figure=False,
)

# %%
# then gerate visualisations for all data sets
for data_set in tqdm(list_data_sets(multivariate=True)):
    visualize_time_series(
        data_set_name=data_set,
        n_samples_per_class=10,
        max_num_dimensions=5,
        max_number_classes=5,
        save_figure=True,
    )


# %%
def _visualize_performance_data_set(
    data_set_name, multivariate=True, save_figure=False
):
    data_set_performance = _get_data_set_descriptive_performance(
        data_set_name, multivariate=True
    )
    # Extract algorithm names
    algorithm_names = [
        algorithm_name.strip("_ACC") for algorithm_name in data_set_performance.columns
    ]

    # Set the figure size for better visualization
    plt.figure(figsize=(10, 6))

    # Generate the heatmap
    sns.heatmap(
        data_set_performance,
        cmap="coolwarm",
        annot=True,
        fmt=".2f",
        cbar=True,
        annot_kws={"size": 7},
    )

    # Add labels and title
    plt.xlabel("Algorithm", fontsize=15)
    plt.ylabel("Descriptive Performance", fontsize=15)  # Updated y-axis label
    plt.title(f"Performance on {data_set_name} [$ACC\%$]", fontsize=20)

    # Calculate the positions and labels for x-ticks
    x_tick_positions = [i + 0.5 for i in range(len(algorithm_names))]
    x_tick_labels = algorithm_names

    # Set the x-ticks at the calculated positions and use the labels
    plt.xticks(
        x_tick_positions,
        x_tick_labels,
        rotation=90,
        ha="center",
        fontsize=10,
    )

    plt.yticks(rotation=0, fontsize=10)

    # Show the plot
    plt.grid(visible=True, linestyle="--", alpha=0.7)
    plt.tight_layout()  # Ensures that labels and ticks fit within the figure area
    if save_figure:
        fig_saver.save_fig(f"data_set_{data_set_name}_performance")
    # Show the plot
    plt.show()


# %%
# genereate visualisations for one specific data set example
data_set_name = "AtrialFibrillation"
_visualize_performance_data_set(data_set_name, save_figure=True)


# %%
# generate visualisations for all data sets
for data_set in tqdm(list_data_sets(multivariate=True)):
    # try catch block
    try:
        _visualize_performance_data_set(
            data_set_name=data_set,
            save_figure=True,
        )
    except:
        pass
# %%
