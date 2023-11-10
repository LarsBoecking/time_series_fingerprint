# %%
# import required functions and classes
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from src.utils import _load_data_set, list_data_sets
from src.utils_visualization import NotebookFigureSaver
from src.utils_performance import (
    _get_data_set_descriptive_performance,
)
from tqdm import tqdm

# Where to save the figures
CHAPTER_ID = "a_explore_data"
fig_saver = NotebookFigureSaver(CHAPTER_ID)


# %%
def _visualize_instances_per_target(
    data_set_name: str = "ArrowHead",
    n_samples_per_class: int = 5,
    max_number_classes: int = 5,
    save_figure: bool = False,
) -> None:
    """
    Generate a visualization of instances per target class in a dataset.

    Parameters:
        data_set_name (str): The name of the dataset to load. Default is "ArrowHead".
        n_samples_per_class (int): The number of samples to select from each target class. Default is 5.
        save_figure (bool): Whether to save the figure. Default is False.

    Returns:
        None
    """
    train_data, test_data = _load_data_set(
        data_set_name=data_set_name, multivariate=False
    )

    classes = train_data["class_val"].unique()
    available_classes = len(classes)

    if available_classes > max_number_classes:
        classes = classes[:max_number_classes]

    # Create subplots for each class
    fig, axes = plt.subplots(
        len(classes), 1, figsize=(12, 3 * len(classes)), sharex=True, sharey=True
    )

    for i, target_class in enumerate(sorted(classes)):
        # Filter the data for the current class
        class_data = train_data[train_data["class_val"] == target_class]

        if len(class_data) < n_samples_per_class:
            print(
                f"\n ATTENTION:\n len(class_data)= {len(class_data)} < n_samples_per_class {n_samples_per_class}"
            )
            # If there are fewer instances in the class than n_samples_per_class, adjust it
            n_samples_per_class = len(class_data)

        # Select n random samples from the target class
        class_samples = class_data.sample(n=n_samples_per_class)

        # Plot the selected time series on the corresponding subplot with the same color
        for idx, row in class_samples.iterrows():
            axes[i].plot(
                row["dim_0"], color="b", alpha=0.3
            )  # Use blue color with some transparency

        # Calculate and plot the average line for the class (bold line)
        average_line = class_samples["dim_0"].apply(lambda x: pd.Series(x)).mean(axis=0)
        axes[i].plot(
            average_line, color="r", linewidth=2, label="Average", alpha=0.9
        )  # Use red color for the average line

        # Add title and labels to each subplot
        if i == 0:
            axes[i].set_title(
                "Data set: "
                + r"$\bf{"
                + str(data_set_name)
                + "}$"
                + f", [Number of classes: "
                + r"$\bf{"
                + str(available_classes)
                + "}$] \n"
                + "Target class:"
                + r"$\bf{"
                + str(target_class)
                + "}$"
            )
        else:
            # Add title and labels to each subplot
            axes[i].set_title(f"Target class: " + r"$\bf{" + str(target_class) + "}$")

        axes[i].set_xlabel("Time Step")
        axes[i].set_ylabel("Value")
        axes[i].legend(loc="upper right")
        axes[i].set_xlim(row["dim_0"].index.min(), row["dim_0"].index.max())

        if i >= max_number_classes:
            break

    # Adjust layout spacing between subplots
    plt.tight_layout()

    if save_figure:
        fig_saver.save_fig(
            f"data_set_{data_set_name}_samples_{n_samples_per_class}_classes_{max_number_classes}"
        )
    # Show the plot
    plt.show()


# %%
# show that the visualisation is working for one example data set
_visualize_instances_per_target(
    data_set_name="Beef",
    n_samples_per_class=10,
    max_number_classes=5,
    save_figure=False,
)

# %%
# be honest
does_the_user_have_limitless_computation_power = False

if does_the_user_have_limitless_computation_power:
    # visualise all available data sets
    for data_set in list_data_sets(multivariate=False):
        print(data_set)
        _visualize_instances_per_target(
            data_set_name=data_set,
            max_number_classes=3,
            n_samples_per_class=10,
            save_figure=True,
        )


# %%
def _visualize_performance_data_set(
    data_set_name, multivariate=False, save_figure=False
):
    data_set_performance = _get_data_set_descriptive_performance(
        data_set_name, multivariate=multivariate
    )
    # Extract algorithm names
    algorithm_names = [
        algorithm_name.strip("_ACC") for algorithm_name in data_set_performance.columns
    ]

    # Set the figure size for better visualization
    plt.figure(figsize=(15, 6))

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
data_set_name = "Beef"
_visualize_performance_data_set(data_set_name, multivariate=False, save_figure=False)

# %%
# be honest
does_the_user_have_limitless_computation_power = False

if does_the_user_have_limitless_computation_power:
    # generate visualisations for all data sets
    for data_set in tqdm(list_data_sets(multivariate=False)):
        # try catch block
        try:
            _visualize_performance_data_set(
                data_set_name=data_set,
                save_figure=True,
            )
        except:
            pass

# %%
