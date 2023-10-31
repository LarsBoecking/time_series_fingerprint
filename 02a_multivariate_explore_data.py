# %%
# import required functions and classes
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


from src.utils_multivariate import _load_multivariate_data_set, _get_performance_master_dict, _calculate_descriptive_performance, _get_data_set_descriptive_performance


# %%
data_set_name = "ArticularyWordRecognition"

train_data, test_data = _load_multivariate_data_set(data_set_name = data_set_name)
train_data.head(3)

# %% [markdown]
# ### Visualise the multivariate timeseries data

# %%
def visualize_time_series(
    data_set_name, n_samples_per_class=10, max_number_classes=4, max_num_dimensions=5
):
    # Load the data
    train_data, test_data = _load_multivariate_data_set(data_set_name=data_set_name)

    # Filter classes and dimensions
    classes = train_data["class_val"].unique()
    classes = (
        classes[:max_number_classes] if len(classes) > max_number_classes else classes
    )

    num_dimensions = min(
        max_num_dimensions, sum(["dim" in col for col in train_data.columns])
    )

    # Create subplots for each class and dimension
    fig, axes = plt.subplots(
        len(classes),
        num_dimensions,
        figsize=(12, 3 * len(classes)),
        sharex=True,
        sharey=True,
    )

    for i, target_class in enumerate(classes):
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
            axes[i][dim].set_title(f"Target {target_class}, Dimension {dim}")

    # Adjust layout spacing between subplots
    plt.tight_layout()

    # Show the plot
    plt.show()

# %%
data_set_name="AtrialFibrillation"
data_set_name="BasicMotions"
data_set_name="Epilepsy"

visualize_time_series(
    data_set_name=data_set_name,
    n_samples_per_class=10,
    max_number_classes=4,
    max_num_dimensions=5,
)
# %%



# %%
def _visualize_performance_data_set(data_set_name):
    

    data_set_performance = _get_data_set_descriptive_performance(data_set_name)
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

    # Rotate the tick labels for better readability
    plt.xticks(
        range(len(algorithm_names)), algorithm_names, rotation=30, ha="right", fontsize=10
    )
    plt.yticks(rotation=0, fontsize=10)

    # Show the plot
    plt.grid(visible=True, linestyle="--", alpha=0.7)
    plt.tight_layout()  # Ensures that labels and ticks fit within the figure area
    plt.show()

# %%
data_set_name = "AtrialFibrillation"
_visualize_performance_data_set(data_set_name)

# %%
data_set_name = "ArticularyWordRecognition"
_visualize_performance_data_set(data_set_name)

# %%
data_set_name = "FingerMovements"
_visualize_performance_data_set(data_set_name)

# %%
