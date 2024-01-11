# %%
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import seaborn as sns
from get_root import PROJECT_ROOT

from src.utils import _load_data_set, list_data_sets
from src.utils_visualization import NotebookFigureSaver
from src.utils_performance import (
    _get_data_set_descriptive_performance,
    _get_performance_master_dict
)
from tqdm import tqdm
from matplotlib.cm import get_cmap

# Where to save the figures
CHAPTER_ID = "a_explore_data"
fig_saver = NotebookFigureSaver(CHAPTER_ID)



#%%
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

    cmap = matplotlib.colormaps.get_cmap("tab10")  # You can choose different colormaps like 'viridis', 'plasma', etc.

    # Create subplots for each class
    fig, axes = plt.subplots(
        len(classes), 1, figsize=(10, 3 * len(classes)), sharex=True, sharey=True
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

        first_instance_in_class = True
        # Plot the selected time series on the corresponding subplot with the same color
        for idx, row in class_samples.iterrows():
            
            label = f"Instances 1-{n_samples_per_class}" if first_instance_in_class else ""
            first_instance_in_class = False
                
            axes[i].plot(
                row["dim_0"], color=cmap(0), alpha=1,  linewidth=0.5, label = label
            )  # Use blue color with some transparency

        # Calculate and plot the average line for the class (bold line)
        average_line = class_samples["dim_0"].apply(lambda x: pd.Series(x)).mean(axis=0)
        axes[i].plot(
            average_line, color=cmap(1), linewidth=3, label="Average", alpha=1
        )  # Use red color for the average line


        # Add title and labels to each subplot
        axes[i].set_title(f"Target Class: " + r"$\bf{" + str(target_class) + "}$", fontsize=18)

        if i == len(classes) - 1:
            axes[i].set_xlabel("Time Step")
        elif i == 0:
            axes[i].set_ylabel("Normalized Value", fontsize=18)
            axes[i].legend(loc="upper right",fontsize=18)
        axes[i].set_xlim(row["dim_0"].index.min(), row["dim_0"].index.max())
        axes[i].set_ylim(0, 1)
        axes[i].set_yticks([0, 0.5, 1])
        axes[i].tick_params(axis="x", labelsize=18) 
        axes[i].tick_params(axis="y", labelsize=18) 
        axes[i].grid(True)

        if i >= max_number_classes:
            break

    # Adjust layout spacing between subplots
    # plt.xticks(fontsize=18)
    # plt.yticks(fontsize=18)
    plt.xlabel("Time Step", fontsize=18)
    
    plt.subplots_adjust(hspace=.2, wspace=0., left=0.1, right=1., bottom=0.11, top=.95) 
    if save_figure:
        fig_saver.save_fig(
            f"data_set_{data_set_name}_samples_{n_samples_per_class}_classes_{max_number_classes}"
        )
    # Show the plot
    plt.show()


# %%
# show that the visualisation is working for one example data set
_visualize_instances_per_target(
    data_set_name="Yoga",
    n_samples_per_class=10,
    max_number_classes=2,
    save_figure=True,
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
performance_dict = _get_performance_master_dict(multivariate=False)
# %%

# Define the dataset name and number of algorithms
data_set_name = "Beef"
number_algorithms = 10

data_set_performance = performance_dict[data_set_name]

plt.figure(figsize=(12, 6)) 

# Iterate through the first `number_algorithms` algorithms
for algorithm_name in list(data_set_performance.columns)[:number_algorithms]:
    # Create a line plot for each algorithm
    sorted_performance = sorted(data_set_performance[algorithm_name].values.flatten())
    plt.plot(range(len(sorted_performance)), sorted_performance, label=algorithm_name.strip("_ACC"), marker='o')

# Enhance the plot with more descriptive labels and title
plt.xlabel("Cross-validation Folds", fontsize=12)
plt.ylabel("Performance (Accuracy %)", fontsize=12)
plt.title(f"Performance on the {data_set_name} Dataset", fontsize=14)

# Add a legend outside the plot
plt.legend(title="Algorithms", loc='upper left',ncol=5)

# Adjust layout to accommodate the legend and avoid cutting off labels/titles
plt.tight_layout()

# Show the plot
plt.show()

# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

def plot_performance_distributions_matplotlib(data, number_algorithms):
    # Determine the overall range of performance values
    x_min, x_max = data.min().min(), data.max().max()
    x_range = np.linspace(x_min, x_max, 500)

    # Create a color palette
    cmap = get_cmap("tab20")  # You can choose different colormaps like 'viridis', 'plasma', etc.

    # Create a figure with subplots
    fig, axes = plt.subplots(number_algorithms, 1, figsize=(10, number_algorithms * 0.75), sharex=True)

    # Ensure axes is an array even if there's only one plot
    if number_algorithms == 1:
        axes = [axes]

    # Iterate over algorithms
    for i, (algorithm_name, performance) in enumerate(data.items()):
        if i >= number_algorithms:
            break

        # Compute KDE
        kde = gaussian_kde(performance)
        kde.set_bandwidth(bw_method=kde.factor / 3.)
        
        kde_values = kde(x_range)

        # Plot KDE on the corresponding axes
        ax = axes[i]
        color = cmap(i / number_algorithms)
        ax.fill_between(x_range, kde_values, alpha=0.5, color=color)
        ax.plot(x_range, kde_values, lw=1, color=color, label=algorithm_name.strip("_ACC"))
        ax.legend(loc='lower left', fontsize=18)
        # ax.set_ylabel(algorithm_name.strip("_ACC"), fontsize=10)
        ax.set_ylabel("")
        ax.set_yticks([])
        ax.set_xlim([x_min, x_max])
        ax.grid(True)
    
    # Set labels and titles
    plt.xlabel("Performance (Accuracy %)", fontsize=18)
    plt.xticks(fontsize=18)
    

    # Adjust layout
    plt.subplots_adjust(hspace=.0, wspace=0.,left=0., right=1., bottom=0.15, top=1.) 
    fig_saver.save_fig(
            f"data_set_{data_set_name}_performance_distributions_{number_algorithms}_algorithms"
        )
    plt.show()

# Example usage
data_set_name = "Yoga"
number_algorithms = 6
data_set_performance = performance_dict[data_set_name]
plot_performance_distributions_matplotlib(data_set_performance, number_algorithms)


# %%