# %%
# import required functions and classes
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from src.utils_univariate import _load_data_set



# %%
def _visualize_instances_per_target(n_samples_per_class=5, data_set_name = "ArrowHead"):
    """
    Generate a visualization of instances per target class in a dataset.

    Parameters:
        n_samples_per_class (int): The number of samples to select from each target class. Default is 5.
        data_set_name (str): The name of the dataset to load. Default is "ArrowHead".

    Returns:
        None
    """
    train_data, test_data = _load_data_set(data_set_name = data_set_name)

    classes = train_data['class_val'].unique()

    # Create subplots for each class
    fig, axes = plt.subplots(len(classes), 1, figsize=(12, 3 * len(classes)), sharex=True, sharey=True)

    for i, target_class in enumerate(classes):
        # Filter the data for the current class
        class_data = train_data[train_data['class_val'] == target_class]
        
        if len(class_data) < n_samples_per_class:
            print(f"\n ATTENTION:\n len(class_data)= {len(class_data)} < n_samples_per_class {n_samples_per_class}")
            # If there are fewer instances in the class than n_samples_per_class, adjust it
            n_samples_per_class = len(class_data)
        
        # Select n random samples from the target class
        class_samples = class_data.sample(n=n_samples_per_class)
        
        # Plot the selected time series on the corresponding subplot with the same color
        for idx, row in class_samples.iterrows():
            axes[i].plot(row['dim_0'], color='b', alpha=0.3)  # Use blue color with some transparency
        
        # Calculate and plot the average line for the class (bold line)
        average_line = class_samples['dim_0'].apply(lambda x: pd.Series(x)).mean(axis=0)
        axes[i].plot(average_line, color='r', linewidth=2, label='Average', alpha=0.9)  # Use red color for the average line
        
        # Add title and labels to each subplot
        axes[i].set_title(f'Target class: {target_class}')
        axes[i].set_xlabel('Time Step')
        axes[i].set_ylabel('Value')
        axes[i].legend(loc="upper right")
        axes[i].set_xlim(row['dim_0'].index.min(), row['dim_0'].index.max())

    # Adjust layout spacing between subplots
    plt.tight_layout()

    # Show the plot
    plt.show()

# %%
# Call the function with your desired parameters
_visualize_instances_per_target(data_set_name="Beef", n_samples_per_class=10)

# %%
def _visualize_descriptives_targets(data_set_name = "ArrowHead", n_samples_per_class = 10):
    """
    Generate a visualization of statistical summary information for each time series in a given dataset.
    
    Parameters:
        data_set_name (str): The name of the dataset to load. Default is "ArrowHead".
        n_samples_per_class (int): The number of samples to select from each target class. Default is 10.
    
    Returns:
        None
    """

    train_data, test_data = _load_data_set(data_set_name=data_set_name)

    classes = train_data['class_val'].unique()

    # Create subplots for each class
    fig, axes = plt.subplots(1,len(classes), figsize=(4 * len(classes), 4), sharex=False, sharey=True)

    # Initialize vmin and vmax to cover the entire range of statistics
    vmin = np.inf
    vmax = -np.inf

    for i, target_class in enumerate(classes):
        # Filter the data for the current class
        class_data = train_data[train_data['class_val'] == target_class]
        
        if len(class_data) < n_samples_per_class:
            print(f"\n ATTENTION:\n len(class_data)= {len(class_data)} < n_samples_per_class {n_samples_per_class}")
            # If there are fewer instances in the class than n_samples_per_class, adjust it
            n_samples_per_class = len(class_data)
        
        # Select n random samples from the target class
        class_samples = class_data.sample(n=n_samples_per_class)
        
        # Calculate statistical summary information for each time series
        statistics = class_samples['dim_0'].apply(lambda x: pd.Series(x).describe().drop(columns=["count"])).T
        statistics = statistics.T.drop(columns=["count"]).T
        
        # Update vmin and vmax based on the statistics
        current_vmin = statistics.values.min()
        current_vmax = statistics.values.max()
        
        if current_vmin < vmin:
            vmin = current_vmin
        if current_vmax > vmax:
            vmax = current_vmax
        
        # Create a heatmap to visualize the statistics
        sns.heatmap(statistics, ax=axes[i], cmap='coolwarm', annot=False, center=True, vmin=vmin, vmax=vmax, fmt=".2f")
        
        # Add title and labels to each subplot
        axes[i].set_title(f'Target class: {target_class}')
        axes[i].set_ylabel('Statistics')
        axes[i].set_xlabel('Time Series Index')

    # Adjust layout spacing between subplots
    plt.tight_layout()

    # Show the plot
    plt.show()


# %%
_visualize_descriptives_targets(data_set_name = "ArrowHead", n_samples_per_class = 20)