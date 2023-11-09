from sktime.datasets._data_io import _load_provided_dataset
import os
from collections import defaultdict
import pandas as pd
from tqdm import tqdm
import numpy as np
from collections import defaultdict
from typing import Tuple, Dict
import warnings
from warnings import simplefilter
from collections import defaultdict
from tqdm import tqdm
from scipy.stats import skew, kurtosis, pearsonr, entropy
from matplotlib.patches import Rectangle

from src.utils import _load_data_set

simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

# Define the data paths
MULTIVARIATE_DATA_PATH = os.path.join(os.getcwd(), "datasets", "Multivariate_ts")
UNIVARIATE_DATA_PATH = os.path.join(os.getcwd(), "datasets", "Univariate_ts")


def _get_dataset_descriptives(
    data_set_name: str,
    multivariate: bool = True,
) -> dict:
    """
    Retrieves descriptive statistics about a given dataset.

    Parameters:
        data_set_name (str): The name of the dataset to retrieve statistics for.
        multivariate (bool): Flag indicating whether the performance results are for multivariate classification.


    Returns:
        dict: A dictionary containing the following descriptive statistics:
        - dim_count (int): The number of columns that contain the string "dim_" in the train_data.
        - number_train_instances (int): The number of instances in the train_data.
        - length_train_instance (int): The length of each instance in the train_data.
        - number_test_instances (int): The number of instances in the test_data.
        - length_test_instance (int): The length of each instance in the test_data.
        - number_target_classes (int): The number of unique target classes in the train_data.
    """
    train_data, test_data = _load_data_set(
        data_set_name=data_set_name, multivariate=multivariate
    )

    if multivariate:
        # Count columns that contain the string "dim_" in train_data
        dim_count = train_data.filter(like="dim_").shape[1]

        # Get the number of instances and length of each instance in the train_data
        number_train_instances = len(train_data["dim_0"])
        length_train_instance = len(train_data["dim_0"][0])

        # Get the number of instances and length of each instance in the test_data
        number_test_instances = len(test_data["dim_0"])
        length_test_instance = len(test_data["dim_0"][0])

        # Get the number of unique target classes in the train_data
        number_target_classes = len(train_data["class_val"].unique())

        # Generate a dictionary with all the calculated values
        data_set_descriptive_dict = {
            "dim_count": dim_count,
            "number_train_instances": number_train_instances,
            "length_train_instance": length_train_instance,
            "number_test_instances": number_test_instances,
            "length_test_instance": length_test_instance,
            "number_target_classes": number_target_classes,
        }
    else:
        data_set_descriptive_dict = {}

    return data_set_descriptive_dict


def _get_data_set_class_level_characteristics(data_set_name, multivariate=False):
    # Load the data
    train_data, test_data = _load_data_set(
        data_set_name=data_set_name, multivariate=multivariate
    )

    # get the individual classes
    classes = train_data["class_val"].unique()

    target_class_descriptives = {}

    for i, target_class in enumerate(classes):
        class_samples = train_data[train_data["class_val"] == target_class]

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

    return target_class_descriptives


def _get_data_set_comparative_characteristics(data_set_name, multivariate=False):
    target_class_descriptives = _get_data_set_class_level_characteristics(
        data_set_name=data_set_name, multivariate=multivariate
    )

    mean_target_class_descriptives = pd.DataFrame()
    # iterate all target_class keys and values in target_class_descriptives
    for target_class, statistics in target_class_descriptives.items():
        # calculate the mean values over all class instances
        mean_statistics = statistics.mean()
        mean_statistics.name = (
            target_class  # Set the name of the Series to the target class
        )
        # convert the series to a dataframe and use the name as the index
        df_mean_statistics = mean_statistics.to_frame().T
        # concat the dataframe with the mean_target_class_descriptives dataframe
        mean_target_class_descriptives = pd.concat(
            [mean_target_class_descriptives, df_mean_statistics]
        )

    # name the index
    mean_target_class_descriptives.index.name = "target_class"

    # Initialize a dictionary to hold the spread/difference statistics
    difference_descriptives = {}

    # For each statistic in the mean_target_class_descriptives, calculate the various measures
    for statistic in mean_target_class_descriptives.columns:
        stats_values = mean_target_class_descriptives[statistic]
        stats_difference = {
            f"{statistic}_std": stats_values.std(),
            f"{statistic}_range": stats_values.max() - stats_values.min(),
            f"{statistic}_iqr": stats_values.quantile(0.75)
            - stats_values.quantile(0.25),
            f"{statistic}_variance": stats_values.var(),
        }
        # Add the calculated differences for each statistic to the main dictionary
        difference_descriptives.update(stats_difference)

    return difference_descriptives


def _get_overall_data_set_characteristics(data_set_name, multivariate=False):
    difference_descriptives = _get_data_set_comparative_characteristics(
        data_set_name, multivariate=multivariate
    )
    data_set_high_level_characteristics = _get_dataset_descriptives(
        data_set_name, multivariate=multivariate
    )
    combined_characteristics = (
        difference_descriptives | data_set_high_level_characteristics
    )

    return combined_characteristics


def _get_dataset_descriptives_master_table(multivariate: bool = True):
    """
    Generate the master dictionary for dataset descriptives.
    Iterating over all available multivariate datasets in the "datasets/Multivariate_ts" folder,
    the descriptives are extracted and stored in a master dictionary.

    Parameters:
        multivariate (bool): Flag indicating whether multivariate or univariate classification should be explored

    Returns:
        dict: (Dict[str, pd.DataFrame]):
            A dictionary where the keys are the dataset names and the values are pandas DataFrames.
            The columns in the DataFrame correspond to the available algorithms and
            the rows to the calculated descriptives.
    """
    # Set the path based on the multivariate flag
    data_path = MULTIVARIATE_DATA_PATH if multivariate else UNIVARIATE_DATA_PATH

    # Get a list of all files in the data path
    available_data_sets = sorted(os.listdir(data_path))

    # Filter out unnecessary files
    available_data_sets = [
        f for f in available_data_sets if f not in [".DS_Store", ".gitkeep"]
    ]

    descriptive_dict = {}

    # Iterate over all available data sets and extract the folder name
    for data_set_name in tqdm(available_data_sets):
        descriptive_dict[data_set_name] = _get_dataset_descriptives(data_set_name)

    return descriptive_dict
