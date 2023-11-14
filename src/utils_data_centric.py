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

from src.utils import _load_data_set, list_data_sets

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
        statistics.rename(columns={
            'mean': '$\\bar{x}$',
            'std': '$\sigma$',
            'min': 'min',
            'max': 'max',
            '10%': '$x_{10}$',
            '20%': '$x_{20}$',
            '30%': '$x_{30}$',
            '40%': '$x_{40}$',
            '50%': 'median',  # median is already a percentile
            '60%': '$x_{60}$',
            '70%': '$x_{70}$',
            '80%': '$x_{80}$',
            '90%': '$x_{90}$',
        }, inplace=True)

        # Calculate additional statistics with LaTeX symbols
        statistics['$IQR$'] = class_samples["dim_0"].apply(lambda x: np.percentile(x, 75) - np.percentile(x, 25))
        statistics['$CV$'] = class_samples["dim_0"].apply(lambda x: np.std(x) / np.mean(x) if np.mean(x) != 0 else np.nan)

        # Calculate statistics on the mean change in a time step and max change in a time step
        statistics[r'$\overline{\Delta x}$'] = class_samples["dim_0"].apply(lambda x: np.mean(np.diff(x)))
        statistics['$max(\Delta x)$'] = class_samples["dim_0"].apply(lambda x: np.max(np.diff(x)))
        statistics['$\sigma(\Delta x)$'] = class_samples["dim_0"].apply(lambda x: np.std(np.diff(x)))

        # Fisher-Pearson coefficient of skewness
        statistics['$\gamma_{1}$'] = class_samples["dim_0"].apply(skew)
        # Kurtosis is the fourth central moment divided by the square of the variance
        statistics['$Kurt[X]$'] = class_samples["dim_0"].apply(kurtosis)

        # Calculate pearson autocorrelation with a lag of 1
        statistics['$R_{XX}$'] = class_samples["dim_0"].apply(lambda x: pearsonr(x[:-1], x[1:])[0])


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
        #! could also be the median or another descriptive statistic, sigma, range, IQR
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
        #! is this really a good descriptive for how different the classes are? 
        stats_difference = {
            rf"$\sigma ($"+statistic+"$)$": stats_values.std(),
            rf"$range ($"+statistic+"$)$": stats_values.max() - stats_values.min(),
            rf"$IQR ($"+statistic+"$)$": stats_values.quantile(0.75) - stats_values.quantile(0.25),
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


def _get_all_data_set_characteristics(multivariate=False, number_data_sets=None):
    all_data_set_characteristics = {}
    
    if number_data_sets is None:
        data_sets = list_data_sets(multivariate=multivariate)
    else:
        data_sets = list_data_sets(multivariate=multivariate)[:number_data_sets]
        
    for data_set in tqdm(data_sets):
        # try catch block
        try:
            all_data_set_characteristics[data_set] = _get_overall_data_set_characteristics(
                data_set_name=data_set,
            )
        except:
            pass
    # convert all_data_set_characteristics to dataframe
    data_set_characteristics = pd.DataFrame.from_dict(
        all_data_set_characteristics, orient="index"
    )

    # Normalize the dataset characteristics by subtracting the mean and dividing by the standard deviation
    normalized_data_set_characteristics = (
        data_set_characteristics - data_set_characteristics.min()
    ) / (data_set_characteristics.max() - data_set_characteristics.min())
    
    if not multivariate:
        normalized_data_set_characteristics=normalized_data_set_characteristics.drop(columns=["dim_count"])
    
    return normalized_data_set_characteristics


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
