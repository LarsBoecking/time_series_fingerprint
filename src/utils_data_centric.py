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

from src.utils import _load_data_set, list_data_sets, config, PROJECT_ROOT

simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

# Define the data paths
MULTIVARIATE_DATA_PATH = os.path.join(PROJECT_ROOT, "datasets", "Multivariate_ts")
UNIVARIATE_DATA_PATH = os.path.join(PROJECT_ROOT, "datasets", "Univariate_ts")
DATA_CENTRIC_PATH = os.path.join(PROJECT_ROOT, "datasets", "data_centric")


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
        dict: A dictionary containing selected descriptive statistics as specified in a YAML configuration.
    """
    train_data, test_data = _load_data_set(
        data_set_name=data_set_name, multivariate=multivariate
    )
    # get the list of meta characteristics that should be calculated
    required_descriptives = config.get("dataset_meta_descriptives", [])

    data_set_descriptive_dict = {}

    if "dim_count" in required_descriptives:
        # Count columns that contain the string "dim_" in train_data
        data_set_descriptive_dict[r"$N^{d}$"] = train_data.filter(like="dim_").shape[1]

    if "number_train_instances" in required_descriptives:
        # Get the number of instances in the train_data
        data_set_descriptive_dict[r"$||X_{train}||$"] = len(train_data["dim_0"])

    if "length_train_instance" in required_descriptives:
        # Get the length of each instance in the train_data
        data_set_descriptive_dict[r"$||x_{train}^{i}||$"] = len(train_data["dim_0"][0])

    if "number_test_instances" in required_descriptives:
        # Get the number of instances in the test_data
        data_set_descriptive_dict[r"$||X_{test}||$"] = len(test_data["dim_0"])

    if "length_test_instance" in required_descriptives:
        # Get the length of each instance in the test_data
        data_set_descriptive_dict[r"$||x_{test}^{i}||$"] = len(test_data["dim_0"][0])

    if "number_target_classes" in required_descriptives:
        # Get the number of unique target classes in the train_data
        data_set_descriptive_dict[r"$||Y_{C}||$"] = len(
            train_data["class_val"].unique()
        )

    # Count instances per class
    class_counts = train_data['class_val'].value_counts()

    if "min_instances_per_class" in required_descriptives:
        # Minimum number of instances in a class
        data_set_descriptive_dict[r"$min(||I_{c}||)$"] = class_counts.min()

    if "max_instances_per_class" in required_descriptives:
        # Maximum number of instances in a class
        data_set_descriptive_dict[r"$max(||I_{c}||)$"] = class_counts.max()

    if "mean_instances_per_class" in required_descriptives:
        # Mean number of instances in a class
        data_set_descriptive_dict[r"$mean(||I_{c}||)$"] = class_counts.mean()

    if "std_instances_per_class" in required_descriptives:
        # Standard Deviation of the number of instances in a class
        data_set_descriptive_dict[r"$std(||I_{c}||)$"] = class_counts.std()

    return data_set_descriptive_dict


def _get_data_set_class_level_characteristics(data_set_name, multivariate=False):
    # Load the data
    train_data, test_data = _load_data_set(
        data_set_name=data_set_name, multivariate=multivariate
    )

    # get the individual classes
    classes = train_data["class_val"].unique()

    # collect the list of characteristics that should be calculated
    required_characteristics = config.get("dataset_instance_descriptives", [])

    target_class_descriptives = {}

    for i, target_class in enumerate(classes):
        class_samples = train_data[train_data["class_val"] == target_class]
        statistics = pd.DataFrame()

        if "percentiles" in required_characteristics:
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
            statistics.rename(
                columns={
                    "mean": "$\\bar{x}$",
                    "std": "$\sigma$",
                    "min": "min",
                    "max": "max",
                    "10%": "$x_{10}$",
                    "20%": "$x_{20}$",
                    "30%": "$x_{30}$",
                    "40%": "$x_{40}$",
                    "50%": "median",  # median is already a percentile
                    "60%": "$x_{60}$",
                    "70%": "$x_{70}$",
                    "80%": "$x_{80}$",
                    "90%": "$x_{90}$",
                },
                inplace=True,
            )

        if "IQR" in required_characteristics:
            statistics["$IQR$"] = class_samples["dim_0"].apply(
                lambda x: np.percentile(x, 75) - np.percentile(x, 25)
            )

        if "CV" in required_characteristics:
            statistics["$CV$"] = class_samples["dim_0"].apply(
                lambda x: np.std(x) / np.mean(x) if np.mean(x) != 0 else np.nan
            )

        if "mean_change" in required_characteristics:
            # Calculate statistics on the mean change in a time step and max change in a time step
            statistics[r"$\overline{\Delta x}$"] = class_samples["dim_0"].apply(
                lambda x: np.mean(np.diff(x))
            )

        if "max_change" in required_characteristics:
            statistics["$max(\Delta x)$"] = class_samples["dim_0"].apply(
                lambda x: np.max(np.diff(x))
            )

        if "std_change" in required_characteristics:
            statistics["$\sigma(\Delta x)$"] = class_samples["dim_0"].apply(
                lambda x: np.std(np.diff(x))
            )

        if "skewness" in required_characteristics:
            # Fisher-Pearson coefficient of skewness
            statistics["$\gamma_{1}$"] = class_samples["dim_0"].apply(skew)

        if "kurtosis" in required_characteristics:
            # Kurtosis is the fourth central moment divided by the square of the variance
            statistics["$Kurt[X]$"] = class_samples["dim_0"].apply(kurtosis)

        if "autocorrelation" in required_characteristics:
            # Calculate pearson autocorrelation with a lag of 1
            statistics["$R_{XX}$"] = class_samples["dim_0"].apply(
                lambda x: pearsonr(x[:-1], x[1:])[0]
            )

        target_class_descriptives[target_class] = statistics

    return target_class_descriptives


def _get_data_set_comparative_characteristics(data_set_name, multivariate=False):
    target_class_descriptives = _get_data_set_class_level_characteristics(
        data_set_name=data_set_name, multivariate=multivariate
    )

    aggregation_method = config.get("aggregation_within_class", "mean")
    comparative_measures = config.get("disparity_among_classes", "std")

    aggregated_target_class_descriptives = pd.DataFrame()
    # iterate all target_class keys and values in target_class_descriptives
    for target_class, statistics in target_class_descriptives.items():
        # Apply the specified aggregation method
        if aggregation_method == "mean":
            aggregated_statistics = statistics.mean()
        elif aggregation_method == "median":
            aggregated_statistics = statistics.median()
        # Add other aggregation methods if needed

        aggregated_statistics.name = target_class
        df_aggregated_statistics = aggregated_statistics.to_frame().T
        aggregated_target_class_descriptives = pd.concat(
            [aggregated_target_class_descriptives, df_aggregated_statistics]
        )

    aggregated_target_class_descriptives.index.name = "target_class"

    # Initialize a dictionary to hold the spread/difference statistics
    difference_descriptives = {}

    # Calculate comparative measures
    for statistic in aggregated_target_class_descriptives.columns:
        stats_values = aggregated_target_class_descriptives[statistic]

        if "std" in comparative_measures:
            difference_descriptives[
                rf"$\sigma ($" + statistic + "$)$"
            ] = stats_values.std()

        if "range" in comparative_measures:
            difference_descriptives[rf"$range ($" + statistic + "$)$"] = (
                stats_values.max() - stats_values.min()
            )

        if "IQR" in comparative_measures:
            difference_descriptives[
                rf"$IQR ($" + statistic + "$)$"
            ] = stats_values.quantile(0.75) - stats_values.quantile(0.25)

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


def _get_all_data_set_characteristics(
    multivariate=False, number_data_sets=None, normalize_each_characteristic=False
):
    # Define the filename based on the parameters
    file_name = f"data_centric_mv_{multivariate}_num_{number_data_sets}_norm_{normalize_each_characteristic}.csv"
    file_path = os.path.join(DATA_CENTRIC_PATH, file_name)

    # Check if the CSV file already exists
    if os.path.exists(file_path):
        # Load the DataFrame from the CSV file
        df = pd.read_csv(file_path, index_col=0)
        return_data_set_characteristics = df

    else:
        all_data_set_characteristics = {}

        if number_data_sets is None:
            data_sets = list_data_sets(multivariate=multivariate)
        else:
            data_sets = list_data_sets(multivariate=multivariate)[:number_data_sets]

        for data_set in tqdm(data_sets):
            # try catch block
            try:
                all_data_set_characteristics[
                    data_set
                ] = _get_overall_data_set_characteristics(
                    data_set_name=data_set,
                )
            except Exception as e:
                print(e)
                pass
        # convert all_data_set_characteristics to dataframe
        data_set_characteristics = pd.DataFrame.from_dict(
            all_data_set_characteristics, orient="index"
        )

        if normalize_each_characteristic:
            # Normalize the dataset characteristics by subtracting the mean and dividing by the standard deviation
            return_data_set_characteristics = (
                data_set_characteristics - data_set_characteristics.min()
            ) / (data_set_characteristics.max() - data_set_characteristics.min())
        else:
            return_data_set_characteristics = data_set_characteristics

        if not multivariate:
            if "$N^{d}$" in return_data_set_characteristics.columns:
                return_data_set_characteristics = return_data_set_characteristics.drop(
                    columns=["$N^{d}$"]
                )

        return_data_set_characteristics.to_csv(file_path)

    return return_data_set_characteristics


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
