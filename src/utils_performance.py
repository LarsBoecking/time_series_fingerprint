import os
from collections import defaultdict
import pandas as pd
from tqdm import tqdm
import numpy as np
from collections import defaultdict
from typing import Dict
from collections import defaultdict
from tqdm import tqdm

# Define the data paths
MULTIVARIATE_DATA_PATH = os.path.join(os.getcwd(), "datasets", "Multivariate_ts")
UNIVARIATE_DATA_PATH = os.path.join(os.getcwd(), "datasets", "Univariate_ts")


def _get_performance_master_dict(multivariate=True):
    """
    Generate a dictionary containing the raw performance metrics for each algorithm on each data set and all available folds.
    Note:
        This function assumes that the algorithm performance results are stored in CSV files.
        The CSV files should be located in the "results/classification/Multivariate" directory.
        The CSV file names should correspond to the algorithm names.
        The first column of each CSV file should contain the fold numbers, labeled as "folds:".
        The remaining columns should contain the performance metrics for each fold.
    Args:
        multivariate (bool): Flag indicating whether the performance results are for multivariate classification.
        If True, the results are assumed to be stored in the "results/classification/Multivariate" directory.
        If False, the results are assumed to be stored in the "results/classification/Univariate" directory.
    Returns:
        performance_dict (Dict[str, pd.DataFrame]):
            A dictionary where the keys are the data set names and the values are pandas DataFrame.
            The columns in the DataFrame corespond to the available algorithms and the rows to the folds.
    """
    if multivariate:
        # result path multi variate classification
        result_path = os.path.join("results", "classification", "Multivariate")
    else:
        # result path uni variate classification
        result_path = os.path.join("results", "classification", "Univariate")

    # List all files in the current directory
    algorithm_result = os.listdir(result_path)

    performance_dict = defaultdict(dict)

    for index_algorithm, algorithm_name in enumerate(algorithm_result):
        algorithm_performance = _load_algorithm_performance(
            algorithm_name=algorithm_name, multivariate=multivariate
        )
        algorithm_name = algorithm_name.rstrip(".csv")

        fold_columns = algorithm_performance.columns.drop("folds:")

        for index_data_set, row in algorithm_performance.iterrows():
            data_set_name = row["folds:"]

            if index_algorithm == 0:
                performance_dict[data_set_name] = {}

            # get the raw performance for the algorithm csv and the data set row
            algorithm_data_set_performance_raw = algorithm_performance.loc[
                algorithm_performance["folds:"] == data_set_name
            ][fold_columns].values.flatten()

            if len(algorithm_data_set_performance_raw) == 30:
                performance_dict[data_set_name][
                    algorithm_name
                ] = algorithm_data_set_performance_raw
            # else:
            #     print(f"Algorithm {algorithm_name} has {len(algorithm_data_set_performance_raw)} folds")

    # iterate over all data sets in performance_dict and transform the value into a pd.DataFrame
    for data_set_name, algorithm_performance in performance_dict.items():
        performance_dict[data_set_name] = pd.DataFrame(algorithm_performance)

    return performance_dict


def _calculate_descriptive_performance(
    algorithm_data_set_performance_raw: np.ndarray,
) -> Dict[str, float]:
    """
    Calculates various descriptive statistics for the given algorithm data set performance.
    This function serves as a wrapper function to calculate the descriptive statistics for each algorithm,
    and is called by the function calculate_algorithm_descriptives()

    Parameters:
        algorithm_data_set_performance_raw (numpy.ndarray): An array of algorithm performance values.

    Returns:
        dict: A dictionary containing various descriptive statistics including mean, standard deviation,
              minimum, maximum, and selected percentiles of algorithm data set performance.
    """
    percentiles = [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
    percentiles_dict = {
        f"$Q_{{{int(p*100)}}}$": np.quantile(algorithm_data_set_performance_raw, p)
        for p in percentiles
    }

    descriptive_dict = {
        "$\hat{\mu}$": np.mean(algorithm_data_set_performance_raw),
        "$\hat{\sigma}$": np.std(algorithm_data_set_performance_raw),
        "$Q_{min}$": np.min(algorithm_data_set_performance_raw),
        "$Q_{max}$": np.max(algorithm_data_set_performance_raw),
        **percentiles_dict,
    }

    return descriptive_dict


def _calculate_algorithm_descriptives(
    performance_dict: Dict[str, Dict[str, np.array]]
) -> Dict[str, pd.DataFrame]:
    """
    Generate a dictionary containing the DESCRIPTIVE performances for each algorithm on each data set
    _calculate_descriptive_performance
    Returns:
        performance_dict (Dict[str, pd.DataFrame]):
            A dictionary where the keys are the data set names and the values are pandas DataFrame.
            The columns in the DataFrame corespond to the available algorithms and
            the rows to the calculated descriptives.
    """

    algorithm_descriptives = defaultdict(dict)

    for data_set, algorithms in performance_dict.items():
        for algorithm, performance_data in algorithms.items():
            descriptive_stats = _calculate_descriptive_performance(performance_data)
            algorithm_descriptives[data_set][algorithm] = descriptive_stats

    # iterate over all data sets in performance_dict and transform the value into a pd.DataFrame
    for data_set_name, algorithm_performance in algorithm_descriptives.items():
        algorithm_descriptives[data_set_name] = pd.DataFrame(algorithm_performance)

    return algorithm_descriptives


def _load_algorithm_performance(
    algorithm_name: str = "Arsenal_ACC.csv",
    multivariate: bool = True,
) -> pd.DataFrame:
    """
    Load the performance csv for a specific algorithm.

    Args:
        algorithm_name (str, optional): The name of the algorithm. Defaults to "Arsenal_ACC.csv".
        multivariate (bool): Flag indicating whether the performance results are for multivariate classification.


    Returns:
        pd.DataFrame: The algorithm performance data.
    """
    if multivariate:
        # result path multi variate classification
        result_path = os.path.join("results", "classification", "Multivariate")
    else:
        # result path uni variate classification
        result_path = os.path.join("results", "classification", "Univariate")

    algorithm_performance = pd.read_csv(os.path.join(result_path, algorithm_name))

    return algorithm_performance


def _get_data_set_descriptive_performance(data_set_name, multivariate: bool = True):
    """
    Get the descriptive performance ofall available algorithms on the specified data set.
    For each algorithm the function _calculate_descriptive_performance is called processing
    the raw performances of the algorithm over all folds

    Parameters:
        data_set_name (str):
            The name of the data set.

    Returns:
        model_performance_descriptive (pandas.DataFrame):
            A DataFrame containing the descriptive performance of each algorithm on the specified data set.
    """
    performance_dict = _get_performance_master_dict(multivariate=multivariate)
    model_performance_descriptive = {}
    for algorithm_name, algorithm_performance in performance_dict[
        data_set_name
    ].items():
        performance_data = algorithm_performance.values.flatten()

        descriptive_stats = _calculate_descriptive_performance(performance_data)
        model_performance_descriptive[algorithm_name] = descriptive_stats

    # convert to pandas
    model_performance_descriptive = pd.DataFrame(model_performance_descriptive)

    return model_performance_descriptive
