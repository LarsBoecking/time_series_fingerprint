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

simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

DATA_PATH = os.path.join(os.getcwd(), "datasets", "Multivariate_ts")


def _load_multivariate_data_set(
    data_set_name: str = "ArticularyWordRecognition",
    debugging_information: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load a specific multivariate data set with the predefined train test split from sk-time.

    Parameters:
        data_set_name (str): The name of the data set to load. Defaults to "ArticularyWordRecognition".
        debugging_information (bool): Whether to print debugging information. Defaults to False.

    Returns:
        tuple: A tuple containing the train data and test data.
    """
    train_data = _load_provided_dataset(
        name=data_set_name,
        split="train",
        return_X_y=False,
        return_type=None,
        extract_path=DATA_PATH,
    )

    test_data = _load_provided_dataset(
        name=data_set_name,
        split="test",
        return_X_y=False,
        return_type=None,
        extract_path=DATA_PATH,
    )

    if debugging_information:
        print(f"Data set train instances: {train_data.shape[0]}")
        print(f"Train targets: {train_data.iloc[:,-1].value_counts().to_dict()}")
        train_instance_length = [
            train_data.iloc[instance, 0].shape[0] for instance in range(len(train_data))
        ]
        print(
            f"Train instance lengths: {pd.DataFrame(train_instance_length).value_counts().to_dict()}"
        )
        num_dimensions = len([col for col in train_data.columns if "dim" in col])
        print(f"The Train data has {num_dimensions} dimensions\n")

        print(f"Data set test instances: {test_data.shape[0]}")
        print(f"Test targets: {test_data.iloc[:,-1].value_counts().to_dict()}")
        test_instance_length = [
            test_data.iloc[instance, 0].shape[0] for instance in range(len(test_data))
        ]
        print(
            f"Test instance lengths: {pd.DataFrame(test_instance_length).value_counts().to_dict()}"
        )
        num_dimensions = len([col for col in test_data.columns if "dim" in col])
        print(f"The Test data has {num_dimensions} dimensions")

    return train_data, test_data


def _get_performance_master_dict():
    """
    Generate a dictionary containing the raw performance metrics for each algorithm on each data set and all available folds.
    Note:
        This function assumes that the algorithm performance results are stored in CSV files.
        The CSV files should be located in the "results/classification/Multivariate" directory.
        The CSV file names should correspond to the algorithm names.
        The first column of each CSV file should contain the fold numbers, labeled as "folds:".
        The remaining columns should contain the performance metrics for each fold.

    Returns:
        performance_dict (Dict[str, pd.DataFrame]):
            A dictionary where the keys are the data set names and the values are pandas DataFrame.
            The columns in the DataFrame corespond to the available algorithms and the rows to the folds.
    """
    
    # result path multi variate classification
    mutlivariate_result_path = os.path.join("results", "classification", "Multivariate")

    # List all files in the current directory
    algorithm_result = os.listdir(mutlivariate_result_path)

    performance_dict = defaultdict(dict)
    # Print the list of files
    for index_algorithm, algorithm_name in enumerate(algorithm_result):
        algorithm_performance = _load_algorithm_performance(
            algorithm_name=algorithm_name
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

            performance_dict[data_set_name][
                algorithm_name
            ] = algorithm_data_set_performance_raw

    # iterate over all data sets in performance_dict and transform the value into a pd.DataFrame
    for data_set_name, algorithm_performance in performance_dict.items():
        performance_dict[data_set_name] = pd.DataFrame(algorithm_performance)

    return performance_dict


def _calculate_descriptive_performance(algorithm_data_set_performance_raw: np.ndarray) -> Dict[str, float]:
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


def _calculate_algorithm_descriptives(performance_dict: Dict[str, Dict[str, np.array]]) -> Dict[str, pd.DataFrame]:

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


def _load_algorithm_performance(algorithm_name: str = "Arsenal_ACC.csv") -> pd.DataFrame:
    """
    Load the performance csv for a specific algorithm.

    Args:
        algorithm_name (str, optional): The name of the algorithm. Defaults to "Arsenal_ACC.csv".

    Returns:
        pd.DataFrame: The algorithm performance data.
    """
    mutlivariate_result_path = os.path.join("results", "classification", "Multivariate")

    algorithm_performance = pd.read_csv(
        os.path.join(mutlivariate_result_path, algorithm_name)
    )

    return algorithm_performance


def _get_dataset_descriptives_master_table(just_testing=True):
    """
    Generate the master dictonary for dataset descriptives.
    Iterating over all available multivariate datasets in the "datasets/Multivariate_ts" folder,
    the descriptives are extracted and stored in a master dictonary

    Parameters:
        just_testing (bool): If True, only the first 3 datasets will be included in the master table. 
                             If False, all datasets will be included. Default is True.

    Returns:
        dict: (Dict[str, pd.DataFrame]):
            A dictionary where the keys are the data set names and the values are pandas DataFrame.
            The columns in the DataFrame corespond to the available algorithms and
            the rows to the calculated descriptives.
    """
    # result path multi variate classification
    multivariate_path = os.path.join("datasets", "Multivariate_ts")

    # List all files in the current directory
    if just_testing:
        available_data_sets = sorted(os.listdir(multivariate_path))[:3]
    else:
        available_data_sets = sorted(os.listdir(multivariate_path))

    # check if .DS_Store is in available_data_sets and filter it out
    if ".DS_Store" in available_data_sets:
        available_data_sets.remove(".DS_Store")
    # check if .gitignore is in available_data_sets and filter it out
    if ".gitkeep" in available_data_sets:
        available_data_sets.remove(".gitkeep")

    descriptive_dict = defaultdict(dict)

    # iterate over all available data sets in available_data_sets, extract the folder name
    for data_set_name in tqdm(available_data_sets):
        descriptive_dict[data_set_name] = _get_dataset_descriptives(data_set_name)

    return descriptive_dict


def get_dataset_descriptives(data_set_name: str) -> dict:
    """
    Retrieves descriptive statistics about a given dataset.

    Parameters:
    data_set_name (str): The name of the dataset to retrieve statistics for.

    Returns:
    dict: A dictionary containing the following descriptive statistics:
        - dim_count (int): The number of columns that contain the string "dim_" in the train_data.
        - number_train_instances (int): The number of instances in the train_data.
        - length_train_instance (int): The length of each instance in the train_data.
        - number_test_instances (int): The number of instances in the test_data.
        - length_test_instance (int): The length of each instance in the test_data.
        - number_target_classes (int): The number of unique target classes in the train_data.
    """
    train_data, test_data = _load_multivariate_data_set(data_set_name=data_set_name)

    # count columns that contain string dim_ in train_data
    dim_count = train_data.filter(like="dim_").shape[1]

    # get the data shapes of train and test
    number_train_instances = len(train_data["dim_0"])
    length_train_instance = len(train_data["dim_0"][0])
    number_test_instances = len(test_data["dim_0"])
    length_test_instance = len(test_data["dim_0"][0])

    # get the number of target classes
    number_target_classes = len(train_data["class_val"].unique())

    # generate a dict with all values calculated above
    buffer_dict = {
        "dim_count": dim_count,
        "number_train_instances": number_train_instances,
        "length_train_instance": length_train_instance,
        "number_test_instances": number_test_instances,
        "length_test_instance": length_test_instance,
        "number_target_classes": number_target_classes,
    }

    return buffer_dict


def _get_data_set_descriptive_performance(data_set_name):
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
    performance_dict = _get_performance_master_dict()
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
