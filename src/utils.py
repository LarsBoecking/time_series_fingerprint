import os
from typing import Tuple
import pandas as pd
from sktime.datasets._data_io import _load_provided_dataset

# Define the data paths
MULTIVARIATE_DATA_PATH = os.path.join(os.getcwd(), "datasets", "Multivariate_ts")
UNIVARIATE_DATA_PATH = os.path.join(os.getcwd(), "datasets", "Univariate_ts")


def list_data_sets(multivariate: bool = True):
    """
    List all data sets in the defined path.

    Args:
        multivariate (bool): Whether to search for multivariate data sets.

    Returns:
        list: A list of subfolder names representing data sets.
    """
    search_path = MULTIVARIATE_DATA_PATH if multivariate else UNIVARIATE_DATA_PATH
    
    subfolders = [f.name for f in os.scandir(search_path) if f.is_dir()]
    
    return subfolders

def _load_data_set(
    data_set_name: str = "ArticularyWordRecognition",
    multivariate: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load a specific data set with the predefined train test split from sk-time.

    Parameters:
        data_set_name (str): The name of the data set to load. Defaults to "ArticularyWordRecognition".
        multivariate (bool): Whether the data set is multivariate or univariate. Defaults to True.
        debugging_information (bool): Whether to print debugging information. Defaults to False.

    Returns:
        tuple: A tuple containing the train data and test data.
    """
    # Determine the extract path based on whether the data set is multivariate or univariate
    extract_path = MULTIVARIATE_DATA_PATH if multivariate else UNIVARIATE_DATA_PATH
    
    #! normalize data set, such that the embedding is comparable

    # Load the train data set
    train_data = _load_provided_dataset(
        name=data_set_name,
        split="train",
        return_X_y=False,
        return_type=None,
        extract_path=extract_path,
    )
    
    #! normalize to have values between 0 and 1 based on train data 

    # Load the test data set
    test_data = _load_provided_dataset(
        name=data_set_name,
        split="test",
        return_X_y=False,
        return_type=None,
        extract_path=extract_path,
    )
    
    # Return the train and test data sets
    return train_data, test_data