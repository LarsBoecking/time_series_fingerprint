import os
from typing import Tuple
import pandas as pd
from sktime.datasets._data_io import _load_provided_dataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import numpy as np
import yaml
import os
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

# Define the data paths
UNIVARIATE_DATA_PATH = os.path.join(PROJECT_ROOT, "datasets", "Univariate_ts")

# Read the normalization parameter from the YAML file
with open(os.path.join(PROJECT_ROOT,"config.yaml"), "r") as file:  # Replace with your YAML file path
    config = yaml.safe_load(file)


def list_data_sets():
    """
    List all data sets in the defined path.

    Returns:
        list: A list of subfolder names representing data sets.
    """
    search_path = UNIVARIATE_DATA_PATH
    
    subfolders = [f.name for f in os.scandir(search_path) if f.is_dir()]
    
    return sorted(subfolders)


def _load_data_set(
    data_set_name: str = "Beef",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load a specific data set with the predefined train test split from sk-time.

    Parameters:
        data_set_name (str): The name of the data set to load. Defaults to "ArticularyWordRecognition".
    Returns:
        tuple: A tuple containing the train data and test data, possibly normalized.
    """
    extract_path = UNIVARIATE_DATA_PATH
    normalization = config.get("data_set_normalization", "none")  # Default to 'none' if not found
    
    # Load the train data set
    train_data = _load_provided_dataset(
        name=data_set_name,
        split="train",
        return_X_y=False,
        return_type=None,
        extract_path=extract_path,
    )

    # Load the test data set
    test_data = _load_provided_dataset(
        name=data_set_name,
        split="test",
        return_X_y=False,
        return_type=None,
        extract_path=extract_path,
    )

    # Apply normalization only if required
    if normalization != "none":
        if normalization == "minmax":
            scaler = MinMaxScaler()
        elif normalization == "zscore":
            scaler = StandardScaler()
        elif normalization == "mean":
            scaler = MinMaxScaler(feature_range=(-1, 1))
        elif normalization == "robust":
            scaler = RobustScaler()
        elif normalization == "unit_vector":
            scaler = lambda x: x / np.linalg.norm(x)

        # Apply normalization to train and test data
        for col in train_data.columns:
            if col.startswith('dim_'):
                # Concatenate all series in the column
                concatenated_train = pd.concat(train_data[col].tolist())
                concatenated_test = pd.concat(test_data[col].tolist())

                if normalization != "unit_vector":
                    # Normalize the concatenated series
                    concatenated_train = scaler.fit_transform(concatenated_train.values.reshape(-1, 1)).ravel()
                    concatenated_test = scaler.transform(concatenated_test.values.reshape(-1, 1)).ravel()
                else:
                    concatenated_train = scaler(concatenated_train)
                    concatenated_test = scaler(concatenated_test)

                # Split the series back to the original format
                split_indices = train_data[col].apply(len).cumsum()
                train_data[col] = np.split(concatenated_train, split_indices[:-1])
                train_data[col] = [pd.Series(data) for data in train_data[col]]

                split_indices = test_data[col].apply(len).cumsum()
                test_data[col] = np.split(concatenated_test, split_indices[:-1])
                test_data[col] = [pd.Series(data) for data in test_data[col]]

    # Return the train and test data sets
    return train_data, test_data

# %%