from sktime.datasets._data_io import _load_provided_dataset
import os 
import pandas as pd

DATA_PATH = os.path.join(os.getcwd(), "datasets", "Univariate_ts")

def _load_data_set(data_set_name = "ArrowHead"):
    """
    Load the specified data set.
    
    Args:
        data_set_name (str): The name of the data set to load. Default is "ArrowHead".
        
    Returns:
        train_data (pandas.DataFrame): The training data set.
        test_data (pandas.DataFrame): The test data set.
    """

    train_data = _load_provided_dataset(name=data_set_name,split="train", return_X_y=False, return_type=None, extract_path=DATA_PATH
                                )

    test_data = _load_provided_dataset(name=data_set_name,split="test", return_X_y=False, return_type=None, extract_path=DATA_PATH
                                )


    print(f"Data set train instances: {train_data.shape[0]}")
    print(f"Train targets: {train_data.iloc[:,1].value_counts().to_dict()}")
    train_instance_length = [train_data.iloc[instance, 0].shape[0] for instance in range(len(train_data))]
    print(f"Train instance lengths: {pd.DataFrame(train_instance_length).value_counts().to_dict()}\n")

    print(f"Data set test instances: {test_data.shape[0]}")
    print(f"Test targets: {test_data.iloc[:,1].value_counts().to_dict()}")
    test_instance_length = [test_data.iloc[instance, 0].shape[0] for instance in range(len(train_data))]
    print(f"Test instance lengths: {pd.DataFrame(test_instance_length).value_counts().to_dict()}")
    
    return train_data, test_data