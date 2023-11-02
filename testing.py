# %%
from src.utils import (
    _calculate_descriptive_performance,
    _get_data_set_descriptive_performance,
    _get_performance_master_dict,
    _load_data_set,_get_dataset_descriptives_master_table,
    _calculate_algorithm_descriptives
)
import pandas as pd
import numpy as np
import os 

# %% [markdown]
# ### Check individual function to see their outputs



# %%
data_set_name = "ArticularyWordRecognition"

train_data, test_data = _load_data_set(data_set_name=data_set_name, multivariate=True)
train_data.head(3)

# %%
# check master table performance
performance_dict = _get_performance_master_dict()
performance_dict["ArticularyWordRecognition"].head(3)

# %%
# calculate the descriptive statistics
algorithm_descriptives = _calculate_algorithm_descriptives(performance_dict)
algorithm_descriptives["ArticularyWordRecognition"].head(3)
