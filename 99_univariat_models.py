# %%
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sktime.classification.deep_learning import InceptionTimeClassifier
from sktime.classification.deep_learning.cnn import CNNClassifier
from sktime.classification.dictionary_based import MUSE, WEASEL, BOSSEnsemble
from sktime.classification.distance_based import ProximityForest
from sktime.classification.dummy import DummyClassifier
from sktime.classification.feature_based import TSFreshClassifier
from sktime.classification.hybrid import HIVECOTEV1, HIVECOTEV2
from sktime.classification.interval_based import (
    RandomIntervalSpectralEnsemble,
    TimeSeriesForestClassifier,
)
from sktime.classification.kernel_based import RocketClassifier
from sktime.classification.shapelet_based import MrSEQL, MrSQM
from sktime.datasets._data_io import _load_provided_dataset
from sktime.transformations.panel import channel_selection
from sktime.transformations.panel.rocket import Rocket

DATA_PATH = os.path.join(os.getcwd(), "datasets", "Univariate_ts")

# %%
data_set_name = "ArrowHead"

X_train, y_train = _load_provided_dataset(
    name=data_set_name,
    split="train",
    return_X_y=True,
    return_type=None,
    extract_path=DATA_PATH,
)

X_test, y_test = _load_provided_dataset(
    name=data_set_name,
    split="test",
    return_X_y=True,
    return_type=None,
    extract_path=DATA_PATH,
)

# %%
DummyClassifier_model = DummyClassifier()
RocketClassifier_model = RocketClassifier(num_kernels=500)
CNNClassifier_model = CNNClassifier(n_epochs=50,batch_size=4)
InceptionTimeClassifier_model = InceptionTimeClassifier(n_epochs=1, batch_size=64, kernel_size=40, n_filters=32)
MUSE_model = MUSE()

RandomIntervalSpectralEnsemble_model = RandomIntervalSpectralEnsemble(n_estimators=10) 
BOSSEnsemble_model = BOSSEnsemble() 
WEASEL_model = WEASEL() 
TimeSeriesForestClassifier_model = TimeSeriesForestClassifier(n_estimators=10) 


# cs = channel_selection.ElbowClassSum()  # ECS
cs = channel_selection.ElbowClassPairwise()  # ECP

rocket_pipeline = make_pipeline(cs, Rocket(), RidgeClassifierCV())

# %%
RandomIntervalSpectralEnsemble_model.fit(X_train, y_train)
RandomIntervalSpectralEnsemble_model.score(X_test, y_test)

# %%



