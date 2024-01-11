#%%

# load classic ml packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# load titanic data set via sk learn
from sklearn.datasets import fetch_openml
titanic = fetch_openml('titanic', version=1, as_frame=True)

# scale input and target data, apply split and train decision tree
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Assuming titanic.data is a pandas DataFrame
column_dtypes = titanic.data.dtypes

# Separate the numeric and categorical column names
numeric_columns = column_dtypes[column_dtypes != 'object'].index.tolist()
categorical_columns = column_dtypes[column_dtypes == 'object'].index.tolist()

# Convert dtypes accordingly

titanic.data[numeric_columns] = titanic.data[numeric_columns].apply(pd.to_numeric, errors='coerce')


# Apply scaling to numeric columns
scaler = StandardScaler()
titanic.data[numeric_columns] = scaler.fit_transform(titanic.data[numeric_columns])

# Apply encoding to categorical columns
encoder = OneHotEncoder()
encoded_data = encoder.fit_transform(titanic.data[categorical_columns])
encoded_columns = encoder.get_feature_names_out(categorical_columns)
titanic.data = pd.concat([titanic.data.drop(columns=categorical_columns), pd.DataFrame(encoded_data.toarray(), columns=encoded_columns)], axis=1)

# Apply split and train decision tree
X_train, X_test, y_train, y_test = train_test_split(titanic.data[numeric_columns], titanic.target, test_size=0.2, random_state=42)

clf = DecisionTreeClassifier()
from sklearn.impute import SimpleImputer

# Create an instance of SimpleImputer with strategy='mean'
imputer = SimpleImputer(strategy='mean')

# Fit the imputer on the training data
imputer.fit(X_train)

# Transform the training and test data using the imputer
X_train = imputer.transform(X_train)
X_test = imputer.transform(X_test)

# Fit the classifier on the imputed training data
clf.fit(X_train, y_train)

# Get feature importances
importances = clf.feature_importances_

# Create a DataFrame to store feature importances
importances = clf.feature_importances_
feature_importances = pd.DataFrame({'Feature': titanic.data.columns[:len(importances)], 'Importance': importances})

# Sort the DataFrame by importance
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

# Plot feature importances
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importances)
plt.title('Feature Importances')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()

import numpy as np
import pandas as pd

import pandas as pd
import numpy as np

def generate_time_series(distributions):
    time_series = []
    for size, mean, std_dev in distributions:
        distribution = pd.Series(np.random.normal(mean, std_dev, size))
        time_series.append(distribution)
    time_series = pd.concat(time_series)
    return time_series

generate_time_series([(100, 0, 1), (100, 1, 1), (100, 0, 0.5)])


# %%
# use river package to apply adwin alogrithm
from river.drift import ADWIN
# create adwin object
adwin = ADWIN()
# create time series
time_series = generate_time_series([(100, 0, 1), (100, 1, 1), (100, 0, 0.5)])
# apply adwin algorithm
for i, val in enumerate(time_series):
    adwin.update(val)
    if adwin.drift_detected:
        print(f'Drift detected at index {i}, input value: {val}')

# %%
