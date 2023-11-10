# %%
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from src.utils_data_centric import _get_all_data_set_characteristics, _get_dataset_descriptives, _get_overall_data_set_characteristics

from src.utils_visualization import NotebookFigureSaver

# Where to save the figures
CHAPTER_ID = "c_univariate_data_centric_agg"
fig_saver = NotebookFigureSaver(CHAPTER_ID)


# %% [markdown]
# ### Generate descriptive embedding
# calcuate descriptive embedding across all target classes of a given data set


# %%
# check one individual data set as an example
data_set_name = "Beef"
_get_overall_data_set_characteristics(data_set_name, multivariate=False)


#%%

# then load all data sets available
normalized_data_set_characteristics = _get_all_data_set_characteristics(multivariate=False, number_data_sets=10)

# %%
normalized_data_set_characteristics

# %%
# generate a heatmap for df
plt.figure(figsize=(15, 6))
sns.heatmap(
    normalized_data_set_characteristics,
    cmap="coolwarm",
    annot=False,
    cbar=True,
    # min and max values for the colorbar selected by try and error
    vmax=1,
    vmin=-1,
)

# Add labels and title
plt.xlabel("Algorithm", fontsize=15)
plt.ylabel("Descriptive Performance", fontsize=15)
plt.title("Summarizing descriptive statistics of various data sets", fontsize=20)

# rotate the x ticks
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)

# Show the plot
plt.grid(visible=True, linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()

# %%
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import SymLogNorm

# Define a symlog normalization: The linthresh parameter defines the range within
# which the scale is linear (to avoid taking the log of zero or very small numbers).
# You can adjust this value as needed.
linthresh = 100000  # This value should be close to the smallest non-zero value in your data.
vmax = normalized_data_set_characteristics.max().max()
vmin = normalized_data_set_characteristics.min().min()

plt.figure(figsize=(15, 6))
ax = sns.heatmap(
    normalized_data_set_characteristics,
    cmap="coolwarm",
    annot=False,
    cbar=True,
    norm=SymLogNorm(linthresh=linthresh, linscale=1, vmin=vmin, vmax=vmax, base=10)
)

# Add labels and title
plt.xlabel("Dataset", fontsize=15)
plt.ylabel("Dataset statistical descriptives", fontsize=15)
# plt.title("Summarizing descriptive statistics of various data sets", fontsize=20)

# Rotate the x ticks
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)

# Show the plot
plt.grid(visible=True, linestyle="--", alpha=0.7)
plt.tight_layout()

fig_saver.save_fig(f"data_set_descriptives")

plt.show()

# %%
