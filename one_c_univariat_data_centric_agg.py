# %%
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from src.utils_data_centric import _get_overall_data_set_characteristics

from src.utils import list_data_sets

from src.utils_visualization import NotebookFigureSaver

# Where to save the figures
CHAPTER_ID = "01c_univariate_data_centric_agg"
fig_saver = NotebookFigureSaver(CHAPTER_ID)


# %% [markdown]
# ### Generate descriptive embedding
# calcuate descriptive embedding across all target classes of a given data set


# %%
data_set_name = "Beef"
_get_overall_data_set_characteristics(data_set_name, multivariate=False)
# %%

def _get_all_data_set_characteristics(multivariate=False):
    all_data_set_characteristics = {}
    
    #! remove limit to 30 data sets
    for data_set in tqdm(list_data_sets(multivariate=multivariate)[:30]):
        # try catch block
        try:
            all_data_set_characteristics[data_set] = _get_overall_data_set_characteristics(
                data_set_name=data_set,
            )
        except:
            pass
    # convert all_data_set_characteristics to dataframe
    data_set_characteristics = pd.DataFrame.from_dict(
        all_data_set_characteristics, orient="index"
    )
    # Normalize the dataset characteristics by subtracting the mean and dividing by the standard deviation
    normalized_data_set_characteristics = (
        data_set_characteristics - data_set_characteristics.mean()
    ) / data_set_characteristics.std()
    
    
    return normalized_data_set_characteristics

# %%

normalized_data_set_characteristics = _get_all_data_set_characteristics(multivariate=False)
# generate a heatmap for df
plt.figure(figsize=(15, 6))
sns.heatmap(
    normalized_data_set_characteristics,
    cmap="coolwarm",
    annot=False,
    cbar=True,
    # min and max values for the colorbar selected by try and error
    vmax=3,
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
