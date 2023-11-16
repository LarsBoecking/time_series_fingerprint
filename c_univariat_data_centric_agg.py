# %%
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from src.utils_data_centric import (
    _get_all_data_set_characteristics,
    _get_overall_data_set_characteristics,
)

from src.utils_visualization import NotebookFigureSaver

# Where to save the figures
CHAPTER_ID = "c_data_centric_agg"
fig_saver = NotebookFigureSaver(CHAPTER_ID)

# %%
def _visualize_descriptives_data_set(
    number_data_sets=None,
    multivariate=False,
    normalize_each_characteristic=True,
    save_figure=False,
):
    normalized_data_set_characteristics = _get_all_data_set_characteristics(
        multivariate=multivariate,
        number_data_sets=number_data_sets,
        normalize_each_characteristic=normalize_each_characteristic,
    )

    # scaling the figure to a reasonable size
    if number_data_sets is None:
        fig_size = (20, int(128/5)+ 2)
    else:
        fig_size = (20, int(number_data_sets/5)+ 2)
        
    plt.figure(figsize=fig_size)
    sns.heatmap(
        normalized_data_set_characteristics,
        cmap="coolwarm",
        annot=False,
        cbar=True,
    )

    # Add labels and title
    plt.xlabel("Data Set", fontsize=15)
    plt.ylabel("Descriptive Characteristics", fontsize=15)
    plt.title("Summarizing descriptive statistics of various data sets", fontsize=20)

    # rotate the x ticks
    # plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    # Show the plot
    plt.grid(visible=True, linestyle="--", alpha=0.7)
    plt.tight_layout()

    if save_figure:
        if number_data_sets is None:
            fig_saver.save_fig(f"data_set_descriptives_all")
        else:
            fig_saver.save_fig(f"data_set_descriptives_subset_{number_data_sets}")

    plt.show()


# %%
_visualize_descriptives_data_set(
    number_data_sets=50,
    normalize_each_characteristic=True,
    save_figure=True,
)
# %%

for number_data_sets in [10,20,30,50,None]:
    _visualize_descriptives_data_set(
        number_data_sets=number_data_sets,
        normalize_each_characteristic=True,
        save_figure=True,
    )

# %%
