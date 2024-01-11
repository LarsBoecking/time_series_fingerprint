# %%
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from get_root import PROJECT_ROOT


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
    number_embedding_shown=20,
    save_figure=False,
):
    normalized_data_set_characteristics = _get_all_data_set_characteristics(
        multivariate=multivariate,
        number_data_sets=number_data_sets,
        normalize_each_characteristic=normalize_each_characteristic,
    )
    fig_size = (10, int(number_embedding_shown / 3)+1)
    fig, ax = plt.subplots(1, 1, figsize=fig_size, sharex=False, sharey=True)

    cbar_ax = fig.add_axes([0.92, 0.25, 0.02, .73])

    sns.heatmap(
        normalized_data_set_characteristics.iloc[:, :number_embedding_shown].T,
        cmap="coolwarm",
        ax=ax,
        cbar_ax=cbar_ax,
        annot=False,
        cbar=True,
    )

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=18)
    ax.set_ylabel("Data set level embedding $h(.)$", fontsize=18)  # Set the y-axis label
    ax.tick_params(axis="x",labelsize=18)
    ax.tick_params(axis="y", labelsize=18)
    ax.grid(visible=True, linestyle="--", alpha=0.7)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")  #.set_xticklabels
    plt.subplots_adjust(hspace=.5, wspace=0.05, left=0.2, right=0.9, bottom=0.25, top=.98)

    if save_figure:
        if number_data_sets is None:
            fig_saver.save_fig(f"data_set_descriptives_all")
        else:
            fig_saver.save_fig(f"data_set_descriptives_subset_{number_data_sets}")

    plt.show()


# %%
_visualize_descriptives_data_set(
    number_data_sets=15,
    number_embedding_shown=20,
    normalize_each_characteristic=True,
    save_figure=True,
)


# %%

for number_data_sets in [10, 20, 30, 50, None]:
    _visualize_descriptives_data_set(
        number_data_sets=number_data_sets,
        normalize_each_characteristic=True,
        save_figure=True,
    )

# %%
