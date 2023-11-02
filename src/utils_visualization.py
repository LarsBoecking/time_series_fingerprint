import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib as mpl

# make sure outputs are reproducable
np.random.seed(42)

# define default visualization values
FIGURE_SIZE = (15, 8)
TITLE_FONT_SIZE = 20
AXIS_LABEL_FONT_SIZE = 16
AXIS_TICKS_FONT_SIZE = 16
LEGEND_FONT_SIZE = 16


def _get_image_path(PROJECT_ROOT_DIR, CHAPTER_ID):
    IMAGES_PATH = os.path.join(os.getcwd(), "images", CHAPTER_ID)
    os.makedirs(IMAGES_PATH, exist_ok=True)

    return IMAGES_PATH


class NotebookFigureSaver:
    def __init__(self, CHAPTER_ID) -> None:
        self.PROJECT_ROOT_DIR = os.getcwd()
        self.IMAGES_PATH = _get_image_path(self.PROJECT_ROOT_DIR, CHAPTER_ID)
        self.FIGURE_SIZE = FIGURE_SIZE
        self.TITLE_FONT_SIZE = TITLE_FONT_SIZE
        self.AXIS_LABEL_FONT_SIZE = AXIS_LABEL_FONT_SIZE
        self.AXIS_TICKS_FONT_SIZE = AXIS_TICKS_FONT_SIZE
        self.LEGEND_FONT_SIZE = LEGEND_FONT_SIZE

    # function to save matplots to images folder
    def save_fig(self, fig_id, tight_layout=False, fig_extension="png", resolution=300):
        path = os.path.join(self.IMAGES_PATH, fig_id + "." + fig_extension)
        print("Saving figure", fig_id)
        if tight_layout:
            plt.tight_layout()
        plt.savefig(path, format=fig_extension, dpi=resolution)