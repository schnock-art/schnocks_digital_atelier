"""Original Schnock Experiment, edits image on pixel basis.
"""
from numba import njit, prange
import numpy as np
from experiment_classes.base_experiment import BaseExperiment


@njit(cache=True, nogil=True, fastmath=False)
def editar_pixel(
    pixel: np.array,
    high_shift: np.uint8,
    mid_shift: np.uint8,
    low_shift: np.uint8,
    high_threshold: np.uint8,
    low_threshold: np.uint8,
):
    """Pixel based edition fo image.
    RGB channels are edited according high mid and low
    High channel is the channel with the highest value (argmax)
    Min channel ist the channel with the lowest vlaue (argmin)
    Mid is the remaining channel.
    The threshold are applied on the high oird low channel respectively, the threshold is not the same for all pixels.

    Args:
        pixel (np.array): Input Pixel
        high_shift (np.uint8): How is the high channel above high threshold edited
        mid_shift (np.uint8): How is the mid channel edited
        low_shift (np.uint8): How is the high channel below low threshold edited
        high_threshold (np.uint8): Threshold for the high channel
        low_threshold (np.uint8): Threshold for the low channel
    """
    max_value = pixel.max()
    min_value = pixel.min()
    arg_max = pixel.argmax()
    arg_min = pixel.argmin()
    # arg_mid = 3 - arg_max - arg_min
    if max_value < low_threshold:
        # Darker pixels
        pixel[arg_max] += mid_shift
        # pixel[arg_mid] += mid_shift
        pixel[arg_min] = max(0, pixel[arg_min]-low_shift)
    elif min_value > high_threshold:
        # Brighter pixels
        pixel[arg_min] = min(255, pixel[arg_min] + high_shift)
        # pixel[arg_mid] = min(255, pixel[arg_mid] + mid_shift)
        pixel[arg_max] = min(255, pixel[arg_max] - low_shift)
    else:
        # Main  illuminated area
        pixel[arg_max] = min(255, pixel[arg_max] + high_shift)
        # pixel[arg_max] = 255
        # pixel[arg_mid] += mid_shift
        pixel[arg_min] = max(0, pixel[arg_min] - low_shift)
        # pixel[arg_min] = 0


@njit(parallel=False, cache=True, nogil=True, fastmath=False)
def edit_image(
    imagen: np.array,
    high_shift: np.uint8 = np.uint8(30),
    mid_shift: np.uint8 = np.uint(0),
    low_shift: np.uint8 = np.uint(20),
    high_threshold: np.uint8 = np.uint(20),
    low_threshold: np.uint8 = np.uint(20),
):
    """Edits image with given configuration

    Args:
        imagen (np.array): _description_
        high_shift (np.uint8, optional): How is the high channel above high threshold edited. Defaults to np.uint8(30).
        mid_shift (np.uint8, optional): How is the remeaining channel edited. Defaults to np.uint(0).
        low_shift (np.uint8, optional): How is the low channel below low threshold edited. Defaults to np.uint(20).
        high_threshold (np.uint8, optional): Threshold for the high channel. Defaults to np.uint(20).
        low_threshold (np.uint8, optional): Threshold for the low channel. Defaults to np.uint(20).
    """
    shape = imagen.shape
    shape_x = shape[0]
    shape_y = shape[1]
    for i in prange(shape_x):
        for j in prange(shape_y):
            editar_pixel(
                imagen[i][j],
                high_shift=high_shift,
                mid_shift=mid_shift,
                low_shift=low_shift,
                high_threshold=high_threshold,
                low_threshold=low_threshold,
            )


class SchnockExperiment(BaseExperiment):
    """Schnock experiments, edits an image Pixel by Pixel with a custom function

    Args:
        BaseExperiment (BaseExperiment): BaseClass
    """

    def __init__(self):
        """Intiializes class"""
        super().__init__()
        self.high_shift = 10
        self.mid_shift = 0
        self.low_shift = 10
        self.high_threshold = 220
        self.mid_threshold = 150
        self.low_threshold = 50

    # Process Image Methods
    def compute_new_matrix(self):
        self.new_matrix = self.source_image.copy()
        edit_image(
            self.new_matrix,
            high_shift=self.high_shift,
            mid_shift=self.mid_shift,
            low_shift=self.low_shift,
            high_threshold=self.high_threshold,
            low_threshold=self.low_threshold,
        )

    # Set  Attributes Methods
    # Set Shifts
    def set_low_shift(self, new_value: int):
        """Sets low shift

        Args:
            new_value (int): New low shift value
        """
        self.low_shift = np.uint8(new_value)

    def set_mid_shift(self, new_value: int):
        """Sets mid shift

        Args:
            new_value (int): New mid shift value
        """
        self.mid_shift = np.uint8(new_value)

    def set_high_shift(self, new_value: int):
        """Sets high shift

        Args:
            new_value (int): New High shift value
        """
        self.high_shift = np.uint8(new_value)

    # Set thesholds
    def set_low_threshold(self, new_value: int):
        """Sets low thershold

        Args:
            new_value (int): New low thershold value
        """
        self.low_threshold = np.uint8(new_value)

    def set_mid_threshold(self, new_value: int):
        """Sets mid thershold

        Args:
            new_value (int): New mid thershold value
        """
        self.mid_threshold = np.uint8(new_value)

    def set_high_threshold(self, new_value: int):
        """Sets high thershold

        Args:
            new_value (int): New high thershold value
        """
        self.high_threshold = np.uint8(new_value)


# %%
if __name__ == "__main__":
    experiment = SchnockExperiment()
    # TODO: Implement launching from command line
# %%
