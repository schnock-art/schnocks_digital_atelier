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
    max_value = pixel.max()
    min_value = pixel.min()
    arg_max = pixel.argmax()
    arg_min = pixel.argmin()
    arg_mid = 3 - arg_max - arg_min
    if max_value < low_threshold:
        pixel[arg_max] += low_shift
        # pixel[arg_mid] += mid_shift
        # pixel[arg_min] = 0 #max(0, pixel[arg_min]-low_shift)
    elif min_value > high_threshold:
        pixel[arg_min] -= high_shift
        pixel[arg_mid] -= mid_shift
        # pixel[arg_max] -= low_shift
    else:
        pixel[arg_max] = 255
        pixel[arg_mid] += mid_shift
        pixel[arg_min] = 0


@njit(parallel=False, cache=True, nogil=True, fastmath=False)
def editar_imagen(
    imagen: np.array,
    high_shift: np.uint8 = np.uint8(30),
    mid_shift: np.uint8 = np.uint(0),
    low_shift: np.uint8 = np.uint(20),
    high_threshold: np.uint8 = np.uint(20),
    low_threshold: np.uint8 = np.uint(20),
):
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
    def __init__(self):
        self.high_shift = 10
        self.mid_shift = 0
        self.low_shift = 10
        self.high_threshold = 220
        self.mid_threshold = 150
        self.low_threshold = 50

    def set_high_shift(self, new_value: int):
        self.high_shift = np.uint8(new_value)

    def set_mid_shift(self, new_value: int):
        self.mid_shift = np.uint8(new_value)

    def set_low_shift(self, new_value: int):
        self.low_shift = np.uint8(new_value)

    def set_low_threshold(self, new_value: int):
        self.low_threshold = np.uint8(new_value)

    def set_mid_threshold(self, new_value: int):
        self.mid_threshold = np.uint8(new_value)

    def set_high_threshold(self, new_value: int):
        self.high_threshold = np.uint8(new_value)

    def compute_new_matrix(self):
        self.new_matrix = self.source_image.copy()
        editar_imagen(
            self.new_matrix,
            high_shift=self.high_shift,
            mid_shift=self.mid_shift,
            low_shift=self.low_shift,
            high_threshold=self.high_threshold,
            low_threshold=self.low_threshold,
        )


# %%
if __name__ == "__main__":
    experiment = SchnockExperiment()
    # TODO: Implement launching from command line
# %%
