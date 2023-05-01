# %%

import numpy as np
from math import sin, cos, pi
import logging
from experiment_classes.base_experiment import BaseExperiment
from experiment_classes.original_schnock import SchnockExperiment


class GradientExperiment(BaseExperiment):
    def __init__(self):
        try:
            self.source_diff = {}
            self.source_image = None
            self.matrix_list = []
            self.current_iteration_n = 0
            self.alternate_counter = 0
            self.multiplier_amplitude = 1
            self.multiplier_frequency = 4
            self.grade = 1
            self.clip_images = False
            self.alternate_every_n = 0
            self.initialize_dynamic_multiplier_dict()
            self.initialize_merge_mode_dict()
            self.initialize_video_merge_mode_dict()
            self.initialize_output_start_images_dict()
        except Exception as error:
            logging.error(
                "Failed initialization with error: {0}".format(error))
            raise error

    def pass_output_start_image(self, output_start_image):
        self.new_matrix = output_start_image

    def set_clip_images(self, value: bool):
        self.clip_images = value

    def get_difference_matrices(self):
        self.padded_source_image = {}
        for n in range(1, self.grade + 1):
            self.padded_source_image[n] = np.pad(
                self.source_image, pad_width=((n, n), (n, n), (0, 0)), mode="edge"
            )
            self.source_diff["x_{0}".format(n)] = np.diff(
                self.padded_source_image[n][n:-n, :, :], n=n, axis=1
            )
            self.source_diff["y_{0}".format(n)] = np.diff(
                self.padded_source_image[n][:, n:-n, :], n=n, axis=0
            )

    def compute_from_video(self, source_image):
        self.pass_source_image(source_image=source_image)
        self.pass_output_start_image(output_start_image=source_image)

    def compute_new_matrix(self):
        self.dynamic_multpiplier_functions_dict[self.multiplier_mode]()
        logging.debug("Dynamic multiplier: {0}".format(
            self.dynamic_multiplier))
        matrix_stack = []
        for n in range(1, self.grade + 1):
            padded_n = np.pad(
                self.new_matrix, pad_width=((n, n), (n, n), (0, 0)), mode="edge"
            )
            positive_x = padded_n[
                n:-n, (n + 1):, :
            ] + self.dynamic_multiplier * np.negative(
                self.source_diff["x_{0}".format(n)][:, n:, :]
            )
            negative_x = (
                padded_n[n:-n, : -(n + 1), :]
                + self.dynamic_multiplier
                * self.source_diff["x_{0}".format(n)][:, :-n, :]
            )
            positive_y = padded_n[
                (n + 1):, n:-n, :
            ] + self.dynamic_multiplier * np.negative(
                self.source_diff["y_{0}".format(n)][n:, :]
            )
            negative_y = (
                padded_n[0: -(n + 1), n:-n, :]
                + self.dynamic_multiplier *
                self.source_diff["y_{0}".format(n)][:-n, :]
            )
            if self.clip_images == True:
                positive_x = np.clip(positive_x, 0, 255)
                negative_x = np.clip(negative_x, 0, 255)
                positive_y = np.clip(positive_y, 0, 255)
                negative_y = np.clip(negative_y, 0, 255)
            matrix_stack += [
                positive_x,
                negative_x,
                positive_y,
                negative_y,
            ]

        self.difference_matrix = np.stack(tuple(matrix_stack))
        self.merge_mode_functions_dict[self.merge_mode]()
        self.current_iteration_n += 1

    def initialize_dynamic_multiplier_dict(self):
        self.dynamic_multpiplier_functions_dict = {
            "constant": self.dynamic_multiplier_constant,
            # "current_iteration_n": self.dynamic_multiplier_linear_reduction,
            "exponential_reduction": self.dynamic_multiplier_exponential_reduction,
            "cosinus": self.dynamic_multiplier_cos,
            "sinus": self.dynamic_multiplier_sin,
        }

    def initialize_merge_mode_dict(self):
        self.merge_mode_functions_dict = {
            "average": self.merge_average,
            "min": self.merge_min,
            "max": self.merge_max,
            "sum": self.merge_sum,
            "alternate": self.merge_alternate,
        }

    def initialize_output_start_images_dict(self):
        self.output_start_mode_dict = {
            "gray": self.ouput_start_image_gray,
            "black": self.ouput_start_image_black,
            "white": self.ouput_start_image_white,
            "source_average": self.ouput_start_image_source_average,
            "schnock_experiment": self.output_start_image_schnock_experiment,
            "video": self.output_start_image_current_source_image,
        }

    def initialize_video_merge_mode_dict(self):
        self.video_merge_mode_functions_dict = {
            "average": self.merge_average_video,
            "min": self.merge_min_video,
            "max": self.merge_max_video,
            "sum": self.merge_sum_video,
        }

    def ouput_start_image_gray(self):
        self.new_matrix = np.full(
            shape=self.source_image.shape, fill_value=127, dtype=np.uint8
        )
        self.current_iteration_n = 0
        self.alternate_counter = 0

    def ouput_start_image_black(self):
        self.new_matrix = np.full(
            shape=self.source_image.shape, fill_value=0, dtype=np.uint8
        )
        self.current_iteration_n = 0
        self.alternate_counter = 0

    def ouput_start_image_white(self):
        self.new_matrix = np.full(
            shape=self.source_image.shape, fill_value=255, dtype=np.uint8
        )
        self.current_iteration_n = 0
        self.alternate_counter = 0

    def ouput_start_image_custom(self):
        self.new_matrix = np.full(
            shape=self.source_image.shape, fill_value=127, dtype=np.uint8
        )
        self.current_iteration_n = 0
        self.alternate_counter = 0

    def ouput_start_image_source_average(self):
        mean_pixel = np.mean(self.source_image, axis=(0, 1)
                             ).round().astype(np.uint8)
        self.new_matrix = np.full(
            shape=self.source_image.shape, fill_value=mean_pixel, dtype=np.uint8
        )
        self.current_iteration_n = 0
        self.alternate_counter = 0

    def output_start_image_current_source_image(self):
        if not hasattr(self, "new_matrix"):
            self.new_matrix = self.source_image
        elif self.new_matrix.shape != self.source_image.shape:
            self.new_matrix = self.source_image
        else:
            self.video_merge_mode_functions_dict[self.video_merge_mode]()
        self.current_iteration_n = 0
        self.alternate_counter = 0

    def output_start_image_schnock_experiment(self):
        schnock_experiment = SchnockExperiment()
        schnock_experiment.pass_source_image(source_image=self.source_image)
        schnock_experiment.compute_new_matrix()
        self.new_matrix = schnock_experiment.new_matrix
        self.current_iteration_n = 0
        self.alternate_counter = 0

    def merge_average(self):
        logging.debug("Merge mode: Avg")
        self.new_matrix = np.average(
            self.difference_matrix, axis=0).astype(np.uint8)
        return

    def merge_min(self):
        logging.debug("Merge mode: Min")
        self.new_matrix = np.min(
            self.difference_matrix, axis=0).astype(np.uint8)
        return

    def merge_max(self):
        logging.debug("Merge mode: Max")
        self.new_matrix = np.max(
            self.difference_matrix, axis=0).astype(np.uint8)
        return

    def merge_sum(self):
        logging.debug("Merge mode: Sum")
        self.new_matrix = np.sum(
            self.difference_matrix, axis=0).astype(np.uint8)
        return

    def merge_alternate(self):
        md = self.alternate_counter // self.alternate_every_n
        if md == 0:
            self.merge_max()
        # elif md==1:
        #     self.merge_average()
        elif md == 1:
            self.merge_min()
        else:
            logging.error(md)
            logging.error(self.alternate_counter)
            raise Exception("Error modulo")
        self.alternate_counter += 1
        md = self.alternate_counter // self.alternate_every_n
        if md == 2:
            self.alternate_counter = 0

    def merge_average_video(self):
        logging.debug("Merge mode: Avg")
        self.new_matrix = np.average(
            np.stack((self.new_matrix, self.source_image)), axis=0
        ).astype(np.uint8)
        return

    def merge_min_video(self):
        logging.debug("Merge mode: Min")
        self.new_matrix = np.min(
            np.stack((self.new_matrix, self.source_image)), axis=0
        ).astype(np.uint8)
        return

    def merge_max_video(self):
        logging.debug("Merge mode: Max")
        self.new_matrix = np.max(
            np.stack((self.new_matrix, self.source_image)), axis=0
        ).astype(np.uint8)
        return

    def merge_sum_video(self):
        logging.debug("Merge mode: Sum")
        self.new_matrix = np.sum(
            np.stack((self.new_matrix, self.source_image)), axis=0
        ).astype(np.uint8)
        return

    def set_merge_mode(self, new_merge_mode: str):
        if new_merge_mode not in self.merge_mode_functions_dict.keys():
            raise Exception(
                "Invalid merge_mode ({0}), merge mode should be in {1}".format(
                    new_merge_mode, self.merge_mode_functions_dict.keys()
                )
            )
        self.merge_mode = new_merge_mode
        pass

    def set_video_merge_mode(self, new_merge_mode: str):
        if new_merge_mode not in self.video_merge_mode_functions_dict.keys():
            raise Exception(
                "Invalid merge_mode ({0}), merge mode should be in {1}".format(
                    new_merge_mode, self.merge_mode_functions_dict.keys()
                )
            )
        self.video_merge_mode = new_merge_mode
        pass

    def set_multiplier_mode(self, new_multiplier_mode: str):
        if new_multiplier_mode not in self.dynamic_multpiplier_functions_dict.keys():
            raise Exception(
                "Invalid merge_mode ({0}), merge mode should be in {1}".format(
                    new_multiplier_mode, self.dynamic_multpiplier_functions_dict.keys()
                )
            )
        self.multiplier_mode = new_multiplier_mode
        pass

    def set_multiplier_amplitude(self, new_amplitude: float):
        self.multiplier_amplitude = new_amplitude

    def set_multiplier_frequency(self, new_frequency: float):
        self.multiplier_frequency = new_frequency

    def set_alternate_every_n(self, new_value):
        self.alternate_every_n = new_value
        self.alternate_counter = 0

    def dynamic_multiplier_constant(self):
        self.dynamic_multiplier = self.multiplier_amplitude

    def dynamic_multiplier_linear_reduction(self):
        self.dynamic_multiplier = self.multiplier_amplitude * (
            1 - 1 / self.current_iteration_n
        )

    def dynamic_multiplier_exponential_reduction(self):
        self.dynamic_multiplier = 1 + int(
            self.multiplier_amplitude * np.exp(-self.current_iteration_n)
        )

    def dynamic_multiplier_cos(self):
        self.dynamic_multiplier = int(
            self.multiplier_amplitude
            * cos(2 * pi * self.current_iteration_n / self.multiplier_frequency)
        )

    def dynamic_multiplier_sin(self):
        self.dynamic_multiplier = int(
            self.multiplier_amplitude
            * sin(2 * pi * self.current_iteration_n / self.multiplier_frequency)
        )


# %%
if __name__ == "__main__":
    experiment = GradientExperiment()
    # TODO: Implement launching from command line
