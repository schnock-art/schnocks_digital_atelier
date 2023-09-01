"""Gradient Experiment Module, specific experiment for Schnock UI
"""
# %%
# Standard imports
import logging
from math import sin, cos, pi

import numpy as np


from experiment_classes.base_experiment import BaseExperiment
from experiment_classes.original_schnock import SchnockExperiment


class GradientExperiment(BaseExperiment):
    """Gradient Experiment. This class computes the gradients in x+,x-,y+ and y-.
    Starting witha  blank image (e.g. a gray image), new matrixes are computed with these gradient amtrixes and merged.

    Args:
        BaseExperiment (BaseExperiment): BaseClass.
    """

    def __init__(self):
        """Initializes class attributes"""
        super().__init__()
        self.source_diff = {}
        self.source_image = None
        self.difference_matrix = None
        self.matrix_list = []
        self.padded_source_image = {}
        self.config["current_iteration_n"] = 0
        self.config["alternate_counter"] = 0
        self.config["multiplier_amplitude"] = 1
        self.config["multiplier_frequency"] = 4
        self.config["fast_forward_iterations"] = 23
        self.config["grade"] = 1
        self.config["clip_images"] = False
        self.config["alternate_every_n"] = 4
        self.config["dynamic_multipler"] = 1
        self.config["multiplier_mode"] = "constant"
        self.config["video_merge_mode"] = "alternate"
        self.config["merge_mode"] = "alternate"
        self.initialize_dynamic_multiplier_dict()
        self.initialize_merge_mode_dict()
        self.initialize_video_merge_mode_dict()
        self.initialize_output_start_images_dict()

    # ä#### Process Image Methods

    def compute_new_matrix(self):
        self.dynamic_multpiplier_functions_dict[self.config["multiplier_mode"]](
        )
        logging.debug(
            "Dynamic multiplier: {0}".format(self.config["dynamic_multiplier"])
        )
        matrix_stack = []
        for n in range(1, self.config["grade"] + 1):
            padded_n = np.pad(
                self.new_matrix, pad_width=((n, n), (n, n), (0, 0)), mode="edge"
            )
            positive_x = padded_n[n:-n, (n + 1):, :] + self.config[
                "dynamic_multiplier"
            ] * np.negative(self.source_diff["x_{0}".format(n)][:, n:, :])
            negative_x = (
                padded_n[n:-n, : -(n + 1), :]
                + self.config["dynamic_multiplier"]
                * self.source_diff["x_{0}".format(n)][:, :-n, :]
            )
            positive_y = padded_n[(n + 1):, n:-n, :] + self.config[
                "dynamic_multiplier"
            ] * np.negative(self.source_diff["y_{0}".format(n)][n:, :])
            negative_y = (
                padded_n[0: -(n + 1), n:-n, :]
                + self.config["dynamic_multiplier"]
                * self.source_diff["y_{0}".format(n)][:-n, :]
            )
            if self.config["clip_images"] is True:
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
        self.merge_mode_functions_dict[self.config["merge_mode"]]()
        self.config["current_iteration_n"] += 1

    def get_difference_matrices(self):
        """Computes the gradient matrices for source image"""
        self.padded_source_image = {}
        for n in range(1, self.config["grade"] + 1):
            self.padded_source_image[n] = np.pad(
                self.source_image, pad_width=((n, n), (n, n), (0, 0)), mode="edge"
            )
            self.source_diff["x_{0}".format(n)] = np.diff(
                self.padded_source_image[n][n:-n, :, :], n=n, axis=1
            )
            self.source_diff["y_{0}".format(n)] = np.diff(
                self.padded_source_image[n][:, n:-n, :], n=n, axis=0
            )

    def process_source_image(self):
        self.get_difference_matrices()
        self.ouput_start_image_gray()
        iteration = 0
        while iteration < self.config["fast_forward_iterations"]:
            self.compute_new_matrix()
            iteration += 1
        self.image_list = [self.new_matrix]
        pass

    def compute_from_video(self, source_image):
        """Computes new matrix for video input

        Args:
            source_image (_type_): Image from video input
        """
        self.pass_source_image(source_image=source_image)
        self.pass_output_start_image(output_start_image=source_image)

    # Auxiuliary Methods
    def pass_output_start_image(self, output_start_image):
        """Passs ouput_start image

        Args:
            output_start_image (_type_): _description_
        """
        self.new_matrix = output_start_image

    # Image Merge modes
    def merge_average(self):
        """Merge with average"""
        logging.debug("Merge mode: Avg")
        self.new_matrix = np.average(
            self.difference_matrix, axis=0).astype(np.uint8)
        return

    def merge_min(self):
        """Merge with min"""
        logging.debug("Merge mode: Min")
        self.new_matrix = np.min(
            self.difference_matrix, axis=0).astype(np.uint8)
        return

    def merge_max(self):
        """Merge with max"""
        logging.debug("Merge mode: Max")
        self.new_matrix = np.max(
            self.difference_matrix, axis=0).astype(np.uint8)
        return

    def merge_sum(self):
        """Merge with sum"""
        logging.debug("Merge mode: Sum")
        self.new_matrix = np.sum(
            self.difference_matrix, axis=0).astype(np.uint8)
        return

    def merge_alternate(self):
        """Merge alternating between min and max"""
        mod = self.config["alternate_counter"] // self.config["alternate_every_n"]
        if mod == 0:
            self.merge_max()
        # elif md==1:
        #     self.merge_average()
        elif mod == 1:
            self.merge_min()
        else:
            logging.error(mod)
            logging.error(self.config["alternate_counter"])
            raise ValueError("Error modulo")
        self.config["alternate_counter"] += 1
        mod = self.config["alternate_counter"] // self.config["alternate_every_n"]
        if mod == 2:
            self.config["alternate_counter"] = 0

    # Video merge mode
    def merge_average_video(self):
        """Merge video with average"""
        logging.debug("Merge mode: Avg")
        self.new_matrix = np.average(
            np.stack((self.new_matrix, self.source_image)), axis=0
        ).astype(np.uint8)
        return

    def merge_min_video(self):
        """Merge video with min"""
        logging.debug("Merge mode: Min")
        self.new_matrix = np.min(
            np.stack((self.new_matrix, self.source_image)), axis=0
        ).astype(np.uint8)
        return

    def merge_max_video(self):
        """Merge video with max"""
        logging.debug("Merge mode: Max")
        self.new_matrix = np.max(
            np.stack((self.new_matrix, self.source_image)), axis=0
        ).astype(np.uint8)
        return

    def merge_sum_video(self):
        """Merge video with sum"""
        logging.debug("Merge mode: Sum")
        self.new_matrix = np.sum(
            np.stack((self.new_matrix, self.source_image)), axis=0
        ).astype(np.uint8)
        return

    # Initialize Dictionaries
    def initialize_dynamic_multiplier_dict(self):
        """ " initializes dynamic multiplier options dictionarys"""
        self.dynamic_multpiplier_functions_dict = {
            "constant": self.dynamic_multiplier_constant,
            # "current_iteration_n": self.dynamic_multiplier_linear_reduction,
            "exponential_reduction": self.dynamic_multiplier_exponential_reduction,
            "cosinus": self.dynamic_multiplier_cos,
            "sinus": self.dynamic_multiplier_sin,
        }

    def initialize_merge_mode_dict(self):
        """ " initializes merge mode options dictionarys"""
        self.merge_mode_functions_dict = {
            "average": self.merge_average,
            "min": self.merge_min,
            "max": self.merge_max,
            "sum": self.merge_sum,
            "alternate": self.merge_alternate,
        }

    def initialize_output_start_images_dict(self):
        """ " initializes output start image methods dictionarys"""
        self.output_start_mode_dict = {
            "gray": self.ouput_start_image_gray,
            "black": self.ouput_start_image_black,
            "white": self.ouput_start_image_white,
            "source_average": self.ouput_start_image_source_average,
            "schnock_experiment": self.output_start_image_schnock_experiment,
            "video": self.output_start_image_current_source_image,
        }

    def initialize_video_merge_mode_dict(self):
        """ " initializes video merge mode options dictionarys"""
        self.video_merge_mode_functions_dict = {
            "average": self.merge_average_video,
            "min": self.merge_min_video,
            "max": self.merge_max_video,
            "sum": self.merge_sum_video,
        }

    # Set Attributes
    def set_clip_images(self, value: bool):
        """sets if output image should be clipped

        Args:
            value (bool):
        """
        self.config["clip_images"] = value

    # set output start image
    def ouput_start_image_gray(self):
        """Sets output start image to gray"""
        self.new_matrix = np.full(
            shape=self.source_image.shape, fill_value=127, dtype=np.uint8
        )
        self.config["current_iteration_n"] = 0
        self.config["alternate_counter"] = 0

    def ouput_start_image_black(self):
        """Sets output start image to black"""
        self.new_matrix = np.full(
            shape=self.source_image.shape, fill_value=0, dtype=np.uint8
        )
        self.config["current_iteration_n"] = 0
        self.config["alternate_counter"] = 0

    def ouput_start_image_white(self):
        """Sets output start image to white"""
        self.new_matrix = np.full(
            shape=self.source_image.shape, fill_value=255, dtype=np.uint8
        )
        self.config["current_iteration_n"] = 0
        self.config["alternate_counter"] = 0

    def ouput_start_image_custom(self):
        """Sets output start image to custom color"""
        self.new_matrix = np.full(
            shape=self.source_image.shape, fill_value=127, dtype=np.uint8
        )
        self.config["current_iteration_n"] = 0
        self.config["alternate_counter"] = 0

    def ouput_start_image_source_average(self):
        """Sets output start image to source average color"""
        mean_pixel = np.mean(self.source_image, axis=(0, 1)
                             ).round().astype(np.uint8)
        self.new_matrix = np.full(
            shape=self.source_image.shape, fill_value=mean_pixel, dtype=np.uint8
        )
        self.config["current_iteration_n"] = 0
        self.config["alternate_counter"] = 0

    def output_start_image_current_source_image(self):
        """Sets output start image to current source image"""
        if not hasattr(self, "new_matrix"):
            self.new_matrix = self.source_image
        elif self.new_matrix.shape != self.source_image.shape:
            self.new_matrix = self.source_image
        else:
            self.video_merge_mode_functions_dict[self.config["video_merge_mode"]](
            )
        self.config["current_iteration_n"] = 0
        self.config["alternate_counter"] = 0

    def output_start_image_schnock_experiment(self):
        """Sets output start image to Schnock experiment with current parameters"""
        schnock_experiment = SchnockExperiment()
        schnock_experiment.pass_source_image(source_image=self.source_image)
        schnock_experiment.compute_new_matrix()
        self.new_matrix = schnock_experiment.new_matrix
        self.config["current_iteration_n"] = 0
        self.config["alternate_counter"] = 0

    # Set merge modes
    def set_merge_mode(self, new_merge_mode: str):
        """Sets merge mode for images

        Args:
            new_merge_mode (str): Merge mode, should be in the merge mode dictionary keys

        Raises:
            ValueError: Invalid merge mode
        """
        if new_merge_mode not in self.merge_mode_functions_dict:
            raise ValueError(
                "Invalid merge_mode ({0}), merge mode should be in {1}".format(
                    new_merge_mode, self.merge_mode_functions_dict.keys()
                )
            )
        self.config["merge_mode"] = new_merge_mode

    def set_video_merge_mode(self, new_merge_mode: str):
        """Sets merge mode for video

        Args:
            new_merge_mode (str): Merge mode, should be in the merge mode dictionary keys

        Raises:
            ValueError: Invalid merge mode
        """
        if new_merge_mode not in self.video_merge_mode_functions_dict:
            raise ValueError(
                "Invalid merge_mode ({0}), merge mode should be in {1}".format(
                    new_merge_mode, self.merge_mode_functions_dict.keys()
                )
            )
        self.config["video_merge_mode"] = new_merge_mode

    def set_multiplier_mode(self, new_multiplier_mode: str):
        """Set multiplier mode

        Args:
            new_multiplier_mode (str):  mode, should be in the multiplier mode dictionary keys

        Raises:
            ValueError: Invalid multiplier mode
        """
        if new_multiplier_mode not in self.dynamic_multpiplier_functions_dict:
            raise ValueError(
                "Invalid multiplier mode ({0}), merge mode should be in {1}".format(
                    new_multiplier_mode, self.dynamic_multpiplier_functions_dict.keys()
                )
            )
        self.config["multiplier_mode"] = new_multiplier_mode

    def set_multiplier_amplitude(self, new_amplitude: float):
        """Sets multiplier amplitude

        Args:
            new_amplitude (float): New multiplier amplitude
        """
        self.config["multiplier_amplitude"] = new_amplitude

    def set_multiplier_frequency(self, new_frequency: float):
        """Sets multiplier frequency

        Args:
            new_frequency (float): Sets new frequency
        """
        self.config["multiplier_frequency"] = new_frequency

    def set_alternate_every_n(self, new_value: int):
        """Sets altenate every n

        Args:
            new_value (int): New value for alternazte every_n
        """
        self.config["alternate_every_n"] = new_value
        self.config["alternate_counter"] = 0

    def dynamic_multiplier_constant(self):
        """Sets new dynamic multiplier as constant"""
        self.config["dynamic_multiplier"] = self.config["multiplier_amplitude"]

    def dynamic_multiplier_linear_reduction(self):
        """Sets dynamic multiplier with linear reduction (must be revised)"""
        self.config["dynamic_multiplier"] = self.config["multiplier_amplitude"] * (
            1 - 1 / self.config["current_iteration_n"]
        )

    def dynamic_multiplier_exponential_reduction(self):
        """Sets dynamic multiplier with exponential reduction (must be revised)"""
        self.config["dynamic_multiplier"] = 1 + int(
            self.config["multiplier_amplitude"]
            * np.exp(-self.config["current_iteration_n"])
        )

    def dynamic_multiplier_cos(self):
        """Sets dynamic multiplier with cosinus function"""
        self.config["dynamic_multiplier"] = int(
            self.config["multiplier_amplitude"]
            * cos(
                2
                * pi
                * self.config["current_iteration_n"]
                / self.config["multiplier_frequency"]
            )
        )

    def dynamic_multiplier_sin(self):
        """Sets dynamic multiplier with sinus function"""
        self.config["dynamic_multiplier"] = int(
            self.config["multiplier_amplitude"]
            * sin(
                2
                * pi
                * self.config["current_iteration_n"]
                / self.config["multiplier_frequency"]
            )
        )


# %%
if __name__ == "__main__":
    experiment = GradientExperiment()
    # TODO: Implement launching from command line
