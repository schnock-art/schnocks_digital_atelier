""" Module containing BaseClass for other experiments
"""
import os
import cv2
import numpy as np


class BaseExperiment:
    """Baseclass for Image Experiments"""

    def __init__(self):
        self.source_image_path = None
        self.source_folder_path = None
        self.source_image = None
        self.new_matrix = None

    def load_source_image(
        self, source_folder_path: str = None, source_image_path: str = None
    ):
        """Loads source image to classs

        Args:
            source_folder_path (str, optional): Folder where image is. Defaults to None.
            source_image_path (str, optional): Path to image. Defaults to None.

        """
        if source_image_path is None:
            raise ValueError("Must provide an image path")

        self.source_image_path = os.path.normpath(source_image_path)
        if source_folder_path is not None:
            self.source_folder_path = os.path.normpath(source_folder_path)
            self.source_image_path = os.path.join(
                self.source_folder_path, self.source_image_path
            )

        if not os.path.isfile(self.source_image_path):
            raise TypeError(
                "Image + folder is not a valid file: {0}".format(
                    self.source_image_path)
            )

        self.source_image = cv2.imread(self.source_image_path)
        self.source_image = cv2.cvtColor(self.source_image, cv2.COLOR_BGR2RGB)

    def pass_source_image(self, source_image: np.array):
        """Passes new source image to class.
        If Class is GradientExperiment, computes the difference matrices
        """
        self.source_image = source_image
        if self.__class__.__name__ == "GradientExperiment":
            self.get_difference_matrices()

    def compute_new_matrix(self):
        """Placeholder metzhod to create new computed matrix"""
