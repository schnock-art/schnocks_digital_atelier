""" Module containing BaseClass for other experiments
"""
import os
import cv2
import numpy as np
import logging
import json

from .utils import NpEncoder


class BaseExperiment:
    """Baseclass for Image Experiments"""

    def __init__(self):
        self.input_directory = None
        self.source_image_path = None
        self.source_folder_path = None
        self.source_image = None
        self.new_matrix = None
        self.output_directory = None
        self.file_extension = None
        self.input_diretory_files_list = None
        self.config = {}
        self.valid_file_extensions = [
            ".JPG",
            ".PNG"
        ]

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

    def set_input_directory(self, input_directory: str):
        self.input_directory = os.path.abspath(input_directory)
        if not os.path.isdir(self.input_directory):
            raise Exception(
                "Input directory {0} is not a directory!".format(self.input_directory))

    def set_output_directory(self, output_directory: str = None):
        if output_directory is None:
            logging.debug("Setting default output directory")
            self.output_directory = self.input_directory + "_automatic_processed"
        else:
            self.output_directory = os.path.abspath(output_directory)

    def create_output_directory(self):
        try:
            os.makedirs(self.output_directory)
        except FileExistsError:
            logging.error("Output directory already exists!")
            raise

    def set_file_extension(self, file_extension: str):

        if file_extension is None:
            if self.file_extension is None:
                raise Exception("Must specify a file extension!")
        else:
            file_extension = file_extension.upper()
            if file_extension not in self.valid_file_extensions:
                raise Exception(
                    "{0} Not a valid file extension!".format(file_extension))
            self.file_extension = file_extension

    def process_folder(self,
                       input_directory: str = None,
                       output_directory: str = None,
                       file_extension: str = None,
                       ):
        if input_directory is not None:
            self.set_input_directory(input_directory=input_directory)
        if self.input_directory is None:
            raise Exception("Must specify a directory to process images!")

        logging.info("Folder to process: {0}".format(self.input_directory))

        self.set_output_directory(output_directory=output_directory)
        self.create_output_directory()

        self.set_file_extension(file_extension=file_extension)

        self.get_all_images_in_input_directory()

        self.save_config_json(folder=True)

        for path in self.input_diretory_files_list:
            self.process_path(path=path)
        # Parallel(n_jobs=num_cores)(delayed(self.process_path)(path=i) for i in self.input_diretory_files_list)
        # end_time=datetime.now()
        # print("Total time: {0}".format(end_time-start_time))

    def get_all_images_in_input_directory(self):
        self.input_diretory_files_list = []
        for address, dirs, files in os.walk(self.input_directory):
            for filename in files:
                if filename.endswith(self.file_extension):
                    self.input_diretory_files_list.append(
                        os.path.join(address, filename))

    def process_path(self, path: str, output_path: str = None, save_config: bool = False):
        self.load_source_image(source_image_path=path)
        self.compute_new_matrix()
        self.set_output_path(output_path=output_path)
        self.save_output_image(save_config=save_config)

    def set_output_path(self, output_path: str = None):
        if output_path is None:
            self.output_image_path = self.source_image_path.replace(
                self.input_directory, self.output_directory)
        else:
            self.output_image_path = os.path.abspath(output_path)

    def save_output_image(self, save_config: bool = False):

        os.makedirs(os.path.dirname(self.output_image_path), exist_ok=True)

        if os.path.exists(self.output_image_path):
            raise Exception("Output image already exists!")

        self.output_image_path_wo_extension = os.path.splitext(self.output_image_path)[
            0]

        cv2.imwrite(self.output_image_path, self.new_matrix)

        if save_config:
            self.save_config_json()

    def save_config_json(self, folder: bool = False):
        if folder:
            self.config_path = os.path.join(
                self.output_directory, "experiment_config.json")
        else:
            filename_wo_ext = os.path.splitext(self.source_image_path)[0]
            self.config_path = "{0}_config.json".format(filename_wo_ext)
        with open(self.config_path, "w") as outfile:
            json.dump(self.config, outfile, indent=4, cls=NpEncoder)


# %%


# %%
