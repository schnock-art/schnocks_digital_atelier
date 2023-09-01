# %%
# Kmeans Experiment Class
# This class is used to run the kmeans experiment. It inherits from the Experiment class.
# %%
import numpy as np
import cv2
from sklearn.cluster import KMeans
from experiment_classes.base_experiment import BaseExperiment

import logging


class KmeansExperiment(BaseExperiment):
    def __init__(self):
        super().__init__()
        self.config = {}
        self.config["scaling_factor"] = 5
        self.config["number_of_pyramids"] = 3
        self.config["number_of_clusters"] = 200
        self.config["number_of_iterations"] = 10
        self.config["number_of_repetitions"] = 5
        self.config["init_types"] = [
            "random",
            "k-means++",
        ]
        self.config["init_type"] = "random"

    def process_source_image(self):
        self.create_new_array_with_coordinates()
        self.scale_down_image()
        self.flatten_image()
        self.create_training_data()
        self.create_prediction_data()
        self.compute_new_matrix()
        return

    def create_new_array_with_coordinates(self):
        # Create new array with coordinates
        logging.debug("Creating new array with coordinates")
        self.shape = self.source_image.shape
        self.shape_x = self.shape[0]
        self.shape_y = self.shape[1]
        ax = np.arange(self.shape_x)
        ax = ax/self.shape_x*self.config["scaling_factor"]
        ay = np.arange(self.shape_y)
        ay = ay/self.shape_y*self.config["scaling_factor"]
        self.xx, self.yy = np.meshgrid(ay, ax)

    def scale_down_image(self):
        # Scale down image
        logging.debug("Scaling down image")
        self.downsized = {}
        self.downsized["source_image"] = self.source_image.copy()
        self.downsized["xx"] = self.xx.copy()
        self.downsized["yy"] = self.yy.copy()
        self.downsized["shape"] = {}
        self.downsized["shape"]["x"] = self.downsized["source_image"].shape[0]
        self.downsized["shape"]["y"] = self.downsized["source_image"].shape[1]

        for i in range(self.config["number_of_pyramids"]):
            self.downsized["xx"] = cv2.pyrDown(
                self.downsized["xx"],
                dstsize=(self.downsized["shape"]["y"] // 2,
                         self.downsized["shape"]["x"] // 2))
            self.downsized["yy"] = cv2.pyrDown(
                self.downsized["yy"],
                dstsize=(self.downsized["shape"]["y"] // 2,
                         self.downsized["shape"]["x"] // 2))
            self.downsized["source_image"] = cv2.pyrDown(
                self.downsized["source_image"],
                dstsize=(self.downsized["shape"]["y"] // 2,
                         self.downsized["shape"]["x"] // 2))
            self.downsized["shape"]["x"] = self.downsized["source_image"].shape[0]
            self.downsized["shape"]["y"] = self.downsized["source_image"].shape[1]

    def flatten_image(self):
        # Flatten downsized image
        logging.debug("Flattening downsized image")
        self.flattened = {}
        self.flattened["downsized_source_image"] = self.downsized["source_image"].reshape(
            self.downsized["shape"]["x"]*self.downsized["shape"]["y"], 3
        )/255
        self.flattened["downsized_xx"] = self.downsized["xx"].reshape(
            self.downsized["shape"]["x"]*self.downsized["shape"]["y"], 1
        )
        self.flattened["downsized_yy"] = self.downsized["yy"].reshape(
            self.downsized["shape"]["x"]*self.downsized["shape"]["y"], 1
        )

        # Flatten source image
        logging.debug("Flattening source image")
        self.flattened["source_image"] = self.source_image.reshape(
            self.shape_x*self.shape_y, 3
        )/255
        self.flattened["xx"] = self.xx.reshape(
            self.shape_x*self.shape_y, 1
        )
        self.flattened["yy"] = self.yy.reshape(
            self.shape_x*self.shape_y, 1
        )

    def create_training_data(self):
        logging.debug("Creating training data")
        self.training_data = np.append(
            self.flattened["downsized_source_image"],
            self.flattened["downsized_xx"],
            axis=1,
        )
        self.training_data = np.append(
            self.training_data, self.flattened["downsized_yy"], axis=1
        )

    def create_prediction_data(self):
        logging.debug("Creating prediction data")
        self.prediction_data = np.append(
            self.flattened["source_image"], self.flattened["xx"], axis=1
        )
        self.prediction_data = np.append(
            self.prediction_data, self.flattened["yy"], axis=1
        )

    def compute_new_matrix(self):
        logging.debug("Computing new matrix")
        # Run kmeans
        logging.debug("Running kmeans")
        try:
            self.config["kmeans_model"] = KMeans(
                n_clusters=self.config["number_of_clusters"],
                init=self.config["init_type"],
                n_init=self.config["number_of_repetitions"],
                max_iter=self.config["number_of_iterations"],
            ).fit(self.training_data)

            # Get cluster centers
            self.cluster_centers = self.config["kmeans_model"].cluster_centers_
            logging.debug("Predicting new image")
            self.new_matrix = self.config["kmeans_model"].predict(
                self.prediction_data)
            # Assign cluster centers to image
            logging.debug("Assigning cluster centers to image")
            self.new_matrix = np.array([self.cluster_centers[center_index]
                                        for center_index
                                        in self.new_matrix])
            self.new_matrix = self.new_matrix.reshape(
                self.shape_x,
                self.shape_y, 5
            )[:, :, :-2]*255
            self.new_matrix = self.new_matrix.astype(np.uint8)
        except Exception as error:
            logging.error(error)
        return

    def set_number_of_clusters(self, new_number_of_clusters: int):
        self.config["number_of_clusters"] = new_number_of_clusters
        return

    def set_number_of_iterations(self, new_number_of_iterations: int):
        self.config["number_of_iterations"] = new_number_of_iterations
        return

    def set_number_of_repetitions(self, new_number_of_repetitions: int):
        self.config["number_of_repetitions"] = new_number_of_repetitions
        return

    def set_init_type(self, new_init_type: str):
        self.config["init_type"] = new_init_type
        return

    def set_scaling_factor(self, new_scaling_factor: int):
        self.config["scaling_factor"] = new_scaling_factor
        return

    def set_number_of_pyramids(self, new_number_of_pyramids: int):
        self.config["number_of_pyramids"] = new_number_of_pyramids
        return

    def set_kmeans_init_type(self, new_init_type: str):
        self.config["init_type"] = new_init_type
        return
# %%
# Running Standalone


# %%
# This is the main function that runs the experiment. It is called from the command line.
# %%
if __name__ == "__main__":
    experiment = KmeansExperiment()
    experiment.process_path(
        path=r"C:\Users\jange\Python Scripts\schnocks_digital_atelier\data\DSC03572.JPG",
        output_path=r"C:\Users\jange\Python Scripts\schnocks_digital_atelier\data\DSC03572_kmeans.JPG",
    )
