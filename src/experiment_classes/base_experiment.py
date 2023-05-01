
import os
import cv2
import numpy as np

class BaseExperiment:
    def __init__(self):
        pass

    def load_source_image(self, 
        source_folder_path: str=None, 
        source_image_path: str=None
    ):

        if source_image_path is None:
            raise Exception("Must provide an image path")
        
        self.source_image_path=os.path.normpath(source_image_path)
        if source_folder_path is not None:
            self.source_folder_path=os.path.normpath(source_folder_path)
            self.source_image_path= os.path.join(self.source_folder_path,self.source_image_path)
        
        if not os.path.isfile(self.source_image_path):
            raise Exception("Image + folder is not a valid file: {0}".format(self.source_image_path))

        self.source_image = cv2.imread(self.source_image_path)
        self.source_image = cv2.cvtColor(self.source_image, cv2.COLOR_BGR2RGB)
        pass

    def pass_source_image(self, source_image: np.array):
        self.source_image = source_image
        if self.__class__.__name__=="GradientExperiment":
            self.get_difference_matrices()
        pass

    def compute_new_matrix(self):
        pass