# %%

from experiment_classes.base_experiment import BaseExperiment
import numpy as np
import cv2
from multiprocessing.pool import ThreadPool
from datetime import datetime


class GaborFilterExperiment(BaseExperiment):
    def __init__(self):
        super().__init__()
        self.config = {}
        self.config["kernel_size"] = 51
        self.config["sigma"] = 4.0
        self.config["lambda"] = 10.0
        self.config["gamma"] = 0.5
        self.config["psi"] = 0.5
        self.config["kernel_type"] = "CV_32F"
        self.config["mode"] = 1
        self.config["thread_nr"] = 8
        self.config["gabor_mode"] = 1
        self.set_ktypes_dict()
        self.set_gabor_mode_dict()
        self.build_filters()

    def set_ktypes_dict(self):
        self.config["kernel_types_dict"] = {
            "CV_32F": cv2.CV_32F,
            "CV_64F": cv2.CV_64F,
        }

    def set_gabor_mode_dict(self):
        self.config["gabor_mode_dict"] = {
            "1": 1,
            "-1": -1,
        }

    def build_filters(self):
        # print(self.config)
        self.filters = []
        for theta in np.arange(0, np.pi, np.pi / 16):
            kern = cv2.getGaborKernel(
                (self.config["kernel_size"], self.config["kernel_size"]),
                sigma=self.config["sigma"],  # 4.0
                theta=theta,
                lambd=self.config["lambda"],  # 10.0
                gamma=self.config["gamma"],  # 0.5
                psi=self.config["psi"],  # np.pi * 0.5,
                # cv2.CV_32F
                ktype=self.config["kernel_types_dict"][self.config["kernel_type"]],
            )
            kern /= 1.5*kern.sum()

            self.filters.append(kern)

    def compute_new_matrix(self):
        self.build_filters()
        self.new_matrix = np.zeros_like(self.source_image)

        def f(kern):
            return cv2.filter2D(self.source_image, cv2.CV_8UC3, kern)
        pool = ThreadPool(processes=self.config["thread_nr"])
        for fimg in pool.imap_unordered(f, self.filters):
            if self.config["gabor_mode"] == 1:
                np.maximum(self.new_matrix, fimg, self.new_matrix)
            elif self.config["gabor_mode"] == -1:
                np.minimum(self.new_matrix, fimg, self.new_matrix)
            else:
                np.maximum(self.new_matrix, fimg, self.new_matrix)

    def set_kernel_size(self, new_kernel_size):
        self.config["kernel_size"] = new_kernel_size
        # self.build_filters()

    def set_sigma(self, new_sigma):
        self.config["sigma"] = new_sigma
        # self.build_filters()

    def set_lambda(self, new_lambda):
        self.config["lambda"] = new_lambda
        # self.build_filters()

    def set_gamma(self, new_gamma):
        self.config["gamma"] = new_gamma
        # self.build_filters()

    def set_psi(self, new_psi):
        self.config["psi"] = new_psi
        # self.build_filters()

    def set_kernel_type(self, new_kernel_type):
        self.config["kernel_type"] = new_kernel_type
        # self.build_filters()

    def set_gabor_mode(self, new_gabor_mode):
        self.config["gabor_mode"] = self.config["gabor_mode_dict"][new_gabor_mode]

    def set_thread_nr(self, new_thread_nr):
        self.config["thread_nr"] = new_thread_nr
