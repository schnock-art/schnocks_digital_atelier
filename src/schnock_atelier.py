""" Module containing UI class and executing ui
"""
# %%
# standard imports
from experiment_classes.original_schnock import SchnockExperiment
from experiment_classes.gradient_experiment import GradientExperiment
from experiment_classes.gabor_filter_experiment import GaborFilterExperiment
from experiment_classes.kmeans_experiment import KmeansExperiment
import sys
import time
import logging
import os

from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QPushButton,
    QRadioButton,
    QMainWindow,
    QLabel,
    QErrorMessage,
    QSlider,
    QComboBox,
    QSpinBox,
    QDoubleSpinBox,
    QTabWidget,
    QLCDNumber,
    QInputDialog
)

from PyQt6 import uic
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import pyqtSignal, pyqtSlot, QThread, QCoreApplication
import numpy as np
import cv2

translate = QCoreApplication.translate


# Module imports


class VideoThread(QThread):
    """Video Threat for Webcam input

    Args:
        QThread (QThread):
    """

    change_pixmap_signal = pyqtSignal(np.ndarray)

    def run(self):
        """capture from web cam"""
        cap = cv2.VideoCapture(0)
        while True:
            ret, cv_img = cap.read()
            if ret:
                self.change_pixmap_signal.emit(cv_img)


class UI(QMainWindow):
    """Main UI for the application. Loads the main_ui.ui file.
    This file can be edited in the QtDesigner program, readme has additional information.
    Qtdesigner can be opened via makefile

    Args:
        QMainWindow (QMainWindow): QMainwindow base class
    """

    def __init__(self):
        try:
            super(UI, self).__init__()
            self.base_dir = os.path.dirname(__file__)
            self.ui_path = os.path.join(self.base_dir, "main_ui.ui")
            uic.loadUi(self.ui_path, self)
            logging.basicConfig(level=logging.INFO)
            self.max_img_height = 640
            self.max_img_width = 640

            self.matrix_list = []
            self.current_matrix_index = 0
            self.webcam_frame_counter = 0
            self.clip_images = False
            self.multiplier_mode = "constant"
            self.merge_mode = "mean"
            self.alternate_every_n = 5
            self.multiplier = 1
            self.webcam_delay = 1
            self.source_original_image_data = None
            self.source_image_data = None
            self.experiment = None
            self.result_image_data = None
            self.source_filename = None
            self.edit_mode_tab = None
            self.current_edit_mode_tab_name = None
            self.config = {}
            self.define_widgets()
            self.init_values()
            self.connect_buttons()
            self.show()

        except Exception as error:
            logging.error(
                "Failed initialization with error: {0}".format(error))
            raise error

    def init_values(self):
        """Initializes attributes in order to be able to load Experimetn classes"""
        self.set_edit_mode()
        if self.config["edit_mode"] == "schnock_original":
            self.init_schnock_experiment()
        elif self.config["edit_mode"] == "gradient_experiment":
            self.init_gradient_experiment()
        elif self.config["edit_mode"] == "gabor_filter_experiment":
            self.init_gabor_filter_experiment()
        elif self.config["edit_mode"] == "kmeans_experiment":
            self.init_kmeans_experiment()
        else:
            raise Exception("No edit mode specified!")

    def init_common_values(self):
        """Initializes common attribute values"""
        self.set_fast_forward_iterations()
        self.set_webcam_delay()

    def init_gradient_experiment(self):
        """Initializes a GradientExperiment and sets its attributes"""
        self.experiment = GradientExperiment()
        self.config = {}
        self.config["edit_mode"] = "gradient_experiment"
        self.set_multiplier_mode()
        self.set_merge_mode()
        self.set_multiplier_amplitude()
        self.set_multiplier_frequency()
        self.set_output_start_image_mode()
        self.set_alternate_every_n()
        self.set_video_merge_mode()
        self.config[
            "compute_on_original"
        ] = self.compute_on_original_radio_button.isChecked()
        if self.source_original_image_data is not None:
            self.set_source_image_data()
            self.reset_output_image()
        self.init_common_values()

    def init_schnock_experiment(self):
        """Initializes a GradientExperiment and sets its attributes"""
        self.experiment = SchnockExperiment()
        self.config = {}
        self.config["edit_mode"] = "schnock_experiment"
        self.set_low_shift()
        self.set_mid_shift()
        self.set_high_shift()
        self.set_low_threshold()
        self.set_mid_threshold()
        self.set_high_threshold()
        self.config[
            "compute_on_original"
        ] = self.compute_on_original_radio_button.isChecked()
        if self.source_original_image_data is not None:
            self.set_source_image_data()
        self.init_common_values()

    def init_gabor_filter_experiment(self):
        self.experiment = GaborFilterExperiment()
        self.config = {}
        self.config["edit_mode"] = "gabor_filter_experiment"
        self.set_kernel_size()
        self.set_kernel_type()
        self.set_kernel_gamma()
        self.set_kernel_lambda()
        self.set_kernel_psi()
        self.set_kernel_sigma()
        self.set_gabor_mode()
        self.config[
            "compute_on_original"
        ] = self.compute_on_original_radio_button.isChecked()
        if self.source_original_image_data is not None:
            self.set_source_image_data()
        self.init_common_values()

    def init_kmeans_experiment(self):
        self.experiment = KmeansExperiment()
        self.config = {}
        self.config["edit_mode"] = "kmeans_experiment"
        self.set_number_of_clusters()
        self.set_number_of_iterations()
        self.set_number_of_repetitions()
        self.set_number_of_pyramids()
        self.set_scaling_factor()
        self.set_kmeans_init_type()
        self.config[
            "compute_on_original"
        ] = self.compute_on_original_radio_button.isChecked()
        if self.source_original_image_data is not None:
            self.set_source_image_data()
        self.init_common_values()

    def define_widgets(self):
        """Defines widgets as class attributes, these are read from the main_ui.ui file"""
        logging.debug("Defining Widgets")
        self.define_general_widgets()
        self.define_gradient_experiment_widgets()
        self.define_schnock_experiment_widgets()
        self.define_gabor_experiment_widgets()
        self.define_kmeans_experiment_widgets()

    def define_general_widgets(self):
        """Defines general widgets, these are read from the main_ui.ui file"""
        # General
        self.define_general_pushbutton_widgets()
        self.define_general_radiobutton_widgets()
        self.fast_forward_iterations_spinbox = self.findChild(
            QSpinBox, "spinBox_fast_forward_iterations"
        )
        # Horizontal sliders
        self.webcam_delay_horizontal_slider = self.findChild(
            QSlider, "horizontalSlider_webcam_delay"
        )
        # LCD Numbers
        self.webcam_delay_lcd_number = self.findChild(
            QLCDNumber, "lcdNumber_webcam_delay"
        )
        self.define_image_widgets()
        self.define_tabs_widgets()

    def define_gradient_experiment_widgets(self):
        """Defines Gradient Experiment widgets, these are read from the main_ui.ui file"""
        # Gradient Experiment
        self.reset_output_image_button = self.findChild(
            QPushButton, "pushButton_reset_output"
        )
        self.clip_images_radio_button = self.findChild(
            QRadioButton, "radioButton_clip_images"
        )

        # Comboboxes
        # Output start image
        self.output_start_image_combobox = self.findChild(
            QComboBox, "comboBox_output_start_image"
        )
        for item in GradientExperiment().output_start_mode_dict:
            self.output_start_image_combobox.addItem(item)

        # Proccessing Modes
        self.multiplier_mode_combobox = self.findChild(
            QComboBox, "comboBox_multiplier_mode"
        )
        for item in GradientExperiment().dynamic_multpiplier_functions_dict:
            self.multiplier_mode_combobox.addItem(item)

        self.merge_mode_combobox = self.findChild(
            QComboBox, "comboBox_merge_mode")
        for item in GradientExperiment().merge_mode_functions_dict:
            self.merge_mode_combobox.addItem(item)

        self.video_merge_mode_combobox = self.findChild(
            QComboBox, "comboBox_video_merge_mode"
        )
        for item in GradientExperiment().video_merge_mode_functions_dict:
            self.video_merge_mode_combobox.addItem(item)

        # Spinboxes
        self.multiplier_amplitude_spinbox = self.findChild(
            QSpinBox, "spinBox_multiplier_amplitude"
        )
        self.multiplier_frequency_spinbox = self.findChild(
            QSpinBox, "spinBox_multiplier_frequency"
        )

        self.alternate_every_n_spinbox = self.findChild(
            QSpinBox, "spinBox_alternate_every_n"
        )

    def define_schnock_experiment_widgets(self):
        """Defines Schnock Experiment widgets, these are read from the main_ui.ui file"""
        # Schnock Experiment
        # Vertical sliders
        self.low_threshold_vertical_slider = self.findChild(
            QSlider, "verticalSlider_low_threshold"
        )
        self.mid_threshold_vertical_slider = self.findChild(
            QSlider, "verticalSlider_mid_threshold"
        )
        self.high_threshold_vertical_slider = self.findChild(
            QSlider, "verticalSlider_high_threshold"
        )
        # Spinboxes
        self.low_shift_spinbox = self.findChild(QSpinBox, "spinBox_low_shift")
        self.mid_shift_spinbox = self.findChild(QSpinBox, "spinBox_mid_shift")
        self.high_shift_spinbox = self.findChild(
            QSpinBox, "spinBox_high_shift")

        # LCD Numbers
        self.low_threshold_lcd_number = self.findChild(
            QLCDNumber, "lcdNumber_low_threshold")
        self.mid_threshold_lcd_number = self.findChild(
            QLCDNumber, "lcdNumber_mid_threshold")
        self.high_threshold_lcd_number = self.findChild(
            QLCDNumber, "lcdNumber_high_threshold")
        pass

    def define_gabor_experiment_widgets(self):
        """Defines Gabor Experiment widgets, these are read from the main_ui.ui file"""
        # Gabor Experiment
        # Horizontal Sliders
        self.kernel_size_horizontal_slider = self.findChild(
            QSlider, "horizontalSlider_kernel_size"
        )
        # Comboboxes
        self.kernel_type_combobox = self.findChild(
            QComboBox, "comboBox_kernel_type"
        )
        for item in GaborFilterExperiment().config["kernel_types_dict"]:
            self.kernel_type_combobox.addItem(item)

        self.gabor_mode_combobox = self.findChild(
            QComboBox, "comboBox_gabor_mode")

        for item in GaborFilterExperiment().config["gabor_mode_dict"]:
            self.gabor_mode_combobox.addItem(item)

        # DoubleSpinbox
        self.sigma_double_spinbox = self.findChild(
            QDoubleSpinBox, "doubleSpinBox_sigma")
        self.lambda_double_spinbox = self.findChild(
            QDoubleSpinBox, "doubleSpinBox_lambda")
        self.gamma_double_spinbox = self.findChild(
            QDoubleSpinBox, "doubleSpinBox_gamma")
        self.psi_double_spinbox = self.findChild(
            QDoubleSpinBox, "doubleSpinBox_psi")

        # LCD Numbers
        self.kernel_size_lcd_number = self.findChild(
            QLCDNumber, "lcdNumber_kernel_size")

        pass

    def define_kmeans_experiment_widgets(self):
        # Kmeans Experiment widgets
        # Horizontal Sliders
        self.number_of_clusters_horizontal_slider = self.findChild(
            QSlider, "horizontalSlider_number_of_clusters")

        self.number_of_iterations_horizontal_slider = self.findChild(
            QSlider, "horizontalSlider_number_of_iterations")

        self.number_of_repetitions_horizontal_slider = self.findChild(
            QSlider, "horizontalSlider_number_of_repetitions")

        self.number_of_pyramids_horizontal_slider = self.findChild(
            QSlider, "horizontalSlider_number_of_pyramids")

        # Spinboxes
        self.scaling_factor_spinbox = self.findChild(
            QDoubleSpinBox, "doubleSpinBox_scaling_factor")

        # Comboboxes
        self.kmeans_init_types_combobox = self.findChild(
            QComboBox, "comboBox_kmeans_init_types")

        for item in KmeansExperiment().config["init_types"]:
            self.kmeans_init_types_combobox.addItem(item)

        # lcd numbers
        self.number_of_clusters_lcd_number = self.findChild(
            QLCDNumber, "lcdNumber_number_of_clusters")

        self.number_of_iterations_lcd_number = self.findChild(
            QLCDNumber, "lcdNumber_number_of_iterations")

        self.number_of_repetitions_lcd_number = self.findChild(
            QLCDNumber, "lcdNumber_number_of_repetitions")

        self.number_of_pyramids_lcd_number = self.findChild(
            QLCDNumber, "lcdNumber_number_of_pyramids")

    def define_general_pushbutton_widgets(self):
        # Push Buttons
        self.load_source_image_button = self.findChild(
            QPushButton, "pushButton_load_input_image"
        )
        self.process_image_button = self.findChild(
            QPushButton, "pushButton_process_image"
        )
        self.process_folder_button = self.findChild(
            QPushButton, "pushButton_process_folder"
        )
        self.next_image_button = self.findChild(
            QPushButton, "pushButton_next_image")
        self.previous_image_button = self.findChild(
            QPushButton, "pushButton_previous_image"
        )
        self.set_output_as_input_button = self.findChild(
            QPushButton, "pushButton_set_output_as_input"
        )

        self.save_image_button = self.findChild(
            QPushButton, "pushButton_save_image")

    def define_general_radiobutton_widgets(self):
        # Radio Buttons
        self.continuous_processing_radiobutton = self.findChild(
            QRadioButton, "radioButton_continuous"
        )

        self.compute_on_original_radio_button = self.findChild(
            QRadioButton, "radioButton_compute_on_original"
        )
        self.webcam_input_radiobutton = self.findChild(
            QRadioButton, "radioButton_webcam_input"
        )

    def define_image_widgets(self):
        # Images
        self.input_image_label = self.findChild(QLabel, "Input_Image")
        self.output_image_label = self.findChild(QLabel, "Output_Image")
        self.input_image_label.setMaximumSize(
            self.max_img_width, self.max_img_height)
        self.output_image_label.setMaximumSize(
            self.max_img_width, self.max_img_height)

    def define_tabs_widgets(self):
        # Tabs
        self.edit_mode_tab = self.findChild(QTabWidget, "tabWidget_edit_mode")

        self.gradient_experiment_config_tab = self.findChild(
            QTabWidget, "tab_gradient_experiment_config")

        self.schnock_original_config_tab = self.findChild(
            QTabWidget, "tab_schnock_original_config")

        self.gabor_experiment_config_tab = self.findChild(
            QTabWidget, "tab_gabor_filter_experiment_config")

    def connect_buttons(self):
        """Connects the buttons to functions"""
        self.connect_general_buttons()
        self.connect_gradient_experiment_buttons()
        self.connect_schnock_experiment_buttons()
        self.connect_gabor_experiment_buttons()
        self.connect_kmeans_experiment_buttons()
        pass

    def connect_general_buttons(self):

        self.connect_general_pushbuttons()
        # Spinboxes
        self.fast_forward_iterations_spinbox.valueChanged.connect(
            self.set_fast_forward_iterations)

        self.connect_general_radiobuttons()

        # Horizontal Sliders
        self.webcam_delay_horizontal_slider.valueChanged.connect(
            self.set_webcam_delay)
        # Tabs
        self.edit_mode_tab.currentChanged.connect(self.set_edit_mode)

    def connect_general_pushbuttons(self):
        # Push Buttons
        self.load_source_image_button.clicked.connect(
            self.load_source_image_from_file)
        self.process_image_button.clicked.connect(self.process_image)
        self.process_folder_button.clicked.connect(self.process_folder)
        self.save_image_button.clicked.connect(self.save_as_file)
        self.set_output_as_input_button.clicked.connect(
            self.set_output_as_input)

    def connect_general_radiobuttons(self):
        # Radio Buttons
        self.compute_on_original_radio_button.toggled.connect(
            self.set_compute_on_original)
        self.webcam_input_radiobutton.toggled.connect(self.start_video)
        self.continuous_processing_radiobutton.toggled.connect(
            self.process_image)

    def connect_gradient_experiment_buttons(self):
        # Combo boxes
        self.merge_mode_combobox.currentTextChanged.connect(
            self.set_merge_mode)
        self.video_merge_mode_combobox.currentTextChanged.connect(
            self.set_video_merge_mode)
        self.multiplier_mode_combobox.currentTextChanged.connect(
            self.set_multiplier_mode)
        self.output_start_image_combobox.currentTextChanged.connect(
            self.set_output_start_image_mode)

        # Push Buttons
        self.reset_output_image_button.clicked.connect(self.reset_output_image)

        # Radiobuttons
        self.clip_images_radio_button.toggled.connect(self.set_clip_images)

        # Spinboxes
        self.multiplier_amplitude_spinbox.valueChanged.connect(
            self.set_multiplier_amplitude)
        self.multiplier_frequency_spinbox.valueChanged.connect(
            self.set_multiplier_frequency)
        self.alternate_every_n_spinbox.valueChanged.connect(
            self.set_alternate_every_n)
        pass

    def connect_schnock_experiment_buttons(self):
        # Spinboxes
        self.low_shift_spinbox.valueChanged.connect(self.set_low_shift)
        self.mid_shift_spinbox.valueChanged.connect(self.set_mid_shift)
        self.high_shift_spinbox.valueChanged.connect(self.set_high_shift)

        # Vertical Sliders
        self.low_threshold_vertical_slider.valueChanged.connect(
            self.set_low_threshold)
        self.mid_threshold_vertical_slider.valueChanged.connect(
            self.set_mid_threshold)
        self.high_threshold_vertical_slider.valueChanged.connect(
            self.set_high_threshold
        )
        pass

    def connect_gabor_experiment_buttons(self):
        # Comboboxes
        self.kernel_type_combobox.currentTextChanged.connect(
            self.set_kernel_type)
        self.gabor_mode_combobox.currentTextChanged.connect(
            self.set_gabor_mode)

        # DoubleSpinbox
        self.sigma_double_spinbox.valueChanged.connect(self.set_kernel_sigma)
        self.lambda_double_spinbox.valueChanged.connect(self.set_kernel_lambda)
        self.psi_double_spinbox.valueChanged.connect(self.set_kernel_psi)
        self.gamma_double_spinbox.valueChanged.connect(self.set_kernel_gamma)

        self.kernel_size_horizontal_slider.valueChanged.connect(
            self.set_kernel_size
        )
        pass

    def connect_kmeans_experiment_buttons(self):
        # Horizontal Sliders
        self.number_of_clusters_horizontal_slider.valueChanged.connect(
            self.set_number_of_clusters)
        self.number_of_iterations_horizontal_slider.valueChanged.connect(
            self.set_number_of_iterations)
        self.number_of_repetitions_horizontal_slider.valueChanged.connect(
            self.set_number_of_repetitions)
        self.number_of_pyramids_horizontal_slider.valueChanged.connect(
            self.set_number_of_pyramids)

        # Spinboxes
        self.scaling_factor_spinbox.valueChanged.connect(
            self.set_scaling_factor)

        # Comboboxes
        self.kmeans_init_types_combobox.currentTextChanged.connect(
            self.set_kmeans_init_type)

        pass

    @pyqtSlot(np.ndarray)
    def set_source_image_data(self):
        """Sets the source image to appropiate widget"""
        if self.config["compute_on_original"] is True:
            self.source_image_data = self.source_original_image_data
        else:
            self.source_image_data = self.resize_image(
                self.source_original_image_data)

        self.experiment.pass_source_image(source_image=self.source_image_data)

        source_image_resized = self.resize_image(self.source_image_data)
        self.input_image_label.setPixmap(
            self.pixmap_from_cv_image(source_image_resized)
        )

    def start_video(self):
        """Starts video threat, will only work if there is an available webcam"""
        if not self.webcam_input_radiobutton.isChecked():
            logging.info("Stopping video Thread")
            # self.thread.stop()
            return
        self.thread = VideoThread()
        self.thread.change_pixmap_signal.connect(
            self.update_source_image_from_webcam)

        # start the thread
        self.thread.start()
        # self.reset_output_image()

    @pyqtSlot(np.ndarray)
    def update_source_image_from_webcam(self, cv_img):
        """Updates the image_label with a new opencv image"""
        if not self.webcam_input_radiobutton.isChecked():
            logging.info("Stopping video")
            # self.thread.stop()
            return
        self.webcam_frame_counter += 1
        self.source_original_image_data = cv_img
        if self.config["compute_on_original"] is True:
            self.source_image_data = self.source_original_image_data
        else:
            self.source_image_data = self.resize_image(
                self.source_original_image_data)

        self.experiment.pass_source_image(source_image=self.source_image_data)

        source_image_resized = self.resize_image(self.source_image_data)
        self.input_image_label.setPixmap(
            self.pixmap_from_cv_image(source_image_resized)
        )

        if self.webcam_frame_counter % self.config["webcam_delay"] == 0:
            if self.config["edit_mode"] == "gradient_experiment":
                self.experiment.output_start_mode_dict["video"]()

        self.experiment.compute_new_matrix()
        self.result_image_data = self.experiment.new_matrix
        result_image_resized = self.resize_image(self.result_image_data)
        self.output_image_label.setPixmap(
            self.pixmap_from_cv_image(result_image_resized)
        )

    def load_source_image_from_file(self):
        """Opens FileDialog to load source image from file."""
        self.source_filename = QFileDialog.getOpenFileName(self, "Open File")[
            0]
        self.source_original_image_data = cv2.imread(self.source_filename)

        self.set_source_image_data()
        self.set_edit_mode()
        self.reset_output_image()

    def set_edit_mode(self):
        """Sets the edit mode to Schnock original or Gradient, will show the correct tab also"""
        self.current_edit_mode_tab_name = (
            self.edit_mode_tab.currentWidget().objectName()
        )
        if self.current_edit_mode_tab_name == "tab_schnock_original_config":
            self.init_schnock_experiment()
            self.config["edit_mode"] = "schnock_original"
        elif self.current_edit_mode_tab_name == "tab_gradient_experiment_config":
            self.init_gradient_experiment()
        elif self.current_edit_mode_tab_name == "tab_gabor_filter_experiment_config":
            self.init_gabor_filter_experiment()
        elif self.current_edit_mode_tab_name == "tab_kmeans_experiment_config":
            self.init_kmeans_experiment()
        else:
            raise ValueError("Invalid edit mode!")

    def resize_image(self, image_data):
        """Resizes image given in argument

        Args:
            image_data (np.array/cv2 image): Image as npo.array to be resized

        Returns:
            np.array  (np.array/cv2 image): Resized image
        """
        scale_percent = min(
            self.max_img_width / image_data.shape[1],
            self.max_img_height / image_data.shape[0],
        )
        width = int(image_data.shape[1] * scale_percent)
        height = int(image_data.shape[0] * scale_percent)
        new_size = (width, height)
        image_resized = cv2.resize(
            image_data, new_size, None, None, None, cv2.INTER_AREA
        )
        return image_resized

    def pixmap_from_cv_image(self, cv_image):
        """Creates a QPixmap from cv2 image

        Args:
            cv_image (cv2 image): Image to convert

        Returns:
            QPixmap: Image as QPixmap
        """
        height, width, _ = cv_image.shape
        bytes_per_line = 3 * width
        q_img = QImage(
            cv_image.data, width, height, bytes_per_line, QImage.Format.Format_RGB888
        ).rgbSwapped()
        return QPixmap(q_img)

    # https://stackoverflow.com/questions/7587490/converting-numpy-array-to-opencv-array

    def save_as_file(self):
        """Saves output image as file"""
        if self.experiment.new_matrix is None:
            error_dialog = QErrorMessage()
            error_dialog.showMessage("No image processed")
            error_dialog.exec()
        else:
            output_file_path = QFileDialog.getSaveFileName(
                self, "Save File", self.source_filename, "Images (*.png *.xpm *.jpg)")[0]
            self.experiment.set_output_path(output_path=output_file_path)
            self.experiment.save_output_image(save_config=True)
            # filename = os.path.abspath(filename)
            # filename_wo_ext = os.path.splitext(filename)[0]
            # # file_extension = os.path.splitext(filename)[1]
            # if len(filename) > 0:
            #     cv2.imwrite(filename, self.experiment.new_matrix)
            #     with open("{0}_config.json".format(filename_wo_ext), "w") as outfile:
            #         json.dump(self.config, outfile, indent=4)

    # Image Processing
    def process_image(self):
        """Processes Source Image with current configuratin.
        If continuous_processing_radiobutton is checked, wil lcontinue processing
        """
        if self.source_image_data is None:
            error_dialog = QErrorMessage()
            error_dialog.showMessage("No image selected")
            error_dialog.exec()
        else:
            iteration_number = 0
            # if self.edit_mode=="schnock_experiment":
            #     self.experiment.pass_source_image(self.source_image_data)
            self.experiment.compute_new_matrix()
            self.result_image_data = self.experiment.new_matrix
            result_image_resized = self.resize_image(self.result_image_data)
            self.output_image_label.setPixmap(
                self.pixmap_from_cv_image(result_image_resized)
            )

            # if self.edit_mode=="gradient_experiment":
            while self.continuous_processing_radiobutton.isChecked() is True:
                iteration_number += 1
                self.experiment.compute_new_matrix()
                self.result_image_data = self.experiment.new_matrix

                if iteration_number % self.config["fast_forward_iterations"] == 0:
                    result_image_resized = self.resize_image(
                        self.result_image_data)
                    self.output_image_label.setPixmap(
                        self.pixmap_from_cv_image(result_image_resized)
                    )
                    # gc.collect()
                    QApplication.processEvents()
                    time.sleep(0)

    # Folder Processing
    def process_folder(self):
        """Processes Source Image with current configuratin.
        If continuous_processing_radiobutton is checked, wil lcontinue processing
        """
        image_folder_to_process = os.path.abspath(
            QFileDialog.getExistingDirectory(self, "Select Directory with images"))
        self.experiment.set_input_directory(
            input_directory=image_folder_to_process)
        self.select_extension_dialog()
        self.experiment.process_folder()

    def select_extension_dialog(self):
        text, ok = QInputDialog.getText(self, 'Select file extension',
                                        'Enter image extension (e.g. ".jpg", ".png"):')
        if ok:
            self.experiment.set_file_extension(str(text).upper())

    # SET ATTRIBUTES
    # General Attributes
    def set_clip_images(self):
        """Indicates if images should be clipped"""
        value = self.clip_images_radio_button.isChecked()
        self.experiment.set_clip_images(value=value)

    def set_compute_on_original(self):
        """Indicates if operations should be on original image or on resized image"""
        self.config[
            "compute_on_original"
        ] = self.compute_on_original_radio_button.isChecked()
        self.set_source_image_data()
        self.reset_output_image()

    def set_webcam_delay(self):
        """Sets the delay of the webcam for gradient experiment"""
        self.config["webcam_delay"] = self.webcam_delay_horizontal_slider.value()
        self.webcam_delay_lcd_number.display(self.config["webcam_delay"])

    def set_fast_forward_iterations(self):
        """Sets fast forward iterations, not really used atm"""
        self.config[
            "fast_forward_iterations"
        ] = self.fast_forward_iterations_spinbox.value()
        self.experiment.set_fast_forward_iterations(
            self.config["fast_forward_iterations"]
        )

    # SchnockExperiment attributes
    def set_low_shift(self):
        """Sets low shift for SchnockExperiment"""
        self.config["low_shift"] = self.low_shift_spinbox.value()
        self.experiment.set_low_shift(new_value=self.config["low_shift"])

    def set_mid_shift(self):
        """Sets mid shift for SchnockExperiment"""
        self.config["mid_shift"] = self.mid_shift_spinbox.value()
        self.experiment.set_mid_shift(new_value=self.config["mid_shift"])

    def set_high_shift(self):
        """Sets high shift for SchnockExperiment"""
        self.config["high_shift"] = self.high_shift_spinbox.value()
        self.experiment.set_high_shift(new_value=self.config["high_shift"])

    def set_low_threshold(self):
        """Sets low thershold for SchnockExperiment"""
        self.config["low_threshold"] = self.low_threshold_vertical_slider.value()
        self.low_threshold_lcd_number.display(self.config["low_threshold"])
        self.experiment.set_low_threshold(
            new_value=self.config["low_threshold"])

    def set_mid_threshold(self):
        """Sets mid thershold for SchnockExperiment"""
        self.config["mid_threshold"] = self.mid_threshold_vertical_slider.value()
        self.mid_threshold_lcd_number.display(self.config["mid_threshold"])
        self.experiment.set_mid_threshold(
            new_value=self.config["mid_threshold"])

    def set_high_threshold(self):
        """Sets high thershold for SchnockExperiment"""
        self.config["high_threshold"] = self.high_threshold_vertical_slider.value()
        self.high_threshold_lcd_number.display(self.config["high_threshold"])
        self.experiment.set_high_threshold(
            new_value=self.config["high_threshold"])

    # GradientExperiment ATTRIBUTES
    def set_merge_mode(self):
        """Sets merge mode for GradientExperiment"""
        self.config["merge_mode"] = self.merge_mode_combobox.currentText()
        self.experiment.set_merge_mode(
            new_merge_mode=self.config["merge_mode"])

    def set_video_merge_mode(self):
        """Sets video merge mode for GradientExperiment"""
        self.config["video_merge_mode"] = self.video_merge_mode_combobox.currentText()
        self.experiment.set_video_merge_mode(
            new_merge_mode=self.config["video_merge_mode"]
        )

    def set_multiplier_mode(self):
        """Sets multiplier mode for GradientExperiment"""
        self.config["multiplier_mode"] = self.multiplier_mode_combobox.currentText()
        self.experiment.set_multiplier_mode(
            new_multiplier_mode=self.config["multiplier_mode"]
        )

    def set_multiplier_amplitude(self):
        """Sets multiplier amplitude for GradientExperiment"""
        self.config["amplitue"] = self.multiplier_amplitude_spinbox.value()
        self.experiment.set_multiplier_amplitude(
            new_amplitude=self.config["amplitue"])

    def set_multiplier_frequency(self):
        """Sets multiplier frequency for GradientExperiment"""
        self.config["frequency"] = self.multiplier_frequency_spinbox.value()
        self.experiment.set_multiplier_frequency(
            new_frequency=self.config["frequency"])

    def set_alternate_every_n(self):
        """Sets alternate_every_n for GradientExperiment"""
        self.config["alternate_every_n"] = self.alternate_every_n_spinbox.value()
        self.experiment.set_alternate_every_n(
            new_value=self.config["alternate_every_n"]
        )

    def set_output_start_image_mode(self):
        """Sets sart image mode for GradientExperiment"""
        self.config[
            "output_start_mode"
        ] = self.output_start_image_combobox.currentText()

    def reset_output_image(self):
        """Resets output image for GradientExperiment"""
        if self.config["edit_mode"] == "gradient_experiment":
            self.experiment.output_start_mode_dict[self.config["output_start_mode"]](
            )
            self.result_image_data = self.experiment.new_matrix
            result_image_resized = self.resize_image(self.result_image_data)
            self.output_image_label.setPixmap(
                self.pixmap_from_cv_image(result_image_resized)
            )

    # Gabor Experiment Attributes
    def set_kernel_size(self):
        """Sets kernel size for gabor filter experiment"""
        self.config["kernel_size"] = self.kernel_size_horizontal_slider.value()
        self.kernel_size_lcd_number.display(self.config["kernel_size"])
        self.experiment.set_kernel_size(
            new_kernel_size=self.config["kernel_size"])

    def set_kernel_type(self):
        """Sets kernel type for gabor filter experiment"""
        self.config["kernel_type"] = self.kernel_type_combobox.currentText()
        self.experiment.set_kernel_type(
            new_kernel_type=self.config["kernel_type"])

    def set_kernel_sigma(self):
        """Sets kernel sigma for gabor filter experiment"""
        self.config["sigma"] = self.sigma_double_spinbox.value()
        self.experiment.set_sigma(new_sigma=self.config["sigma"])

    def set_kernel_lambda(self):
        """Sets kernel lambda for gabor filter experiment"""
        self.config["lambda"] = self.lambda_double_spinbox.value()
        self.experiment.set_lambda(new_lambda=self.config["lambda"])

    def set_kernel_gamma(self):
        """Sets kernel gamma for gabor filter experiment"""
        self.config["gamma"] = self.gamma_double_spinbox.value()
        self.experiment.set_gamma(new_gamma=self.config["gamma"])

    def set_kernel_psi(self):
        """Sets kernel psi for gabor filter experiment"""
        self.config["psi"] = self.psi_double_spinbox.value()
        self.experiment.set_psi(new_psi=self.config["psi"])

    def set_gabor_mode(self):
        """Sets gabor mode for gabor filter experiment"""
        self.config["gabor_mode"] = self.gabor_mode_combobox.currentText()
        self.experiment.set_gabor_mode(
            new_gabor_mode=self.config["gabor_mode"])

    def set_output_as_input(self):
        """Sets output image as input image for Experiment"""
        self.source_original_image_data = self.result_image_data
        self.set_source_image_data()
        self.reset_output_image()

    # Kmeans Experiment Attributes
    def set_number_of_clusters(self):
        self.config["number_of_clusters"] = self.number_of_clusters_horizontal_slider.value()
        self.number_of_clusters_lcd_number.display(
            self.config["number_of_clusters"])
        self.experiment.set_number_of_clusters(
            new_number_of_clusters=self.config["number_of_clusters"])

    def set_number_of_iterations(self):
        self.config["number_of_iterations"] = self.number_of_iterations_horizontal_slider.value()
        self.number_of_iterations_lcd_number.display(
            self.config["number_of_iterations"])
        self.experiment.set_number_of_iterations(
            new_number_of_iterations=self.config["number_of_iterations"])

    def set_number_of_repetitions(self):
        self.config["number_of_repetitions"] = self.number_of_repetitions_horizontal_slider.value()
        self.number_of_repetitions_lcd_number.display(
            self.config["number_of_repetitions"])
        self.experiment.set_number_of_repetitions(
            new_number_of_repetitions=self.config["number_of_repetitions"])

    def set_number_of_pyramids(self):
        self.config["number_of_pyramids"] = self.number_of_pyramids_horizontal_slider.value()
        self.number_of_pyramids_lcd_number.display(
            self.config["number_of_pyramids"])
        self.experiment.set_number_of_pyramids(
            new_number_of_pyramids=self.config["number_of_pyramids"])

    def set_scaling_factor(self):
        self.config["scaling_factor"] = self.scaling_factor_spinbox.value()
        self.experiment.set_scaling_factor(
            new_scaling_factor=self.config["scaling_factor"])

    def set_kmeans_init_type(self):
        self.config["kmeans_init_type"] = self.kmeans_init_types_combobox.currentText()
        self.experiment.set_kmeans_init_type(
            new_init_type=self.config["kmeans_init_type"])


# %%


def main():
    app = QApplication(sys.argv)
    window = UI()
    app.exec()


if __name__ == "__main__":
    main()

# # %%# %%

# %%
