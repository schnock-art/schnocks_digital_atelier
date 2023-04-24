#%%
from PyQt6.QtWidgets import (QApplication, 
    QFileDialog, QPushButton, QRadioButton,
    QMainWindow, QLabel,  QErrorMessage, 
    QSlider, QComboBox, QSpinBox, QTabWidget, QLCDNumber
)
import sys
from PyQt6 import uic
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
import numpy as np
import cv2
import time 
import logging


from gradient_experiment import GradientExperiment
from original_schnock import SchnockExperiment

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def run(self):
        # capture from web cam
        cap = cv2.VideoCapture(0)
        while True:
            ret, cv_img = cap.read()
            if ret:
                self.change_pixmap_signal.emit(cv_img)

class UI(QMainWindow):
    def __init__(self):
        try:
            super(UI, self).__init__()
            uic.loadUi("main_ui.ui",self)
            logging.basicConfig(level=logging.INFO)
            self.max_img_height = 640
            self.max_img_width = 640

            self.matrix_list=[]
            self.current_matrix_index=0
            self.webcam_frame_counter=0
            self.clip_images=False
            self.multiplier_mode="constant"
            self.merge_mode="mean"
            self.alternate_every_n=5
            self.multiplier=1
            self.webcam_delay=1
            self.source_original_image_data=None
            self.config={}
            self.define_widgets()
            self.connect_buttons()
            self.init_values()
            self.show()
            
        except Exception as error:
            logging.error("Failed initialization with error: {0}".format(error))
            raise error


    def init_values(self):       
        self.set_edit_mode()
        if self.config["edit_mode"]=="schnock_original":
            self.init_schnock_experiment()
        elif self.config["edit_mode"]=="gradient_experiment":
            self.init_gradient_experiment()
        else:
            raise Exception("No edit mode specified!")
        
    def init_common_values(self):
        self.set_fast_forward_iterations()
        self.set_webcam_delay()

    def init_gradient_experiment(self):
        self.experiment=GradientExperiment()
        self.config={}
        self.config["edit_mode"]="gradient_experiment"
        self.set_multiplier_mode()
        self.set_merge_mode()
        self.set_multiplier_amplitude()
        self.set_multiplier_frequency()
        self.set_output_start_image_mode()
        self.set_alternate_every_n()
        self.set_video_merge_mode()
        self.config["compute_on_original"] = self.compute_on_original_radio_button.isChecked()
        if self.source_original_image_data is not None:
            self.set_source_image_data()
            self.reset_output_image()
        self.init_common_values()
        
    
    def init_schnock_experiment(self):
        self.experiment=SchnockExperiment()
        self.config={}
        self.config["edit_mode"]="schnock_experiment"
        self.set_low_shift()
        self.set_mid_shift()
        self.set_high_shift()
        self.set_low_threshold()
        self.set_mid_threshold()
        self.set_high_threshold()
        self.config["compute_on_original"] = self.compute_on_original_radio_button.isChecked()
        if self.source_original_image_data is not None:
            self.set_source_image_data()
        self.init_common_values()


    def define_widgets(self):
        # Push Buttons
        self.load_source_image_button = self.findChild(QPushButton, "pushButton_load_input_image" )
        self.process_image_button = self.findChild(QPushButton, "pushButton_process_image" )
        self.next_image_button = self.findChild(QPushButton, "pushButton_next_image" )
        self.previous_image_button = self.findChild(QPushButton, "pushButton_previous_image" )
        self.save_image_button = self.findChild(QPushButton, "pushButton_save_image")
        self.reset_output_image_button = self.findChild(QPushButton, "pushButton_reset_output")

        # Radio Buttons
        self.continuous_processing_radiobutton = self.findChild(QRadioButton, "radioButton_continuous")
        self.clip_images_radio_button = self.findChild(QRadioButton, "radioButton_clip_images")
        self.compute_on_original_radio_button = self.findChild(QRadioButton, "radioButton_compute_on_original")
        self.webcam_input_radiobutton = self.findChild(QRadioButton, "radioButton_webcam_input")
        # Images
        self.input_image_label = self.findChild(QLabel, "Input_Image")
        self.output_image_label = self.findChild(QLabel, "Output_Image")
        self.input_image_label.setMaximumSize(self.max_img_width, self.max_img_height)
        self.output_image_label.setMaximumSize(self.max_img_width, self.max_img_height)

        # Horizontal sliders
        self.webcam_delay_horizontal_slider = self.findChild(QSlider, "horizontalSlider_webcam_delay")
        
        #Vertical sliders
        self.low_threshold_vertical_slider = self.findChild(QSlider, "verticalSlider_low_threshold")
        self.mid_threshold_vertical_slider = self.findChild(QSlider, "verticalSlider_mid_threshold")
        self.high_threshold_vertical_slider = self.findChild(QSlider, "verticalSlider_high_threshold")

        # ComboBoxes
        # Output start image
        self.output_start_image_combobox = self.findChild(QComboBox, "comboBox_output_start_image")
        for item in GradientExperiment().output_start_mode_dict.keys():
            self.output_start_image_combobox.addItem(item)
        # Proccessing Modes
        self.multiplier_mode_combobox = self.findChild(QComboBox, "comboBox_multiplier_mode")
        for item in GradientExperiment().dynamic_multpiplier_functions_dict.keys():
            self.multiplier_mode_combobox.addItem(item)

        self.merge_mode_combobox = self.findChild(QComboBox, "comboBox_merge_mode")
        for item in GradientExperiment().merge_mode_functions_dict.keys():
            self.merge_mode_combobox.addItem(item)

        self.video_merge_mode_combobox = self.findChild(QComboBox, "comboBox_video_merge_mode")
        for item in GradientExperiment().video_merge_mode_functions_dict.keys():
            self.video_merge_mode_combobox.addItem(item)

        # Spinboxes
        self.multiplier_amplitude_spinbox = self.findChild(QSpinBox, "spinBox_multiplier_amplitude")        
        self.multiplier_frequency_spinbox = self.findChild(QSpinBox, "spinBox_multiplier_frequency")
        self.fast_forward_iterations_spinbox = self.findChild(QSpinBox,"spinBox_fast_forward_iterations")
        self.alternate_every_n_spinbox = self.findChild(QSpinBox, "spinBox_alternate_every_n")
        self.low_shift_spinbox = self.findChild(QSpinBox, "spinBox_low_shift")
        self.mid_shift_spinbox = self.findChild(QSpinBox, "spinBox_mid_shift")
        self.high_shift_spinbox = self.findChild(QSpinBox, "spinBox_high_shift")
        
        # Tabs
        self.edit_mode_tab=self.findChild(QTabWidget,"tabWidget_edit_mode")
        # self.photo_mode_tab=self.findChild(QTabWidget, "tab_photo_mode")
        # self.video_mode_tab=self.findChild(QTabWidget, "tab_video_mode")
        
        self.gradient_experiment_config_tab = self.findChild(QTabWidget, "tab_gradient_experiment_config")
        self.schnock_original_config_tab = self.findChild(QTabWidget, "tab_schnock_original_config")
        
        # LCD Numbers
        self.webcam_delay_lcd_number = self.findChild(QLCDNumber, "lcdNumber_webcam_delay")
        self.low_threshold_lcd_number = self.findChild(QLCDNumber, "lcdNumber_low_threshold")
        self.mid_threshold_lcd_number = self.findChild(QLCDNumber, "lcdNumber_mid_threshold")
        self.high_threshold_lcd_number = self.findChild(QLCDNumber, "lcdNumber_high_threshold")
        pass

    def connect_buttons(self):
        #Push Buttons
        self.load_source_image_button.clicked.connect(self.load_source_image_from_file)
        self.process_image_button.clicked.connect(self.process_image)
        self.save_image_button.clicked.connect(self.save_as_file)
        self.reset_output_image_button.clicked.connect(self.reset_output_image)

        # Combo boxes
        self.merge_mode_combobox.currentTextChanged.connect(self.set_merge_mode)
        self.video_merge_mode_combobox.currentTextChanged.connect(self.set_video_merge_mode)
        self.multiplier_mode_combobox.currentTextChanged.connect(self.set_multiplier_mode)
        self.output_start_image_combobox.currentTextChanged.connect(self.set_output_start_image_mode)
        
        #Spinboxes
        self.multiplier_amplitude_spinbox.valueChanged.connect(self.set_multiplier_amplitude)
        self.multiplier_frequency_spinbox.valueChanged.connect(self.set_multiplier_frequency)
        self.fast_forward_iterations_spinbox.valueChanged.connect(self.set_fast_forward_iterations)
        self.alternate_every_n_spinbox.valueChanged.connect(self.set_alternate_every_n)
        self.low_shift_spinbox.valueChanged.connect(self.set_low_shift)
        self.mid_shift_spinbox.valueChanged.connect(self.set_mid_shift)
        self.high_shift_spinbox.valueChanged.connect(self.set_high_shift)

        # Radiobutton
        self.clip_images_radio_button.toggled.connect(self.set_clip_images)
        self.compute_on_original_radio_button.toggled.connect(self.set_compute_on_original)
        self.webcam_input_radiobutton.toggled.connect(self.start_video)

        #Horizontal Sliders
        self.webcam_delay_horizontal_slider.valueChanged.connect(self.set_webcam_delay)
        # Vertical Sliders
        self.low_threshold_vertical_slider.valueChanged.connect(self.set_low_threshold)
        self.mid_threshold_vertical_slider.valueChanged.connect(self.set_mid_threshold)
        self.high_threshold_vertical_slider.valueChanged.connect(self.set_high_threshold)
        
        # Tabs
        self.edit_mode_tab.currentChanged.connect(self.set_edit_mode)
        pass

    @pyqtSlot(np.ndarray)
    def set_source_image_data(self):
        if self.config["compute_on_original"]==True:
            self.source_image_data=self.source_original_image_data
        else:
            self.source_image_data = self.resize_image(self.source_original_image_data)
        
        self.experiment.pass_source_image(source_image=self.source_image_data)

        source_image_resized = self.resize_image(self.source_image_data)
        self.input_image_label.setPixmap(self.pixmap_from_cv_image(source_image_resized))
    

    def start_video(self):
        if not self.webcam_input_radiobutton.isChecked():
            logging.info("Stopping video Thread")
            #self.thread.stop()
            return
        self.thread = VideoThread()
        self.thread.change_pixmap_signal.connect(self.update_source_image_from_webcam)
        
        # start the thread
        self.thread.start()
        #self.reset_output_image()


    @pyqtSlot(np.ndarray)
    def update_source_image_from_webcam(self, cv_img):
        """Updates the image_label with a new opencv image"""
        if not self.webcam_input_radiobutton.isChecked():
            logging.info("Stopping video")
            #self.thread.stop()
            return
        self.webcam_frame_counter+=1
        self.source_original_image_data=cv_img
        if self.config["compute_on_original"]==True:
            self.source_image_data=self.source_original_image_data
        else:
            self.source_image_data = self.resize_image(self.source_original_image_data)
        
        self.experiment.pass_source_image(source_image=self.source_image_data)
        
        source_image_resized = self.resize_image(self.source_image_data)
        self.input_image_label.setPixmap(self.pixmap_from_cv_image(source_image_resized))
        
        if self.webcam_frame_counter%self.config["webcam_delay"]==0:
            if self.config["edit_mode"]=="gradient_experiment":
                self.experiment.output_start_mode_dict["video"]()
            
        
        self.experiment.compute_new_matrix()
        self.result_image_data=self.experiment.new_matrix
        result_image_resized = self.resize_image(self.result_image_data)
        self.output_image_label.setPixmap(self.pixmap_from_cv_image(result_image_resized))

    def load_source_image_from_file(self):
        self.source_filename = QFileDialog.getOpenFileName(self, "Open File")[0]
        self.source_original_image_data = cv2.imread(self.source_filename)

        self.set_source_image_data()
        self.set_edit_mode()
        self.reset_output_image()
        pass

    def set_edit_mode(self):
        self.current_edit_mode_tab_name=self.edit_mode_tab.currentWidget().objectName()
        if self.current_edit_mode_tab_name=="tab_schnock_original_config":
            self.init_schnock_experiment()
            self.config["edit_mode"]="schnock_original"
        elif self.current_edit_mode_tab_name=="tab_gradient_experiment_config":
            self.init_gradient_experiment()
            
        else:
            raise Exception("Invalid edit mode!")
        
        
    def resize_image(self,image_data):
        scale_percent = min(self.max_img_width / image_data.shape[1], self.max_img_height / image_data.shape[0])
        width = int(image_data.shape[1] * scale_percent)
        height = int(image_data.shape[0] * scale_percent)
        newSize = (width, height)
        image_resized = cv2.resize(image_data, newSize, None, None, None, cv2.INTER_AREA)
        return image_resized

    def pixmap_from_cv_image(self, cv_image):
        height, width, _ = cv_image.shape
        bytesPerLine = 3 * width
        qImg = QImage(cv_image.data, width, height, bytesPerLine, QImage.Format.Format_RGB888).rgbSwapped()
        return QPixmap(qImg)    
    # https://stackoverflow.com/questions/7587490/converting-numpy-array-to-opencv-array

    def save_as_file(self):
        if self.experiment.new_matrix is None:
            error_dialog = QErrorMessage()
            error_dialog.showMessage('No image processed')
            error_dialog.exec()
        else:
            filename = QFileDialog.getSaveFileName(self, 'Save File')[0]
            if len(filename) > 0:
                cv2.imwrite(filename, self.experiment.new_matrix)

    def set_clip_images(self):
        value = self.clip_images_radio_button.isChecked()
        self.experiment.set_clip_images(value=value)
    
    def set_compute_on_original(self):
        self.config["compute_on_original"] = self.compute_on_original_radio_button.isChecked()
        self.set_source_image_data()
        self.reset_output_image()

    def set_webcam_delay(self):
        self.config["webcam_delay"] = self.webcam_delay_horizontal_slider.value()
        self.webcam_delay_lcd_number.display(self.config["webcam_delay"])

    def set_low_shift(self):
        self.config["low_shift"] = self.low_shift_spinbox.value()
        self.experiment.set_low_shift(new_value=self.config["low_shift"])

    def set_mid_shift(self):
        self.config["mid_shift"] = self.mid_shift_spinbox.value()
        self.experiment.set_mid_shift(new_value=self.config["mid_shift"])

    def set_high_shift(self):
        self.config["high_shift"] = self.high_shift_spinbox.value()
        self.experiment.set_high_shift(new_value=self.config["high_shift"])

    def set_low_threshold(self):
        self.config["low_threshold"] = self.low_threshold_vertical_slider.value()
        self.low_threshold_lcd_number.display(self.config["low_threshold"])
        self.experiment.set_low_threshold(new_value=self.config["low_threshold"])

    def set_mid_threshold(self):
        self.config["mid_threshold"] = self.mid_threshold_vertical_slider.value()
        self.mid_threshold_lcd_number.display(self.config["mid_threshold"])
        self.experiment.set_mid_threshold(new_value=self.config["mid_threshold"])

    def set_high_threshold(self):
        self.config["high_threshold"] = self.high_threshold_vertical_slider.value()
        self.high_threshold_lcd_number.display(self.config["high_threshold"])
        self.experiment.set_high_threshold(new_value=self.config["high_threshold"])

    def set_merge_mode(self):
        self.config["merge_mode"] = self.merge_mode_combobox.currentText()
        self.experiment.set_merge_mode(new_merge_mode=self.config["merge_mode"]) 

    def set_video_merge_mode(self):
        self.config["video_merge_mode"] = self.video_merge_mode_combobox.currentText()
        self.experiment.set_video_merge_mode(new_merge_mode=self.config["video_merge_mode"]) 

    def set_multiplier_mode(self):
        self.config["multiplier_mode"] = self.multiplier_mode_combobox.currentText()
        self.experiment.set_multiplier_mode(new_multiplier_mode=self.config["multiplier_mode"]) 
        
    def set_multiplier_amplitude(self):
        self.config["amplitue"] = self.multiplier_amplitude_spinbox.value()
        self.experiment.set_multiplier_amplitude(new_amplitude=self.config["amplitue"])

    def set_multiplier_frequency(self):
        self.config["frequency"] = self.multiplier_frequency_spinbox.value()
        self.experiment.set_multiplier_frequency(new_frequency=self.config["frequency"])

    def set_alternate_every_n(self):
        self.config["alternate_every_n"] = self.alternate_every_n_spinbox.value()
        self.experiment.set_alternate_every_n(new_value=self.config["alternate_every_n"])

    def set_output_start_image_mode(self):
        self.config["output_start_mode"] = self.output_start_image_combobox.currentText()

    def set_fast_forward_iterations(self):
        self.config["fast_forward_iterations"] = self.fast_forward_iterations_spinbox.value()

    def reset_output_image(self):
        if self.config["edit_mode"]=="gradient_experiment":
            self.experiment.output_start_mode_dict[self.config["output_start_mode"]]()
            self.result_image_data = self.experiment.new_matrix
            result_image_resized = self.resize_image(self.result_image_data)
            self.output_image_label.setPixmap(self.pixmap_from_cv_image(result_image_resized))

        

    ####### Image Processing

    def process_image(self):
        if self.source_image_data is None:
            error_dialog = QErrorMessage()
            error_dialog.showMessage('No image selected')
            error_dialog.exec()
        else:
            n=0
            # if self.edit_mode=="schnock_experiment":
            #     self.experiment.pass_source_image(self.source_image_data)
            self.experiment.compute_new_matrix()
            self.result_image_data=self.experiment.new_matrix
            result_image_resized = self.resize_image(self.result_image_data)
            self.output_image_label.setPixmap(self.pixmap_from_cv_image(result_image_resized))
            
            #if self.edit_mode=="gradient_experiment":
            while self.continuous_processing_radiobutton.isChecked()==True:
                n+=1
                self.experiment.compute_new_matrix()
                self.result_image_data=self.experiment.new_matrix
                
                if n%self.config["fast_forward_iterations"]==0:
                    result_image_resized = self.resize_image(self.result_image_data)
                    self.output_image_label.setPixmap(self.pixmap_from_cv_image(result_image_resized))
                    #gc.collect()
                    QApplication.processEvents()
                    time.sleep(0)

app=QApplication(sys.argv)
window = UI()
app.exec()

# # %%# %%

# %%
