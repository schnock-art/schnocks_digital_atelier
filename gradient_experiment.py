#%%

import numpy as np
import matplotlib.pyplot as plt
#from array2gif import write_gif
import cv2
import os
from PIL import Image
from math import sin, cos, tan, pi
import logging



folder="Computer-Vision-with-Python\DATA"
file_name1="dog_backpack.png"
class GradientExperiment:
    def __init__(self, 
                #  source_folder_path: str="Computer-Vision-with-Python\DATA", 
                #  source_image_path: str="dog_backpack.png"
        ):
        try:
            self.source_diff={}
            self.source_image=None
            self.matrix_list=[]
            self.current_iteration_n=0
            self.alternate_counter=0
            self.multiplier_amplitude=1
            self.multiplier_frequency=4
            self.grade=1
            self.clip_images=False
            self.alternate_every_n=0
            self.initialize_dynamic_multiplier_dict()
            self.initialize_merge_mode_dict()
            self.initialize_output_start_images_dict()
        except Exception as error:
            logging.error("Failed initialization with error: {0}".format(error))
            raise error
        
        # # self.load_source_image(
        # #     source_folder_path=source_folder_path,
        # #     source_image_path=source_image_path
        # # )

        # self.get_difference_matrices()

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
        self.get_difference_matrices()
        pass
    
    def pass_output_start_image(self, output_start_image):
        self.new_matrix = output_start_image

    def set_clip_images(self, value:bool):
        self.clip_images = value

    def get_difference_matrices(self):
        self.padded_source_image={}
        for n in range(1, self.grade+1):
            self.padded_source_image[n]=np.pad(self.source_image,pad_width=((n,n),(n,n),(0,0)),mode="edge")
            self.source_diff["x_{0}".format(n)]=np.diff(self.padded_source_image[n][n:-n,:,:],n=n, axis=1)
            self.source_diff["y_{0}".format(n)]=np.diff(self.padded_source_image[n][:,n:-n,:],n=n, axis=0)
        # self.source_diff_x=np.diff(self.padded_source_image[n:-n,:,:],n=n, axis=1)
        # self.source_diff_y=np.diff(self.padded_source_image[:,n:-n,:],n=n, axis=0)
        pass


    def compute_new_matrix(self):
        self.dynamic_multpiplier_functions_dict[self.multiplier_mode]()
        logging.debug("Dynamic multiplier: {0}".format(self.dynamic_multiplier))
        matrix_stack=[]
        for n in range(1, self.grade+1):
            padded_n=np.pad(self.new_matrix,pad_width=((n,n),(n,n),(0,0)),mode="edge")
            positive_x = padded_n[n:-n,(n+1):,:] + self.dynamic_multiplier * np.negative(self.source_diff["x_{0}".format(n)][:,n:,:])
            negative_x = padded_n[n:-n,:-(n+1),:]  + self.dynamic_multiplier * self.source_diff["x_{0}".format(n)][:,:-n,:]
            positive_y = padded_n[(n+1):,n:-n,:] + self.dynamic_multiplier * np.negative(self.source_diff["y_{0}".format(n)][n:,:])
            negative_y = padded_n[0:-(n+1),n:-n,:]  + self.dynamic_multiplier * self.source_diff["y_{0}".format(n)][:-n,:]
            if self.clip_images==True:
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

        self.difference_matrix=np.stack(tuple(matrix_stack))
        self.merge_mode_functions_dict[self.merge_mode]()
        self.current_iteration_n +=1
        pass

    
    def initialize_dynamic_multiplier_dict(self):
        self.dynamic_multpiplier_functions_dict={
            "constant": self.dynamic_multiplier_constant,
            #"current_iteration_n": self.dynamic_multiplier_linear_reduction,
            "exponential_reduction": self.dynamic_multiplier_exponential_reduction,
            "cosinus": self.dynamic_multiplier_cos,
            "sinus": self.dynamic_multiplier_sin,
        }

    def initialize_merge_mode_dict(self):
        self.merge_mode_functions_dict={
            "average": self.merge_average,
            "min": self.merge_min,
            "max": self.merge_max,
            "sum": self.merge_sum,
            "alternate": self.merge_alternate,
        }
    
    def initialize_output_start_images_dict(self):
        self.output_start_mode_dict={
            "gray": self.ouput_start_image_gray,
            "black": self.ouput_start_image_black,
            "white": self.ouput_start_image_white,
            "source_average": self.ouput_start_image_source_average,
        }
    
    def ouput_start_image_gray(self):
        self.new_matrix = np.full(shape=self.source_image.shape, fill_value=127, dtype=np.uint8)
        self.current_iteration_n = 0
        self.alternate_counter=0

    def ouput_start_image_black(self):
        self.new_matrix = np.full(shape=self.source_image.shape, fill_value=0, dtype=np.uint8)
        self.current_iteration_n = 0
        self.alternate_counter=0
    
    def ouput_start_image_white(self):
        self.new_matrix = np.full(shape=self.source_image.shape, fill_value=255, dtype=np.uint8)
        self.current_iteration_n = 0
        self.alternate_counter=0
    
    def ouput_start_image_custom(self):
        self.new_matrix = np.full(shape=self.source_image.shape, fill_value=127, dtype=np.uint8)
        self.current_iteration_n = 0
        self.alternate_counter=0

    def ouput_start_image_source_average(self):
        mean_pixel = np.mean(self.source_image, axis=(0,1)).round().astype(np.uint8)
        self.new_matrix = np.full(shape=self.source_image.shape, fill_value=mean_pixel, dtype=np.uint8)
        self.current_iteration_n = 0
        self.alternate_counter=0


    def merge_average(self):
        logging.debug("Merge mode: Avg")
        self.new_matrix=np.average(self.difference_matrix, axis=0).astype(np.uint8)
        return
    
    def merge_min(self):
        logging.debug("Merge mode: Min")
        self.new_matrix=np.min(self.difference_matrix, axis=0).astype(np.uint8)
        return
    
    def merge_max(self):
        logging.debug("Merge mode: Max")
        self.new_matrix=np.max(self.difference_matrix, axis=0).astype(np.uint8)
        return
    
    def merge_sum(self):
        logging.debug("Merge mode: Sum")
        self.new_matrix=np.sum(self.difference_matrix, axis=0).astype(np.uint8)
        return
    
    def merge_alternate(self):
        md=self.alternate_counter//self.alternate_every_n
        if md==0:
            self.merge_max()
        elif md==1:
            self.merge_average()
        elif md==2:
            self.merge_min()
        else:
            logging.error(md)
            logging.error(self.alternate_counter)
            raise Exception("Error modulo")
        self.alternate_counter+=1
        md=self.alternate_counter//self.alternate_every_n
        if md==3:
            self.alternate_counter=0

    def set_merge_mode(self, new_merge_mode: str):
        if new_merge_mode not in self.merge_mode_functions_dict.keys():
            raise Exception("Invalid merge_mode ({0}), merge mode should be in {1}".format(new_merge_mode, self.merge_mode_functions_dict.keys()))
        self.merge_mode=new_merge_mode
        pass

    def set_multiplier_mode(self, new_multiplier_mode: str):
        if new_multiplier_mode not in self.dynamic_multpiplier_functions_dict.keys():
            raise Exception("Invalid merge_mode ({0}), merge mode should be in {1}".format(new_multiplier_mode, self.dynamic_multpiplier_functions_dict.keys()))
        self.multiplier_mode=new_multiplier_mode
        pass

    def set_multiplier_amplitude(self, new_amplitude: float):
        self.multiplier_amplitude = new_amplitude

    def set_multiplier_frequency(self, new_frequency: float):
        self.multiplier_frequency = new_frequency

    def set_alternate_every_n(self, new_value):
        self.alternate_every_n = new_value
        self.alternate_counter=0

    def dynamic_multiplier_constant(self):
        self.dynamic_multiplier = self.multiplier_amplitude
        
    def dynamic_multiplier_linear_reduction(self):
        self.dynamic_multiplier=self.multiplier_amplitude*(self.n_iterations-self.current_iteration_n)/self.n_iterations

    def dynamic_multiplier_exponential_reduction(self):
        self.dynamic_multiplier = 1+int(self.multiplier_amplitude*np.exp(-self.current_iteration_n))
    
    def dynamic_multiplier_cos(self):
        self.dynamic_multiplier = int(self.multiplier_amplitude*cos(2*pi*self.current_iteration_n/self.multiplier_frequency))
    
    def dynamic_multiplier_sin(self):
        self.dynamic_multiplier = int(self.multiplier_amplitude*sin(2*pi*self.current_iteration_n/self.multiplier_frequency))

    
    # def compute_iterative(self):

    #     if self.gif_mode==True:
    #         self.matrix_list=[self.new_matrix]
    #     else:
    #         self.matrix_list=[]
    #     alternate_counter=0
 
    #     for i in range(self.n_iterations):
    #         if self.multiplier_mode=="constant":
    #             dynamic_multiplier=self.multiplier
    #         elif self.multiplier_mode=="linear_reduction":
    #             dynamic_multiplier=(self.multiplier*(self.n_iterations-i)/self.n_iterations)
    #         elif self.multiplier_mode=="exponential_reduction":
    #             dynamic_multiplier=(1+self.multiplier*np.exp(-i))
    #         else:
    #             dynamic_multiplier=self.multiplier
    #         difference_matrix=self.compute_new_matrix(
    #             new_matrix=self.new_matrix, 
    #             multiplier=dynamic_multiplier,
    #         )
    #         if self.merge_mode=="mean":
    #             self.new_matrix=np.average(difference_matrix, axis=0).astype(np.uint8)
    #         elif self.merge_mode=="min":
    #             self.new_matrix=np.min(difference_matrix, axis=0).astype(np.uint8)
    #         elif self.merge_mode=="max":
    #             self.new_matrix=np.max(difference_matrix, axis=0).astype(np.uint8)
    #         elif self.merge_mode=="sum":
    #             self.new_matrix=np.sum(difference_matrix, axis=0).astype(np.uint8)
    #         elif self.merge_mode=="alternate":
                
    #             md=alternate_counter//self.alternate_every_n
    #             if md==0:
    #                 logging.debug("")
    #                 self.new_matrix=np.max(difference_matrix, axis=0).astype(np.uint8)
    #             elif md==1:
    #                 self.new_matrix=np.min(difference_matrix, axis=0).astype(np.uint8)
    #             # elif md==2:
    #             #     new_matrix=np.average(new_matrix, axis=0).astype(np.uint8)
    #             else:
    #                 print(md)
    #                 print(alternate_counter)
    #                 raise Exception("Error modulo")
    #             alternate_counter+=1
    #             md=alternate_counter//self.alternate_every_n
    #             if md==2:
    #                 alternate_counter=0
    #         elif self.merge_mode=="random":
    #             dice=np.random.randint(0,2)
    #             if dice==0:
    #                 self.new_matrix=np.average(difference_matrix, axis=0).astype(np.uint8)
    #             elif dice==1:
    #                 self.new_matrix=np.min(difference_matrix, axis=0).astype(np.uint8)
    #             if dice==2:
    #                 self.new_matrix=np.max(difference_matrix, axis=0).astype(np.uint8)
    #         else:
    #             raise Exception("Not a valid mode provided!")
    #         if self.gif_mode==True:
    #             self.matrix_list.append(self.new_matrix)
    #     #print(new_matrix)
    #     return self.new_matrix, self.matrix_list

    # def create_new_image_from_old(self, 
    #         merge_mode="alternate", 
    #         n_iterations=10, 
    #         multiplier_mode="constant",
    #         multiplier=5,
    #         alternate_every_n=5,
    #         gif_mode=True,
    #         clip_images=False,
    #         save: bool=True
    #     ):
    #     self.alternate_every_n=alternate_every_n
    #     self.merge_mode=merge_mode
    #     self.n_iterations=n_iterations
    #     self.multiplier=multiplier
    #     self.multiplier_mode=multiplier_mode
    #     self.gif_mode=gif_mode
    #     self.clip_images=clip_images
    #     self.new_matrix = np.full(shape=self.source_image.shape, fill_value=127, dtype=np.uint8)
    #     print("starting iterations")
    #     self.compute_iterative()
    #     print("Saving last experiment iterations")
    #     self.save_last_experiment(save_gif=save)
    #     return
    
    # def save_last_experiment(self, path: str="last_experiment", save_gif: bool=True):
    #     Image.fromarray(self.new_matrix).save("{0}.jpg".format(path))
    #     if save_gif:
    #         print("Saving GIF")
    #         imgs = [Image.fromarray(img) for img in self.matrix_list]
    #         imgs[0].save("{0}.gif".format(path), save_all=True, append_images=imgs[1:], duration=100, loop=0)

    #     pass

# experiment=GradientExperiment()
# experiment.create_new_image_from_old(
#         merge_mode="alternate",
#         n_iterations=500,
#         multiplier=2,
#         multiplier_mode="constant",
#         alternate_every_n=6,
#         clip_images=True
#     )

#%%
if __name__=="__main__":
    experiment=GradientExperiment()

    experiment.create_new_image_from_old(
        merge_mode="mean",
        n_iterations=500,
        multiplier=3,
        multiplier_mode="exponential_reduction",
        alternate_every_n=10
    )

#Image.fromarray(experiment.new_matrix).save("latest_experiment.jpg")

#%%

# imgs = [Image.fromarray(img) for img in img_list]

# imgs[0].save("alternate.gif", save_all=True, append_images=imgs[1:200], duration=100, loop=0)

# #%%
# import cv2
# size = img_list[0].shape[1], img_list[0].shape[2]

# layers,width,height=img_list[0].shape

# #video=cv2.VideoWriter('video.avi',0,1,(width,height))
# fps=15
# frame_nr = len(img_list)
# duration=frame_nr, fps
# out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (size[0], size[1]))
# for i in range(frame_nr):
#     #print(img.shape)
#     out.write(img_list[i])
# out.release()
# #%%
# maxim = new_image
# #%%
# minim=new_image

# #%%
# mean=new_image

# #%%
# random=new_image
# # %%
