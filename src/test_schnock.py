# %%
from experiment_classes.original_schnock import SchnockExperiment
import os

experiment = SchnockExperiment()
experiment.process_path(path=os.path.abspath("..\data\DSC03351.JPG"))
# %%
experiment.output_image_path

# %%
