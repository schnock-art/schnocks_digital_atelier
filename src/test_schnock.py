# %%
from experiment_classes.kmeans_experiment import KmeansExperiment
from experiment_classes.original_schnock import SchnockExperiment
import os

experiment = SchnockExperiment()
experiment.process_path(path=os.path.abspath("..\data\DSC03351.JPG"))
# %%
experiment.output_image_path

# %%

# %%
experiment = KmeansExperiment()
# %%
experiment.set_scaling_factor(new_scaling_factor=2)
experiment.set_number_of_iterations(new_number_of_iterations=10)
experiment.set_number_of_piramids(new_number_of_piramids=4)
experiment.set_number_of_repetitions(new_number_of_repetitions=5)
experiment.set_number_of_clusters(new_number_of_clusters=100)
experiment.process_path(path=os.path.abspath("..\data\DSC03717.JPG"))

# %%
pix1 = experiment.cluster_centers[experiment.new_matrix[0]]
pix2 = experiment.cluster_centers[experiment.new_matrix[1]]


# %%
