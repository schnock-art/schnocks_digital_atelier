# %%
import os
import shutil


def test_experiment(experiment_type: str):
    if experiment_type == "schnock":
        # Schnock Experiment
        from experiment_classes.original_schnock import SchnockExperiment
        experiment = SchnockExperiment()
    elif experiment_type == "kmeans":
        from experiment_classes.kmeans_experiment import KmeansExperiment
        experiment = KmeansExperiment()
    elif experiment_type == "gradient":
        from experiment_classes.gradient_experiment import GradientExperiment
        experiment = GradientExperiment()
        experiment.set_fast_forward_iterations(10)
    elif experiment_type == "gabor_filter":
        from experiment_classes.gabor_filter_experiment import GaborFilterExperiment
        experiment = GaborFilterExperiment()

    # Delete output directory if it exists
    output_directory = os.path.abspath(
        r"..\test_results\{0}_experiment".format(experiment_type))
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory)

    # Run experiment for single file and save all variants
    input_path = os.path.abspath(r"..\data\DSC03351.JPG")
    input_directory = os.path.dirname(input_path)
    output_directory = os.path.abspath(
        r"..\test_results\{0}_experiment\single".format(experiment_type))

    experiment.set_input_directory(input_directory=input_directory)
    experiment.set_output_directory(
        output_directory=output_directory,
    )
    experiment.create_output_directory()
    experiment.process_path(
        path=input_path,
        save_all_variants=True,
    )

    # Run experiment for folder without saving all variants
    output_directory = os.path.abspath(
        r"..\test_results\{0}_experiment".format(experiment_type))

    experiment.process_folder(
        input_directory=os.path.abspath(r"..\data"),
        output_directory=output_directory,
        file_extension=".JPG",
    )


# %%
test_experiment(experiment_type="schnock")
# %%
test_experiment(experiment_type="kmeans")
# %%
test_experiment(experiment_type="gradient")
# %%
test_experiment(experiment_type="gabor_filter")
