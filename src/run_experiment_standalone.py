# %%

import os
import shutil
import sys
from tqdm import tqdm


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
        r"test_results\{0}_experiment".format(experiment_type))
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory)

    # Run experiment for single file and save all variants
    input_path = os.path.abspath(r"data\DSC03351.JPG")
    input_directory = os.path.dirname(input_path)
    output_directory = os.path.abspath(
        r"test_results\{0}_experiment\single".format(experiment_type))

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
        r"test_results\{0}_experiment".format(experiment_type))

    experiment.process_folder(
        input_directory=os.path.abspath(r"data"),
        output_directory=output_directory,
        file_extension=".JPG",
    )


# %%


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Runs experiments.')
    parser.add_argument('experiment_type', metavar='e', type=str, nargs='?',
                        choices=["schnock", "kmeans",
                                 "gradient", "gabor_filter"],
                        help='Experiment type to run. Either schnock, kmeans, gradient or gabor_filter')
    parser.add_argument('output_directory', type=str, nargs='?',
                        help='Path to directory to save results to.')
    parser.add_argument('folder_or_file', metavar='f', type=str, nargs='?',
                        choices=["folder", "file"],
                        help='Process single files or folders? Either folder or file.')
    parser.add_argument('--file_path', dest='files', action="extend", nargs="+", type=str,
                        help='Path to file to run experiment on. Can be callled multiple times.')
    parser.add_argument('--input_directory', dest='input_directory', type=str,
                        help='Path to directory to run experiment on.')

    parser.add_argument('--file_extension, -e', dest='file_extension', type=str,
                        choices=[".JPG", ".jpg", ".png", ".PNG"])
    parser.add_argument('--test', dest='test', action="store_true")
    args = parser.parse_args()

    if args.test:
        test_experiment("schnock")
        test_experiment("kmeans")
        test_experiment("gradient")
        test_experiment("gabor_filter")
        sys.exit()

    if args.experiment_type == "schnock":
        # Schnock Experiment
        from experiment_classes.original_schnock import SchnockExperiment
        experiment = SchnockExperiment()
    elif args.experiment_type == "kmeans":
        from experiment_classes.kmeans_experiment import KmeansExperiment
        experiment = KmeansExperiment()
    elif args.experiment_type == "gradient":
        from experiment_classes.gradient_experiment import GradientExperiment
        experiment = GradientExperiment()
    elif args.experiment_type == "gabor_filter":
        from experiment_classes.gabor_filter_experiment import GaborFilterExperiment
        experiment = GaborFilterExperiment()

    output_directory = os.path.abspath(args.output_directory)
    experiment.set_output_directory(output_directory=output_directory)

    if args.folder_or_file == "folder":
        input_directory = os.path.abspath(args.input_directory)
        experiment.process_folder(
            input_directory=input_directory,
            output_directory=output_directory,
            file_extension=".JPG",
        )
    elif args.folder_or_file == "file":
        for file in tqdm(args.files):
            input_path = os.path.abspath(file)
            input_directory = os.path.dirname(input_path)
            experiment.set_input_directory(input_directory=input_directory)
            experiment.process_path(
                path=input_path,
                save_all_variants=True,
            )

# %%
