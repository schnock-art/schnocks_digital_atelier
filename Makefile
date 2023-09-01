
save-env:
	conda env export > environment.yml
	pip list --format=freeze > requirements.txt

create-env:
	conda env create -n schnock --file environment.yml

run-gui:
	python src/schnock_atelier.py

open-designer:
	qt6-tools designer

create-exe:
	pyinstaller  src/schnock_atelier.py --add-data="src/main_ui.ui;." --noconfirm

execute-spec:
	pyinstaller schnock_atelier.spec --noconfirm

auto-format:
	black src/

execute-pylint:
	pylint --exit-zero --fail-on=F,E src/


### Run single experiments
run-single-experiment-schnock:
	python src/run_experiment_standalone.py schnock test_results/argparser/schnock file --file_path  data/DSC03351.JPG

run-single-experiment-gabor-filter:
	python src/run_experiment_standalone.py gabor_filter test_results/argparser/gabor_filter file --file_path  data/DSC03351.JPG

run-single-experiment-gradient:
	python src/run_experiment_standalone.py gradient test_results/argparser/gradient file --file_path  data/DSC03351.JPG

run-single-experiment-kmeans:
	python src/run_experiment_standalone.py kmeans test_results/argparser/kmeans file --file_path  data/DSC03351.JPG


### Run folder experiments
run-folder-experiment-schnock:
	python src/run_experiment_standalone.py schnock test_results/argparser/schnock folder --input_directory  data/

run-folder-experiment-gabor-filter:
	python src/run_experiment_standalone.py gabor_filter test_results/argparser/gabor_filter folder --input_directory  data/

run-folder-experiment-gradient:
	python src/run_experiment_standalone.py gradient test_results/argparser/gradient folder --input_directory  data/

run-folder-experiment-kmeans:
	python src/run_experiment_standalone.py kmeans test_results/argparser/kmeans folder --input_directory  data/
