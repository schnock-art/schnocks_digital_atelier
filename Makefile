
save-env:
	conda env export > environment.yml
	pip3 freeze > requirements.txt

create-env:
	conda env create -n schnock --file environment.yml

run-gui:
	python schnock_atelier_ui.py

open-designer:
	qt6-tools designer