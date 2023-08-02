
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
