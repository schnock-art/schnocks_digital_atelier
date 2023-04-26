# Schnocks Digital Atelier

Schnocks Digital Atelier to play around with Photos.

## Warning!
When using the webcam, some configurations can produce rapidly changing and flashy images, please avoid using the camera input if you are photosensitive, this could potetntially trigger epilepsy.

## Create Environment
#### In terminal
```
conda env create -n schnock --file environment.yml
```
#### Alternatively with make
```
make create-env
```

## Run GUI
```
python schnock_atelier_ui.py
```
#### Alternatively with make
```
make run-gui
```


## To edit the GUI it's best to Use QtDesigner
```
qt6-tools designer
```
#### Alternatively with make
```
make open-designer
```