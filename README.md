# Schnocks Digital Atelier

Schnocks Digital Atelier to play around with Photos.

# Try out the program!
### Just locate the dist\schnock_atelier\schnock_atelier.exe file in this repo and you're ready to go! :D

## Warning!
When using the webcam, some configurations can produce rapidly changing and flashy images, please avoid using the camera input if you are photosensitive, this could potetntially trigger epilepsy.

## Examples
### Gradient Experiment

The primary functionality revolves around manipulating images through the computation of gradients and merging different images based on these gradients.

![Alt text](https://github.com/schnock-art/schnocks_digital_atelier/tree/main/readme_images/gradient_gui.png?raw=true "Title")

### Gabor Filter Experiment
The primary functionality revolves around applying Gabor Filters with different parameters.

![Alt text](https://github.com/schnock-art/schnocks_digital_atelier/tree/main/readme_images/gabor_filter_gui.png?raw=true "Title")

### KMeans Experiment
The primary functionality revolves around applying KMeans to the original image and assigning each pixel to a cluster. The cluster centers are then used to create a new image. This creates the effect of the original image being drawn with a limited number of colors.

![Alt text](https://github.com/schnock-art/schnocks_digital_atelier/tree/main/readme_images/kmeans_gui.png?raw=true "Title")

### Schnock Experiment
The primary functionality revolves around applying a function to each pixel, which is based returns a new pixel where value of the dominant channel is increased. This gives the effect of more saturated colors.

![Alt text](https://github.com/schnock-art/schnocks_digital_atelier/tree/main/readme_images/schnock_original_gui.png?raw=true "Title")

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
