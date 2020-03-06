# Sign to text API

Signtotext API is the main project, that will allow to create models with new datasets to predictions, and other functions like image convertion

### Installing

`python3 setup.py install`

### Importing

Importing the project is somewhat different to how most librarys are imported

To import the objects used for the library use

`from signtotext.Main import *`

To import the functions used in building a model use:

`from signtotext.Train import *`

To import functions used in predictions use:

`from signtotext.Predict import *`

### Main

`signtotext.Main` or Main.py holds objects used to store model data

AI_Model holds the model used for predictions

make_model_from_bin is a function for creating AI_Models from files, its output will be a AI_Model object

### Train

`signtotext.Train` or Train.py holds functions for creating models and training them for use

###### make_and_train_model

`make_and_train_model(epochs, model_path, bin_path, graph_path, dataset, log)`

Log should be a boolean, dataset should be a path to your datasets

###### datasets

A dataset should be as the following

    -----------------------------Dataset
    |
    |
    |------------------------------------Object A
    |                           |-----------------------ObjectA.png
    |                           |-----------------------ObjectA2.png
    |------------------------------------Object B
                                |-----------------------ObjectB.png
                                |-----------------------ObjectB2.png
    
### Predict

`signtotext.Predict` or Predict.py holds functions for predictions

#### make_prediction

`make_prediction` makes predicitions based on the model and array sent to the function

#### make_image_array

`make_image_array` makes a array from an image path, used for `make_prediction`