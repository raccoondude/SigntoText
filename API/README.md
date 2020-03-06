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

