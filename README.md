# Sign to text

A small project that has taken me over a few months,

Its a small API that can take an array, image path, or 32x32x3 NumPy array and pass it through a AI that can detect if someone is attempting to spell in ASL

Sadly, its very......well, useless. It would be like having to spell out each word in regular speach to text, because of this, don't see this as what it is now, but how it can be changed to be better.

### Basics

Now the API is in testing, im far from making something usuable yet. However I have created the basics, a script that can generate and train models and another than can pass images through them to create an output


### Directorys

Datasets/ is a list of datasets used in the project, its the main reason this repo is so huge and I will make a fork later that exculdes this directory. All of the datasets were taken from Kraggle and are credited in the directorys README

Testing/ holds test scripts that will later be used to build the API. Due to the state of the project, this is all we have to show that it has been done.

Models/ holds a .bin and .h5 of pre-trained models

### Models

Models are stored as two files:

X.bin and X.h5, with the X being how many ephoces have been ran through it. 

X.h5 holds the model itself, while X.bin holds the label data

### Contributing

If you wish to add anything, from a dataset you created, or anything else, PLEASE make a pull request, im begging you!

### Dependences

sklearn, tensorflow, imutils, numpy, cv2, matplotlib

###Updates

0.1 Finially made this repo
