import matplotlib
from sklearn.preprocessing import LabelBinarizer
import tensorflow
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import random
import pickle
import cv2
import os
import Main

def make_and_train_model(epochs, model_path, bin_path, dataset):
    data = []
    labels = []
    imagePaths = sorted(list(paths.list_images(dataset)))
    random.seed(42)
    random.shuffle(imagePaths)
    print("loading dataset.....")
    for imagePath in imagePaths:
        image = cv2.imread(imagePath)
        image = cv2.resize(image, (32, 32)).flatten()
        data.append(image)
        label = imagePath.split(os.path.sep)[-2]
        labels.append(label)
    data = np.array(data, dtype="float") / 225.0
    labels = np.array(labels)
    (trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)
    lb = LabelBinarizer()
    trainY = lb.fit_transform(trainY)
    testY = lb.transform(testY)
    model = tensorflow.keras.models.Sequential()
    model.add(tensorflow.keras.layers.Dense(1024, input_shape=(3072,), activation="sigmoid"))
    model.add(tensorflow.keras.layers.Dense(512, activation="relu"))
    model.add(tensorflow.keras.layers.Dense(300, activation="relu"))
    model.add(tensorflow.keras.layers.Dense(len(lb.classes_), activation="softmax"))
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.summary()
    Train = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=epochs, batch_size=32)
    predictions = model.predict(testX, batch_size=32)
    model.save(model_path)
    f = open(bin_path, "wb")
    f.write(pickle.dumps(lb))
    f.close()
