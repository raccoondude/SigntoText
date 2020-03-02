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

def make_and_train_model(epochs, model_path, bin_path, graph_path, dataset, log):
    data = []
    labels = []
    imagePaths = sorted(list(paths.list_images(dataset)))
    random.seed(42)
    random.shuffle(imagePaths)
    if (log == True):
        print("loading dataset.....")
    click = 0
    for imagePath in imagePaths:
        click = click + 1
        if (log == True):
            print(str(click)+"/"+str(len(imagePaths)))
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
    if (graph_path != False):
        N = np.arange(0, int(epochs))
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(N, Train.history["loss"], label="train_loss")
        plt.plot(N, Train.history["val_loss"], label="val_loss")
        plt.plot(N, Train.history["accuracy"], label="train_acc")
        plt.plot(N, Train.history["val_accuracy"], label="accuracy")
        plt.title("Training Loss and Accuracy (Simple NN)")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend()
        plt.savefig(E+".png")
    model.save(model_path)
    f = open(bin_path, "wb")
    f.write(pickle.dumps(lb))
    f.close()
