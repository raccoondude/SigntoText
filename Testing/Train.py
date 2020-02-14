import matplotlib
matplotlib.use("Agg")

# import the necessary packages
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

print("--=|Trainer|=--")
E = input("[Input] Epochs>")

print("[OS] Loading dataset....")
data = []
labels = []
imagePaths = sorted(list(paths.list_images("Dataset")))
random.seed(42)
random.shuffle(imagePaths)
print("[OS] image paths loaded, Image resize/appending....")
count = 0

for imagePath in imagePaths:
    count = count + 1
    print("[Status] "+ str(len(imagePaths)) + "/"+str(count))
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (32, 32)).flatten()
    print(image)
    data.append(image)
    label = imagePath.split(os.path.sep)[-2]
    labels.append(label)

print("[OS] Finish!")
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

print("[OS] building model")
model = tensorflow.keras.models.Sequential()
model.add(tensorflow.keras.layers.Dense(1024, input_shape=(3072,), activation="sigmoid"))
model.add(tensorflow.keras.layers.Dense(512, activation="relu"))
model.add(tensorflow.keras.layers.Dense(300, activation="relu"))
model.add(tensorflow.keras.layers.Dense(len(lb.classes_), activation="softmax"))
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

print("[OS] Running training....")
Train = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=int(E), batch_size=32)

print("[OS] Evaluating.....")

predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
	predictions.argmax(axis=1), target_names=lb.classes_))
GRAPH = False
if GRAPH == True:
    N = np.arange(0, int(E))
    plt.style.use("ggplot")
    plt.figure()
    print("Created graph......(0/4)")
    plt.plot(N, Train.history["loss"], label="train_loss")
    print("Graphed loss (1/4)")
    plt.plot(N, Train.history["val_loss"], label="val_loss")
    print("Graphed val_loss (2/4)")
    plt.plot(N, Train.history["accuracy"], label="train_acc")
    print("Graphed accuray (3/4)")
    plt.plot(N, Train.history["val_accuracy"], label="accuracy")
    print("Graphed val_accuracy (4/4)")
    plt.title("Training Loss and Accuracy (Simple NN)")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.savefig(E+".png")

print("[OS] Saving")
model.save(E+".h5")
f = open(E+".h5.bin", "wb")
f.write(pickle.dumps(lb))
f.close()
