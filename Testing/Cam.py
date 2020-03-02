import tensorflow
import pickle
import cv2
import os

os.system("clear")
print("--=|Cam|=--")

modelPath = input("[model]>")

model = tensorflow.keras.models.load_model(modelPath)
lb = pickle.loads(open(modelPath+".bin", "rb").read())

try:
    while True:
        cam = cv2.VideoCapture(0)

        if cam.isOpened():
            owo, picture = cam.read()
            image = cv2.resize(picture, (32, 32)).flatten()
            image = image.astype("float") / 225.0
            image = image.reshape((1, image.shape[0]))
            preds = model.predict(image)
            print(lb.classes_[preds.argmax(axis=1)[0]])
            cam.release()
except KeyboardInterrupt:
    cam.release()

