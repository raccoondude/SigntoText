import tensorflow
import pickle
import cv2

def make_Prediction(model, predict):
    Model = model.model
    lb  = model.database
    preds = Model.predict(predict)
    out  = preds.argmax(axis=1)[0]
    return lb.classes_[out]

def make_image_array(image_path):
    image = cv2.imread(image_path)
    image = image.copy()
    image = cv2.resize(image, (32, 32)).flatten()
    image = image.astype("float") / 225.0
    image = image.reshape((1, image.shape[0]))
    return image
