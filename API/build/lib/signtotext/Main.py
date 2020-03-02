import tensorflow
import pickle
import cv2

class AI_model:
    def __init__(self, model, database):
        self.model = model
        self.database = database

def make_model_from_bin(model_path, binary_path):
    model = tensorflow.keras.models.load_model(model_path)
    binary = pickle.loads(open(binary_path, "rb").read())
    Out = AI_model(model, binary)
    return Out
