from tensorflow.keras.models import load_model
import pickle
import cv2

print("-=|Test|=-")
filePath = input("[input] file>")

image = cv2.imread(filePath)
output = image.copy()
image = cv2.resize(image, (32, 32)).flatten()

image = image.astype("float") / 255.0
image = image.reshape((1, image.shape[0]))

modelPath = input("[input] model>")
model = load_model(modelPath)
lb = pickle.loads(open(modelPath+".bin", "rb").read())

preds = model.predict(image)

i = preds.argmax(axis=1)[0]
label = lb.classes_[i]

print(preds)
print(label)
