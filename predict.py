import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("cloud_smoke_model.keras")

img = cv2.imread("test.jpg")
img = cv2.resize(img, (224,224))
img = img / 255.0
img = np.expand_dims(img, axis=0)

pred = model.predict(img)[0][0]

if pred > 0.5:
    print("Hasil: Asap")
else:
    print("Hasil: SAwan")