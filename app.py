from flask import Flask, render_template, request
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)
model = load_model("cloud_smoke_model.keras")
UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

def predict_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224,224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prob_asap = float(model.predict(img)[0][0])
    prob_awan = 1 - prob_asap

    persen_asap = round(prob_asap * 100, 2)
    persen_awan = round(prob_awan * 100, 2)

    return persen_asap, persen_awan

@app.route("/", methods=["GET", "POST"])
def index():
    uploaded_image = None
    persen_asap = None
    persen_awan = None

    if request.method == "POST":
        file = request.files["image"]
        if file:
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(filepath)
            persen_asap, persen_awan = predict_image(filepath)
            uploaded_image = filepath

    return render_template("index.html", asap=persen_asap, awan=persen_awan, image_path=uploaded_image)

if __name__ == "__main__":
    app.run(debug=True)