import pandas as pd
import joblib
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Machine Learning Prediction App")

# Load the saved models
stroke_model = joblib.load("stroke_model.joblib")
house_model = joblib.load("house_model.joblib")


@app.get("/", response_class=HTMLResponse)
def home():
    with open("index.html", "r", encoding="utf-8") as file:
        return file.read()


@app.post("/predict/stroke")
def predict_stroke(data: dict):
    features = pd.DataFrame([data])
    prediction = int(stroke_model.predict(features)[0])

    return {
        "prediction": prediction,
        "result": "Stroke predicted" if prediction == 1
                  else "No stroke predicted"
    }


@app.post("/predict/house")
def predict_house(data: dict):
    features = pd.DataFrame([data])
    prediction = float(house_model.predict(features)[0])

    return {
        "predicted_price": round(prediction, 2)
    }
