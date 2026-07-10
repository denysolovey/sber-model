from fastapi import FastAPI, UploadFile, File
import pandas as pd
import pickle
import io

import model

app = FastAPI()

with open("default_model.pkl", "rb") as f:
    MODEL = pickle.load(f)


@app.get("/")
def home():
    return {
        "status": "ok",
        "message": "Модель кредитного скоринга работает"
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    contents = await file.read()

    df = pd.read_csv(io.BytesIO(contents))

    probs = MODEL.predict(df)

    result = []

    for i in range(len(df)):

        result.append({

            "company": i + 1,

            "pd_curve": probs[i].tolist(),

            "max_pd": float(probs[i][-1])

        })

    return result
