from fastapi import FastAPI
import pickle

app = FastAPI()

with open("default_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/predict")
def predict(data: dict):
    return {
        "message": "API работает",
        "received": data
    }
