from fastapi import FastAPI, UploadFile, File
import pandas as pd
import pickle
import io
import os

# Импортируем модель и регистрируем класс для pickle
import model
import __main__
__main__.DefaultCurveModel = model.DefaultCurveModel

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Загружаем модель
with open(os.path.join(BASE_DIR, "default_model.pkl"), "rb") as f:
    MODEL = pickle.load(f)

# ------------------------------------------------------------------
# ДОБАВЛЯЕМ атрибут feature_cols (его нет в сохранённом объекте)
# ------------------------------------------------------------------
MODEL.feature_cols = [
    'age_f',
    'industry_trailing_dr',
    'ros',
    'cash_to_assets_trend',
    'market_trailing_dr',
    'wc_to_assets_level',
    'roa_trend',
    'debt_to_equity_level',
    'op_margin_trend',
    'region_dr'
]

# ------------------------------------------------------------------
# НОРМАЛИЗУЕМ models и scalers в списки с индексацией 0..11
# (если они хранятся как словари с ключами 1..12)
# ------------------------------------------------------------------
if isinstance(MODEL.models, dict):
    # Сортируем ключи (обычно они 1..12) и превращаем в список
    MODEL.models = [MODEL.models[k] for k in sorted(MODEL.models.keys())]
if isinstance(MODEL.scalers, dict):
    MODEL.scalers = [MODEL.scalers[k] for k in sorted(MODEL.scalers.keys())]

# Проверяем, что получилось 12 элементов
if len(MODEL.models) != 12 or len(MODEL.scalers) != 12:
    raise ValueError(f"Ожидалось 12 моделей и 12 скейлеров, получено {len(MODEL.models)} и {len(MODEL.scalers)}")

@app.get("/")
def home():
    return {"status": "ok", "message": "Модель кредитного скоринга работает"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        if not contents:
            return {"error": "Файл пуст"}, 400

        df = pd.read_csv(io.BytesIO(contents))

        # Проверяем наличие всех нужных колонок
        missing = set(MODEL.feature_cols) - set(df.columns)
        if missing:
            return {"error": f"Отсутствуют колонки: {missing}"}, 400

        probs = MODEL.predict(df)

        result = []
        for i in range(len(df)):
            result.append({
                "company": i + 1,
                "pd_curve": probs[i].tolist(),
                "max_pd": float(probs[i][-1])
            })
        return result

    except Exception as e:
        return {"error": str(e)}, 500
