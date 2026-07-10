import pickle
import numpy as np
from flask import Flask, request, jsonify
import os

app = Flask(__name__)

# Загружаем модель и все настройки
with open('default_model.pkl', 'rb') as f:
    container = pickle.load(f)

models = container['models']          # словарь, ключи — горизонты (2..12)
scaler = container['scaler']          # стандартизатор
features = container['features']      # список названий признаков

# Получаем список горизонтов (2,3,4,...,12) и сортируем
horizons = sorted([int(k) for k in models.keys()])

# Эндпоинт для проверки работы (GET-запрос)
@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({'status': 'ok', 'available_horizons': horizons})

# Главный эндпоинт для предсказаний (POST-запрос)
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON received'}), 400

        # Проверяем, что переданы все необходимые признаки
        missing = [f for f in features if f not in data]
        if missing:
            return jsonify({'error': f'Missing features: {missing}'}), 400

        # Превращаем данные в массив в порядке признаков
        input_array = np.array([data[f] for f in features]).reshape(1, -1)
        # Масштабируем (стандартизация)
        input_scaled = scaler.transform(input_array)

        # Для каждого горизонта получаем вероятность дефолта
        results = {}
        for h in horizons:
            model = models[h]                     # берём модель для горизонта h
            prob_default = model.predict_proba(input_scaled)[0, 1]
            results[f'horizon_{h}'] = float(prob_default)

        return jsonify(results)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
