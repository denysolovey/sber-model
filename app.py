import pickle
import numpy as np
from flask import Flask, request, jsonify
import os

app = Flask(__name__)

# Загружаем модель
with open('default_model.pkl', 'rb') as f:
    container = pickle.load(f)

models = container['models']          # словарь {2: модель, 3: модель, ...}
scaler = container['scaler']
features = container['features']
horizons = sorted([int(k) for k in models.keys()])   # [2,3,4,...,12]

@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({'status': 'ok', 'available_horizons': horizons})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Если приходит файл (как мы сейчас отправляем) — читаем CSV
        if 'file' in request.files:
            file = request.files['file']
            import pandas as pd
            df = pd.read_csv(file)
            # Берём первую строку (одна компания)
            row = df.iloc[0].to_dict()
            # Убедимся, что все признаки есть
            missing = [f for f in features if f not in row]
            if missing:
                return jsonify({'error': f'Missing features: {missing}'}), 400
            input_array = np.array([row[f] for f in features]).reshape(1, -1)
        else:
            # Если приходит JSON (старый формат) — тоже поддержим
            data = request.get_json()
            if not data:
                return jsonify({'error': 'No JSON received'}), 400
            missing = [f for f in features if f not in data]
            if missing:
                return jsonify({'error': f'Missing features: {missing}'}), 400
            input_array = np.array([data[f] for f in features]).reshape(1, -1)

        input_scaled = scaler.transform(input_array)

        results = {}
        for h in horizons:
            model = models[h]
            prob_default = model.predict_proba(input_scaled)[0, 1]
            results[f'horizon_{h}'] = float(prob_default)

        return jsonify(results)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
