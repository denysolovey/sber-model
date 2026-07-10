import pickle
import numpy as np
from flask import Flask, request, jsonify
import os

# Создаём приложение
app = Flask(__name__)

# Загружаем модель и все настройки из файла
with open('default_model.pkl', 'rb') as f:
    container = pickle.load(f)

models = container['models']          # словарь моделей для разных горизонтов (2-12)
scaler = container['scaler']          # стандартизатор для признаков
features = container['features']      # список названий 10 признаков

# Получаем список горизонтов (обычно 2,3,4,5,6,7,8,9,10,11,12) и сортируем
horizons = sorted([int(k) for k in models.keys()])

# Эндпоинт для проверки работы сервиса (можно открыть в браузере)
@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({'status': 'ok', 'available_horizons': horizons})

# ГЛАВНЫЙ эндпоинт для предсказаний (сюда будет стучаться base44)
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Получаем JSON с данными от base44
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON received'}), 400

        # 2. Проверяем, что переданы все 10 признаков
        missing = [f for f in features if f not in data]
        if missing:
            return jsonify({'error': f'Missing features: {missing}'}), 400

        # 3. Превращаем полученные данные в массив чисел (строго в том порядке,
        #    в каком модель ожидает)
        input_array = np.array([data[f] for f in features]).reshape(1, -1)

        # 4. Применяем стандартизацию (масштабирование), как при обучении
        input_scaled = scaler.transform(input_array)

        # 5. Для каждого горизонта (2,3,...,12) получаем вероятность дефолта
        results = {}
        for h in horizons:
            model = models[h]  # достаём модель для конкретного горизонта
            # predict_proba возвращает [вероятность_класса_0, вероятность_класса_1]
            # Берём второе число (вероятность дефолта)
            prob_default = model.predict_proba(input_scaled)[0, 1]
            results[f'horizon_{h}'] = float(prob_default)

        # 6. Отправляем результат обратно в base44
        return jsonify(results)

    except Exception as e:
        # Если что-то пошло не так, отправляем ошибку
        return jsonify({'error': str(e)}), 500

# Запуск сервера (для Render очень важно использовать порт из переменных окружения)
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
