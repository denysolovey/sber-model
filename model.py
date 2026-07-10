import numpy as np
import pandas as pd

N_HORIZONS = 12  # количество горизонтов прогноза (месяцев/кварталов)


class DefaultCurveModel:
    """
    Модель для прогнозирования кривой дефолта.
    Ожидает, что при инициализации переданы:
      - models: список из N_HORIZONS обученных классификаторов (sklearn-совместимых)
      - scalers: список из N_HORIZONS стандартизаторов (например, StandardScaler)
      - feature_cols: список названий признаков в том порядке, в котором они используются
    """

    def __init__(self, models, scalers, feature_cols):
        self.models = models
        self.scalers = scalers
        self.feature_cols = feature_cols

    def predict(self, X_df):
        """
        Принимает pandas DataFrame с признаками.
        Возвращает массив вероятностей дефолта для каждого горизонта (12 значений на компанию)
        с кумулятивным максимумом (невозрастающая кривая).
        """
        # Проверяем наличие атрибута feature_cols (на случай, если его нет в загруженном объекте)
        if not hasattr(self, 'feature_cols') or self.feature_cols is None:
            raise AttributeError(
                "Объект модели не содержит 'feature_cols'. "
                "Убедитесь, что при загрузке модели атрибут задан."
            )

        # Берём только нужные колонки, заполняем пропуски нулями и преобразуем в numpy
        X = X_df[self.feature_cols].fillna(0).to_numpy()

        # Для каждого горизонта (индексы 0..11) получаем вероятности
        probs = np.column_stack([
            self.models[h].predict_proba(self.scalers[h].transform(X))[:, 1]
            for h in range(N_HORIZONS)   # исправлено: теперь от 0 до 11, а не от 1 до 12
        ])

        # Кумулятивный максимум (вероятности не убывают со временем)
        return np.maximum.accumulate(probs, axis=1)

    def get_coefficients(self, h):
        """
        Возвращает словарь коэффициент-значение для линейной модели на горизонте h (0-индексация).
        Работает только для моделей, имеющих coef_ (например, логистическая регрессия).
        """
        if h < 0 or h >= N_HORIZONS:
            raise ValueError(f"h должно быть от 0 до {N_HORIZONS-1}")
        return dict(zip(self.feature_cols, self.models[h].coef_[0]))

    def get_feature_importance(self, h):
        """
        Возвращает DataFrame с важностью признаков (по модулю коэффициента)
        для горизонта h, отсортированный по убыванию.
        """
        if h < 0 or h >= N_HORIZONS:
            raise ValueError(f"h должно быть от 0 до {N_HORIZONS-1}")
        coef = self.models[h].coef_[0]
        return (
            pd.DataFrame({
                "feature": self.feature_cols,
                "coef": coef
            })
            .assign(abs_coef=lambda d: d.coef.abs())
            .query("coef != 0")
            .sort_values("abs_coef", ascending=False)
            .drop(columns="abs_coef")
            .reset_index(drop=True)
        )
