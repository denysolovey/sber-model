import numpy as np
import pandas as pd

N_HORIZONS = 12


class DefaultCurveModel:

    def __init__(self, models, scalers, feature_cols):
        self.models = models
        self.scalers = scalers
        self.feature_cols = feature_cols

    def predict(self, X_df):

        X = (
            X_df[self.feature_cols]
            .fillna(0)
            .to_numpy()
        )

        probs = np.column_stack([

            self.models[h]
            .predict_proba(
                self.scalers[h].transform(X)
            )[:, 1]

            for h in range(1, N_HORIZONS + 1)

        ])

        return np.maximum.accumulate(probs, axis=1)

    def get_coefficients(self, h):

        return dict(zip(
            self.feature_cols,
            self.models[h].coef_[0]
        ))

    def get_feature_importance(self, h):

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
