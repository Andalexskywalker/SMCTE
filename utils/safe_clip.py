# safe_clip.py
"""
ClipByLimitsTransformer — transformer picklável para clipping por coluna.

Uso:
    from safe_clip import ClipByLimitsTransformer
    clipper = ClipByLimitsTransformer(limits=clip_limits, feature_names=num_cols)
    # dentro de um Pipeline numérico:
    Pipeline([("imputer", SimpleImputer(...)),
              ("clip", clipper),
              ("scaler", StandardScaler())])

- limits: dict do tipo {"col": {"low": x, "high": y}, ...}
- feature_names: lista com a ORDEM das colunas numéricas à entrada do transformer
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class ClipByLimitsTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        limits: Optional[Dict[str, Dict[str, float]]] = None,
        feature_names: Optional[List[str]] = None,
    ):
        self.limits = limits or {}
        self.feature_names = feature_names

    def fit(self, X, y=None):
        # Guardar os nomes das features na ordem correta
        if hasattr(X, "columns"):
            self.feature_names_ = list(X.columns)
        else:
            if self.feature_names is not None:
                self.feature_names_ = list(self.feature_names)
            else:
                n = int(np.asarray(X).shape[1])
                self.feature_names_ = [f"f{i}" for i in range(n)]
        return self

    def transform(self, X):
        # DataFrame → devolve numpy (para encaixar bem no ColumnTransformer/Pipeline)
        if hasattr(X, "columns"):
            df = X.copy()
            for i, col in enumerate(self.feature_names_):
                lim = self.limits.get(col)
                if lim:
                    low = lim.get("low", None)
                    high = lim.get("high", None)
                    df.iloc[:, i] = df.iloc[:, i].clip(lower=low, upper=high)
            return df.to_numpy()

        # Numpy array
        X = np.asarray(X).copy()
        names = getattr(self, "feature_names_", self.feature_names or [f"f{i}" for i in range(X.shape[1])])
        for i, col in enumerate(names):
            lim = self.limits.get(col)
            if lim:
                low = lim.get("low", None)
                high = lim.get("high", None)
                if low is not None:
                    X[:, i] = np.maximum(X[:, i], low)
                if high is not None:
                    X[:, i] = np.minimum(X[:, i], high)
        return X

    # Opcional: ajuda algumas versões do sklearn a propagarem nomes
    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_ if input_features is None else input_features)
