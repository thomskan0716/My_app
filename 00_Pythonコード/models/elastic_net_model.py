# models/elastic_net_model.py
"""
ElasticNet回帰モデル（接頭辞正規化・型整備・random_stateの条件付与）
"""
from __future__ import annotations
import numpy as np
from sklearn.linear_model import ElasticNet
from .base_model import BaseModel


class ElasticNetModel(BaseModel):
    """ElasticNet回帰モデル"""

    def __init__(self) -> None:
        self.model = None
        self.name = "elastic_net"
        self.available = True

    # ---------------- Optuna param space ----------------
    def suggest_hyperparameters(self, trial):
        """
        ElasticNet用ハイパーパラメータ（接頭辞: elastic_）
        ※ 接頭辞は build() で正式キーに正規化されます
        """
        return {
            "elastic_alpha": trial.suggest_float("elastic_alpha", 1e-5, 10, log=True),
            "elastic_l1_ratio": trial.suggest_float("elastic_l1_ratio", 0.01, 0.99),
            "elastic_fit_intercept": trial.suggest_categorical("elastic_fit_intercept", [True, False]),
            "elastic_precompute": trial.suggest_categorical("elastic_precompute", [True, False]),
            "elastic_max_iter": trial.suggest_int("elastic_max_iter", 1_000, 10_000),
            "elastic_tol": trial.suggest_float("elastic_tol", 1e-5, 1e-2, log=True),
            "elastic_selection": trial.suggest_categorical("elastic_selection", ["cyclic", "random"]),
        }

    # ---------------- Param normalization ----------------
    def _normalize_params(self, params: dict) -> dict:
        """
        elastic_* → 正式キーへ正規化、既定値補完、型整備、
        selection に応じた random_state の付与/削除、受理キー以外の除去
        """
        # 1) 接頭辞を外す
        p = {}
        for k, v in params.items():
            if k.startswith("elastic_"):
                p[k[8:]] = v  # 'elastic_' を外す
            else:
                p[k] = v

        # 2) 既定値補完
        p.setdefault("alpha", 0.1)
        p.setdefault("l1_ratio", 0.5)
        p.setdefault("fit_intercept", True)
        p.setdefault("precompute", False)
        p.setdefault("max_iter", 5000)
        p.setdefault("tol", 1e-4)
        p.setdefault("selection", "cyclic")

        # 3) 型整備（整数化）
        if "max_iter" in p and p["max_iter"] is not None:
            val = p["max_iter"]
            if isinstance(val, (float, np.floating)):
                p["max_iter"] = int(round(val))
            elif isinstance(val, np.integer):
                p["max_iter"] = int(val)

        # 4) selection に応じて random_state を付与/削除
        #    （ElasticNet は selection='random' のときのみ random_state を解釈する）
        rs = params.get("random_state", 42)
        if p.get("selection") == "random" and rs is not None:
            p["random_state"] = int(rs)
        else:
            p.pop("random_state", None)

        # 5) ElasticNet が受け付ける正式キーのみ残す
        allowed = {
            "alpha", "l1_ratio", "fit_intercept", "precompute",
            "max_iter", "tol", "selection", "random_state"
        }
        return {k: v for k, v in p.items() if k in allowed}

    # ---------------- Build ----------------
    def build(self, **params):
        """モデル構築（最適化結果を正しく反映 & 不正キー排除）"""
        clean = self._normalize_params(params)
        self.model = ElasticNet(**clean)
        return self.model

    # ---------------- Type hint for SHAP etc. ----------------
    def get_model_type(self) -> str:
        return "linear"

    # （任意）BaseModel のデフォルト実装でも可。明示しておく。
    def fit(self, X, y, **kwargs):
        if self.model is None:
            self.build()
        return self.model.fit(X, y, **kwargs)

    def predict(self, X):
        return self.model.predict(X)


