# models/ridge_model.py
"""
Ridge回帰モデル（接頭辞正規化・型整備・random_stateの条件付与）
"""
from __future__ import annotations
import numpy as np
from sklearn.linear_model import Ridge
from .base_model import BaseModel


class RidgeModel(BaseModel):
    """Ridge回帰モデル"""

    def __init__(self):
        self.model = None
        self.name = "ridge"
        self.available = True

    # ---------------- Optuna param space ----------------
    def suggest_hyperparameters(self, trial):
        """Ridge用ハイパーパラメータ（接頭辞 ridge_）"""
        return {
            "ridge_alpha": trial.suggest_float("ridge_alpha", 1e-5, 1e4, log=True),
            "ridge_solver": trial.suggest_categorical(
                "ridge_solver", ["auto", "svd", "cholesky", "lsqr", "sag", "saga"]
            ),
            "ridge_fit_intercept": trial.suggest_categorical("ridge_fit_intercept", [True, False]),
            "ridge_max_iter": trial.suggest_int("ridge_max_iter", 1_000, 10_000),
            "ridge_tol": trial.suggest_float("ridge_tol", 1e-5, 1e-2, log=True),
        }

    # ---------------- Param normalization ----------------
    def _normalize_params(self, params: dict) -> dict:
        """
        ridge_* → 正式キーへ正規化、型の健全化、solverに応じたrandom_state付与を管理
        """
        # 1) プレフィックス剥がし
        p = {}
        for k, v in params.items():
            if k.startswith("ridge_"):
                p[k[6:]] = v  # 'ridge_' を削除
            else:
                p[k] = v

        # 2) 既定値補完
        p.setdefault("alpha", 1.0)
        p.setdefault("solver", "auto")
