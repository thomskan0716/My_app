# models/lasso_model.py
"""
Lasso回帰モデル（バグ修正：パラメータ正規化＋型整備）
"""
from __future__ import annotations
import numpy as np
from sklearn.linear_model import Lasso
from .base_model import BaseModel


class LassoModel(BaseModel):
    """Lasso回帰モデル"""

    def __init__(self) -> None:
        self.model = None
        self.name = "lasso"
        self.available = True

    # ---------------- Optuna param space ----------------
    def suggest_hyperparameters(self, trial):
        """
        Lasso用ハイパーパラメータ（接頭辞: lasso_）
        ※ 接頭辞は build() で正式キーに正規化されます
        """
        return {
            "lasso_alpha": trial.suggest_float("lasso_alpha", 1e-4, 1e-1, log=True),
            "lasso_max_iter": trial.suggest_int("lasso_max_iter", 1_000, 10_000),
            "lasso_tol": trial.suggest_float("lasso_tol", 1e-6, 1e-3, log=True),
            "lasso_fit_intercept": trial.suggest_categorical("lasso_fit_intercept", [True, False]),
            "lasso_selection": trial.suggest_categorical("lasso_selection", ["cyclic", "random"]),
        }

    # ---------------- Param normalization ----------------
    def _normalize_params(self, params: dict) -> dict:
        """
        lasso_* → 正式キーへ正規化 + 既定値補完 + 型整備 + 受理キーのフィルタ
        """
        # 1) 接頭辞を外す（lasso_alpha -> alpha など）
        p = {}
        for k, v in params.items():
            if k.startswith("lasso_"):
                p[k.replace("lasso_", "", 1)] = v
            else:
                p[k] = v

        # 2) 既定値の補完
        p.setdefault("alpha", 1.0)
        p.setdefault("max_iter", 5000)
        p.setdefault("tol", 1e-4)
        p.setdefault("fit_intercept", True)
        p.setdefault("selection", "cyclic")

        # 3) 型整備（整数化）
        for key in ("max_iter",):
            if key in p and p[key] is not None:
                val = p[key]
                if isinstance(val, (float, np.floating)):
                    p[key] = int(round(val))
                elif isinstance(val, np.integer):
                    p[key] = int(val)

        # 4) Lasso が受け付ける主要キーのみ残す（余計なキーを除去）
        allowed = {
            "alpha", "max_iter", "tol", "fit_intercept", "selection", "random_state"
        }
        p = {k: v for k, v in p.items() if k in allowed}
        return p

    # ---------------- Build ----------------
    def build(self, **params):
        """モデル構築（提案値を正しく反映）"""
        p = self._normalize_params(params)
        self.model = Lasso(
            alpha=p.get("alpha", 1.0),
            max_iter=p.get("max_iter", 5000),
            tol=p.get("tol", 1e-4),
            fit_intercept=p.get("fit_intercept", True),
            selection=p.get("selection", "cyclic"),
            # selection='random' のときのみ意味を持つが、常時渡しても問題ない
            random_state=p.get("random_state", 42),
        )
        return self.model

    # ---------------- Type hint for SHAP etc. ----------------
    def get_model_type(self) -> str:
        return "linear"

