# models/gradient_boost_model.py
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from .base_model import BaseModel


class GradientBoostModel(BaseModel):
    """Gradient Boosting 回帰モデル（gb_* 接頭辞の正規化＆型整備）"""

    def __init__(self):
        self.model = None
        self.name = "gradient_boost"
        self.available = True

    # ---------------- Optuna param space ----------------
    def suggest_hyperparameters(self, trial):
        """Gradient Boosting 用ハイパーパラメータ（接頭辞: gb_）"""
        return {
            "gb_n_estimators": trial.suggest_int("gb_n_estimators", 50, 300),
            "gb_max_depth": trial.suggest_int("gb_max_depth", 3, 10),
            "gb_learning_rate": trial.suggest_float("gb_learning_rate", 0.01, 0.3, log=True),
            "gb_subsample": trial.suggest_float("gb_subsample", 0.6, 1.0),
            "gb_min_samples_split": trial.suggest_int("gb_min_samples_split", 2, 20),
            "gb_min_samples_leaf": trial.suggest_int("gb_min_samples_leaf", 1, 10),
        }

    # ---------------- Param normalization ----------------
    def _normalize_params(self, params: dict) -> dict:
        """
        gb_* → 正式キーに変換 + 既定値補完 + 型整備 + 受理キー以外の除去
        """
        # 1) 接頭辞を外す
        p = {}
        for k, v in params.items():
            if k.startswith("gb_"):
                p[k[3:]] = v  # 'gb_' を除去
            else:
                p[k] = v

        # 2) 既定値補完
        p.setdefault("n_estimators", 100)
        p.setdefault("max_depth", 3)
        p.setdefault("learning_rate", 0.1)
        p.setdefault("subsample", 0.8)
        p.setdefault("min_samples_split", 2)
        p.setdefault("min_samples_leaf", 1)

        # 3) 型整備（整数パラメータ）
        int_keys = {"n_estimators", "max_depth", "min_samples_split", "min_samples_leaf"}
        for k in list(p.keys()):
            if k in int_keys and p[k] is not None:
                val = p[k]
                if isinstance(val, (float, np.floating)):
                    p[k] = int(round(val))
                elif isinstance(val, np.integer):
                    p[k] = int(val)

        # 4) 数値の安全クリップ
        if "subsample" in p and p["subsample"] is not None:
            # 0 < subsample <= 1.0 が仕様
            p["subsample"] = float(np.clip(p["subsample"], 1e-8, 1.0))
        if "learning_rate" in p and p["learning_rate"] is not None:
            # >0 にしておく（0 は不可）
            p["learning_rate"] = float(max(p["learning_rate"], 1e-12))

        # 5) 受理キーのみ残す
        allowed = {
            "n_estimators", "max_depth", "learning_rate", "subsample",
            "min_samples_split", "min_samples_leaf", "random_state"
        }
        return {k: v for k, v in p.items() if k in allowed}

    # ---------------- Build ----------------
    def build(self, **params):
        """モデル構築（提案値を正しく反映）"""
        clean = self._normalize_params(params)
        self.model = GradientBoostingRegressor(
            n_estimators=clean.get("n_estimators", 100),
            max_depth=clean.get("max_depth", 3),
            learning_rate=clean.get("learning_rate", 0.1),
            subsample=clean.get("subsample", 0.8),
            min_samples_split=clean.get("min_samples_split", 2),
            min_samples_leaf=clean.get("min_samples_leaf", 1),
            random_state=clean.get("random_state", 42),
        )
        return self.model

    # ---------------- Type hint for SHAP, etc. ----------------
    def get_model_type(self):
        return "tree"

