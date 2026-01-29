# models/xgboost_model.py
"""
XGBoost回帰モデル（正規化・型整備・余分な関数削除の修正版）
"""
from __future__ import annotations
import numpy as np

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    xgb = None

from .base_model import BaseModel


class XGBoostModel(BaseModel):
    """XGBoost回帰モデル"""

    def __init__(self) -> None:
        self.model = None
        self.name = "xgboost"
        self.available = XGB_AVAILABLE

    def suggest_hyperparameters(self, trial):
        """Optuna用ハイパーパラメータ提案（接頭辞 xgb_ を付与）"""
        if not self.available:
            return {}
        return {
            "xgb_n_estimators": trial.suggest_int("xgb_n_estimators", 50, 500),
            "xgb_max_depth": trial.suggest_int("xgb_max_depth", 3, 12),
            "xgb_learning_rate": trial.suggest_float("xgb_learning_rate", 0.01, 0.3, log=True),
            "xgb_subsample": trial.suggest_float("xgb_subsample", 0.6, 1.0),
            "xgb_colsample_bytree": trial.suggest_float("xgb_colsample_bytree", 0.6, 1.0),
            "xgb_colsample_bylevel": trial.suggest_float("xgb_colsample_bylevel", 0.6, 1.0),
            "xgb_reg_alpha": trial.suggest_float("xgb_reg_alpha", 0.0, 10.0),
            "xgb_reg_lambda": trial.suggest_float("xgb_reg_lambda", 0.0, 10.0),
            "xgb_gamma": trial.suggest_float("xgb_gamma", 0.0, 5.0),
            "xgb_min_child_weight": trial.suggest_int("xgb_min_child_weight", 1, 10),
        }

    @staticmethod
    def _normalize_params(params: dict) -> dict:
        """
        Optuna の xgb_* キーを XGBRegressor の正式キーへ1回だけ変換。
        既に正式キーならそのまま残す。
        """
        mapping = {
            "xgb_n_estimators": "n_estimators",
            "xgb_max_depth": "max_depth",
            "xgb_learning_rate": "learning_rate",
            "xgb_subsample": "subsample",
            "xgb_colsample_bytree": "colsample_bytree",
            "xgb_colsample_bylevel": "colsample_bylevel",
            "xgb_reg_alpha": "reg_alpha",
            "xgb_reg_lambda": "reg_lambda",
            "xgb_gamma": "gamma",
            "xgb_min_child_weight": "min_child_weight",
        }
        normalized = {}
        for k, v in params.items():
            normalized[mapping.get(k, k)] = v
        return normalized

    def build(self, **params):
        """モデル構築（キー正規化＋整数化）"""
        if not self.available:
            raise ImportError("XGBoost is not installed. Please `pip install xgboost`.")

        # 1) キー正規化（xgb_* → 正式キー）
        clean_params = self._normalize_params(params.copy())

        # 2) 整数パラメータの厳密化
        INT_PARAMS = {"n_estimators", "max_depth", "min_child_weight"}
        for p in list(INT_PARAMS):
            if p in clean_params:
                val = clean_params[p]
                if isinstance(val, (float, np.floating)):
                    clean_params[p] = int(round(val))
                elif isinstance(val, np.integer):
                    clean_params[p] = int(val)

        # 3) モデル構築
        self.model = xgb.XGBRegressor(
            n_estimators=clean_params.get("n_estimators", 100),
            max_depth=clean_params.get("max_depth", 6),
            learning_rate=clean_params.get("learning_rate", 0.1),
            subsample=clean_params.get("subsample", 0.8),
            colsample_bytree=clean_params.get("colsample_bytree", 0.8),
            colsample_bylevel=clean_params.get("colsample_bylevel", 1.0),
            reg_alpha=clean_params.get("reg_alpha", 0.0),
            reg_lambda=clean_params.get("reg_lambda", 1.0),
            gamma=clean_params.get("gamma", 0.0),
            min_child_weight=clean_params.get("min_child_weight", 1),
            objective="reg:squarederror",
            booster="gbtree",
            # ES: ★ CRÍTICO: Forzar a 1 para evitar conflictos de threading con OpenMP/MKL
            # EN: ★ CRITICAL: Force to 1 to avoid threading conflicts with OpenMP/MKL
            # JP: ★ 重要: OpenMP/MKLとのスレッド競合を避けるため1に固定する
            n_jobs=1,
            nthread=1,  # ★ XGBoost también usa nthread
            random_state=clean_params.get("random_state", 42),
            verbosity=0,
            importance_type="gain",
            # （必要なら）eval_metric="mae", tree_method="hist" などを追加可能
        )
        return self.model

    def get_model_type(self) -> str:
        return "tree"
