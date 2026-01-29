"""
LightGBM回帰モデル
"""
import numpy as np

try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False
    lgb = None

from .base_model import BaseModel


class LightGBMModel(BaseModel):
    """LightGBM回帰モデル"""

    def __init__(self):
        self.model = None
        self.name = "lightgbm"
        self.available = LGBM_AVAILABLE

    def suggest_hyperparameters(self, trial):
        """LightGBM用ハイパーパラメータ（接頭辞 lgbm_）"""
        if not self.available:
            return {}
        return {
            "lgbm_n_estimators": trial.suggest_int("lgbm_n_estimators", 50, 1000),
            "lgbm_num_leaves": trial.suggest_int("lgbm_num_leaves", 10, 300),
            "lgbm_learning_rate": trial.suggest_float("lgbm_learning_rate", 0.005, 0.3, log=True),
            "lgbm_feature_fraction": trial.suggest_float("lgbm_feature_fraction", 0.3, 1.0),
            "lgbm_bagging_fraction": trial.suggest_float("lgbm_bagging_fraction", 0.3, 1.0),
            "lgbm_bagging_freq": trial.suggest_int("lgbm_bagging_freq", 1, 10),
            "lgbm_min_child_samples": trial.suggest_int("lgbm_min_child_samples", 5, 100),
            "lgbm_reg_alpha": trial.suggest_float("lgbm_reg_alpha", 0.0, 10.0),
            "lgbm_reg_lambda": trial.suggest_float("lgbm_reg_lambda", 0.0, 10.0),
            "lgbm_min_gain_to_split": trial.suggest_float("lgbm_min_gain_to_split", 0.0, 1.0),
            "lgbm_subsample_for_bin": trial.suggest_int("lgbm_subsample_for_bin", 20000, 300000),
            "lgbm_min_data_in_bin": trial.suggest_int("lgbm_min_data_in_bin", 3, 50),
            "lgbm_max_bin": trial.suggest_int("lgbm_max_bin", 100, 500),
        }

    @staticmethod
    def _normalize_params(params: dict) -> dict:
        """
        lgbm_* → 正式キー に正規化
        """
        mapping = {
            "lgbm_n_estimators": "n_estimators",
            "lgbm_num_leaves": "num_leaves",
            "lgbm_learning_rate": "learning_rate",
            "lgbm_feature_fraction": "feature_fraction",
            "lgbm_bagging_fraction": "bagging_fraction",
            "lgbm_bagging_freq": "bagging_freq",
            "lgbm_min_child_samples": "min_child_samples",
            "lgbm_reg_alpha": "reg_alpha",
            "lgbm_reg_lambda": "reg_lambda",
            "lgbm_min_gain_to_split": "min_gain_to_split",
            "lgbm_subsample_for_bin": "subsample_for_bin",
            "lgbm_min_data_in_bin": "min_data_in_bin",
            "lgbm_max_bin": "max_bin",
        }
        out = {}
        for k, v in params.items():
            out[mapping.get(k, k)] = v
        return out

    def build(self, **params):
        """
        モデル構築（lgbm_* → 正式キーに正規化、整数パラメータはint化）
        """
        if not self.available:
            raise ImportError("LightGBM not available")

        # 1) キー正規化
        clean_params = self._normalize_params(params.copy())

        # 2) intにすべきキーは厳密にint化
        INT_PARAMS = {
            "n_estimators", "num_leaves", "bagging_freq",
            "min_child_samples", "subsample_for_bin", "min_data_in_bin", "max_bin",
        }
        for p in list(clean_params.keys()):
            if p in INT_PARAMS and clean_params[p] is not None:
                val = clean_params[p]
                if isinstance(val, (float, np.floating)):
                    clean_params[p] = int(round(val))
                elif isinstance(val, np.integer):
                    clean_params[p] = int(val)

        # 3) Regressor を構築
        self.model = lgb.LGBMRegressor(
            n_estimators=clean_params.get("n_estimators", 200),
            num_leaves=clean_params.get("num_leaves", 31),
            learning_rate=clean_params.get("learning_rate", 0.1),
            feature_fraction=clean_params.get("feature_fraction", 0.8),
            bagging_fraction=clean_params.get("bagging_fraction", 0.8),
            bagging_freq=clean_params.get("bagging_freq", 5),
            min_child_samples=clean_params.get("min_child_samples", 20),
            reg_alpha=clean_params.get("reg_alpha", 0.0),
            reg_lambda=clean_params.get("reg_lambda", 0.0),
            min_gain_to_split=clean_params.get("min_gain_to_split", 0.0),
            subsample_for_bin=clean_params.get("subsample_for_bin", 200000),
            min_data_in_bin=clean_params.get("min_data_in_bin", 3),
            max_bin=clean_params.get("max_bin", 255),
            objective="regression",
            metric="mae",
            boosting_type="gbdt",
            verbosity=-1,
            # ES: ★ CRÍTICO: Forzar a 1 para evitar conflictos de threading con OpenMP/MKL
            # EN: ★ CRITICAL: Force to 1 to avoid threading conflicts with OpenMP/MKL
            # JP: ★ 重要: OpenMP/MKLとのスレッド競合を避けるため1に固定する
            n_jobs=1,
            num_threads=1,  # ★ LightGBM también usa num_threads
            random_state=clean_params.get("random_state", 42),
            importance_type="gain",
        )
        return self.model

    def get_model_type(self):
        return "tree"
