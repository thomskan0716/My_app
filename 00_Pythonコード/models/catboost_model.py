# models/catboost_model.py
"""
CatBoost回帰モデル（堅牢化版）
- Optuna の提案キー catb_* を正式キーへ正規化
- int系パラメータを強制的に int 化
- random_state → random_seed へ受け渡し
- sklearn 側に余計なキーを渡さない
"""
import numpy as np
from .base_model import BaseModel

# CatBoost の存在チェック
try:
    import catboost  # noqa: F401
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False


class CatBoostModel(BaseModel):
    """CatBoost回帰モデル"""

    def __init__(self):
        self.model = None
        self.name = "catboost"
        self.available = CATBOOST_AVAILABLE
        if not self.available:
            print("Warning: CatBoost not installed")

    def suggest_hyperparameters(self, trial):
        """Optuna用ハイパーパラメータ（接頭辞 catb_）"""
        if not self.available:
            return {}
        return {
            "catb_iterations": trial.suggest_int("catb_iterations", 50, 1000),
            "catb_depth": trial.suggest_int("catb_depth", 3, 12),
            "catb_learning_rate": trial.suggest_float("catb_learning_rate", 0.005, 0.3, log=True),
            "catb_l2_leaf_reg": trial.suggest_float("catb_l2_leaf_reg", 1.0, 30.0),
            "catb_bagging_temperature": trial.suggest_float("catb_bagging_temperature", 0.0, 5.0),
            "catb_random_strength": trial.suggest_float("catb_random_strength", 0.0, 10.0),
            "catb_border_count": trial.suggest_int("catb_border_count", 32, 255),
            "catb_grow_policy": trial.suggest_categorical(
                "catb_grow_policy", ["SymmetricTree", "Depthwise", "Lossguide"]
            ),
            "catb_min_data_in_leaf": trial.suggest_int("catb_min_data_in_leaf", 1, 50),
        }

    @staticmethod
    def _normalize_params(params: dict) -> dict:
        """
        catb_* → 正式キーへ正規化 + 型整備 + 受理キー以外の排除 + 乱数橋渡し
        """
        # 1) プレフィックス正規化（catb_* を正式キーへ）
        p = {}
        for k, v in params.items():
            if k.startswith("catb_"):
                p[k[5:]] = v  # 'catb_' を剥がす
            else:
                p[k] = v

        # 2) 乱数パラメータの橋渡し（上位は random_state を使う想定）
        if "random_seed" not in p and "random_state" in p:
            p["random_seed"] = p.pop("random_state")

        # 3) 整数パラメータは確実に int 化
        INT_PARAMS = {"iterations", "depth", "border_count", "min_data_in_leaf"}
        for key in list(INT_PARAMS):
            if key in p and p[key] is not None:
                val = p[key]
                if isinstance(val, (float, np.floating)):
                    p[key] = int(round(val))
                elif isinstance(val, np.integer):
                    p[key] = int(val)

        # 4) CatBoostRegressor が受理する主なキーのみ残す
        allowed = {
            "iterations", "depth", "learning_rate", "l2_leaf_reg",
            "bagging_temperature", "random_strength", "border_count",
            "grow_policy", "min_data_in_leaf", "loss_function", "eval_metric",
            "verbose", "random_seed", "thread_count"
        }
        p = {k: v for k, v in p.items() if k in allowed or k in {"random_seed"}}

        # 5) 既定値の補完（上位からの不足時に安心して動くように）
        p.setdefault("iterations", 200)
        p.setdefault("depth", 6)
        p.setdefault("learning_rate", 0.1)
        p.setdefault("l2_leaf_reg", 3.0)
        p.setdefault("bagging_temperature", 1.0)
        p.setdefault("random_strength", 1.0)
        p.setdefault("border_count", 128)
        p.setdefault("grow_policy", "SymmetricTree")
        p.setdefault("min_data_in_leaf", 1)
        p.setdefault("random_seed", 42)
        p.setdefault("verbose", False)
        p.setdefault("loss_function", "MAE")
        p.setdefault("eval_metric", "MAE")
        p.setdefault("thread_count", -1)

        return p

    def build(self, **params):
        """モデル構築（最適化結果を正しく反映 & 不正キー排除）"""
        if not self.available:
            raise ImportError("CatBoost not available. Please install: pip install catboost")

        from catboost import CatBoostRegressor  # import ここで

        clean = self._normalize_params(params)
        self.model = CatBoostRegressor(**clean)
        return self.model

    def get_model_type(self):
        return "tree"

    # 明示しておくと読みやすい（BaseModelにも実装あり）
    def fit(self, X, y, **kwargs):
        if self.model is None:
            self.build()
        return self.model.fit(X, y, **kwargs)

    def predict(self, X):
        return self.model.predict(X)
