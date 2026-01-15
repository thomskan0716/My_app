# models/random_forest_model.py
"""
RandomForest回帰モデル（堅牢化版）
- Optuna 提案キー: rf_* を正式キーへ正規化
- int系/float系の型を強制整形
- bootstrap と max_samples の整合性チェック
- 不正キーは sklearn に渡さない
"""
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from .base_model import BaseModel


class RandomForestModel(BaseModel):
    """RandomForest回帰モデル"""

    def __init__(self):
        self.model = None
        self.name = "random_forest"
        self.available = True  # 明示（他モデルと揃える）

    def suggest_hyperparameters(self, trial):
        """RandomForest用ハイパーパラメータ（接頭辞: rf_）"""
        bootstrap = trial.suggest_categorical('rf_bootstrap', [True, False])
        params = {
            'rf_n_estimators': trial.suggest_int('rf_n_estimators', 50, 500),
            'rf_max_depth': trial.suggest_int('rf_max_depth', 3, 50),
            'rf_min_samples_split': trial.suggest_int('rf_min_samples_split', 2, 20),
            'rf_min_samples_leaf': trial.suggest_int('rf_min_samples_leaf', 1, 15),
            'rf_max_features': trial.suggest_categorical('rf_max_features', ['sqrt', 'log2', 0.3, 0.5, 0.7, 0.9]),
            'rf_bootstrap': bootstrap,
            'rf_min_impurity_decrease': trial.suggest_float('rf_min_impurity_decrease', 0.0, 0.1),
        }
        if bootstrap:
            params['rf_max_samples'] = trial.suggest_float('rf_max_samples', 0.5, 1.0)
        else:
            params['rf_max_samples'] = None
        return params

    @staticmethod
    def _normalize_params(params: dict) -> dict:
        """
        rf_* → 正式キーへ正規化 & 型の健全化 & 整合性チェック
        """
        # 1) プレフィックス正規化
        p = {}
        for k, v in params.items():
            if k.startswith('rf_'):
                p[k[3:]] = v  # 'rf_' を剥がす
            else:
                p[k] = v

        # 2) 整数パラメータは確実に int 化
        int_keys = {'n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf'}
        for k in list(int_keys):
            if k in p and p[k] is not None:
                val = p[k]
                if isinstance(val, (float, np.floating)):
                    p[k] = int(round(val))
                elif isinstance(val, np.integer):
                    p[k] = int(val)

        # 3) max_features は sklearn の想定値のみ許可
        #    - 'sqrt' / 'log2' / None / int / float(0,1]
        if 'max_features' in p:
            mf = p['max_features']
            if isinstance(mf, (float, np.floating)):
                # 0.0 < mf <= 1.0 のみ許可（それ以外は 'sqrt' にフォールバック）
                if not (0.0 < float(mf) <= 1.0):
                    p['max_features'] = 'sqrt'
            elif isinstance(mf, (int, np.integer)):
                # 0 以下は無効 → フォールバック
                if int(mf) <= 0:
                    p['max_features'] = 'sqrt'
            elif mf not in ('sqrt', 'log2', None):
                p['max_features'] = 'sqrt'

        # 4) bootstrap と max_samples の整合性
        bootstrap = bool(p.get('bootstrap', True))
        ms = p.get('max_samples', None)
        if not bootstrap or ms is None:
            p.pop('max_samples', None)
        else:
            # int の場合: >0 のみ許可
            if isinstance(ms, (int, np.integer)):
                if ms <= 0:
                    p.pop('max_samples', None)
            # float の場合: 0.0 < ms <= 1.0 のみ許可
            elif isinstance(ms, (float, np.floating)):
                if not (0.0 < float(ms) <= 1.0):
                    p.pop('max_samples', None)
            else:
                # 型が想定外なら削除
                p.pop('max_samples', None)

        # 5) 受理キー以外は落とす（sklearn への余計な引数を排除）
        allowed = {
            'n_estimators', 'criterion', 'max_depth', 'min_samples_split', 'min_samples_leaf',
            'min_weight_fraction_leaf', 'max_features', 'max_leaf_nodes', 'min_impurity_decrease',
            'bootstrap', 'oob_score', 'n_jobs', 'random_state', 'verbose', 'warm_start',
            'ccp_alpha', 'max_samples'
        }
        p = {k: v for k, v in p.items() if k in allowed}
        # 既定値を補完（使い勝手向上）
        p.setdefault('random_state', 42)
        p.setdefault('n_jobs', -1)
        p.setdefault('verbose', 0)
        return p

    def build(self, **params):
        """モデル構築（rf_* を正しく解釈）"""
        cleaned = self._normalize_params(params)
        self.model = RandomForestRegressor(**cleaned)
        return self.model

    def get_model_type(self):
        return 'tree'

    # fit/predict は BaseModel にあるが、明示的に持っておくと読みやすい
    def fit(self, X, y, **kwargs):
        if self.model is None:
            self.build()
        return self.model.fit(X, y, **kwargs)

    def predict(self, X):
        return self.model.predict(X)
