from typing import Dict, Any, Optional, Tuple, Set
import numpy as np

# オプショナル依存
has_lgbm = False
has_xgb = False
try:
    import lightgbm as lgb
    has_lgbm = True
except Exception:
    pass
try:
    import xgboost as xgb
    has_xgb = True
except Exception:
    pass

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


class ModelFactoryCLS:
    """
    分類モデルのハイパラ探索 & モデル生成
    - LightGBM: 公式名で統一（alias 不使用）+ 警告抑制
    - XGBoost/RandomForest/Logistic: config 未定義キーは安全な既定値で補完
    """

    @staticmethod
    def _range(conf: Dict[str, Any], key: str, default: Tuple[float, float]) -> Tuple[float, float]:
        """config[key] がなければ default を返す（KeyError防止）"""
        return tuple(conf.get(key, default))  # type: ignore

    @staticmethod
    def suggest_hyperparams(name: str, trial, config: Dict[str, Any]) -> Dict[str, Any]:
        # ===== LightGBM =====
        if name == "lightgbm" and has_lgbm:
            ne_lo, ne_hi = ModelFactoryCLS._range(config, "n_estimators_range", (200, 1200))
            lr_lo, lr_hi = ModelFactoryCLS._range(config, "learning_rate_range", (0.01, 0.2))
            nl_lo, nl_hi = ModelFactoryCLS._range(config, "num_leaves_range", (31, 255))
            mdl_lo, mdl_hi = ModelFactoryCLS._range(config, "min_data_in_leaf_range", (10, 60))
            mgs_lo, mgs_hi = ModelFactoryCLS._range(config, "min_gain_to_split_range", (0.0, 0.02))
            ff_lo, ff_hi = ModelFactoryCLS._range(config, "feature_fraction_range", (0.6, 1.0))
            bf_lo, bf_hi = ModelFactoryCLS._range(config, "bagging_fraction_range", (0.7, 1.0))
            bq_lo, bq_hi = ModelFactoryCLS._range(config, "bagging_freq_range", (0, 5))
            l1_lo, l1_hi = ModelFactoryCLS._range(config, "lambda_l1_range", (0.0, 3.0))
            l2_lo, l2_hi = ModelFactoryCLS._range(config, "lambda_l2_range", (0.0, 3.0))
            md_lo, md_hi = ModelFactoryCLS._range(config, "max_depth_range", (3, 12))

            return {
                "n_estimators": trial.suggest_int("n_estimators", int(ne_lo), int(ne_hi)),
                "learning_rate": trial.suggest_float("learning_rate", float(lr_lo), float(lr_hi), log=True),
                "num_leaves": trial.suggest_int("num_leaves", int(nl_lo), int(nl_hi)),
                "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", int(mdl_lo), int(mdl_hi)),
                "min_gain_to_split": trial.suggest_float("min_gain_to_split", float(mgs_lo), float(mgs_hi)),
                "feature_fraction": trial.suggest_float("feature_fraction", float(ff_lo), float(ff_hi)),
                "bagging_fraction": trial.suggest_float("bagging_fraction", float(bf_lo), float(bf_hi)),
                "bagging_freq": trial.suggest_int("bagging_freq", int(bq_lo), int(bq_hi)),
                "lambda_l1": trial.suggest_float("lambda_l1", float(l1_lo), float(l1_hi)),
                "lambda_l2": trial.suggest_float("lambda_l2", float(l2_lo), float(l2_hi)),
                "max_depth": trial.suggest_int("max_depth", int(md_lo), int(md_hi)),
                # 公式名のみ。alias（min_child_samples, subsample, colsample_bytree等）は使わない
            }

        # ===== XGBoost =====
        if name == "xgboost" and has_xgb:
            ne_lo, ne_hi = ModelFactoryCLS._range(config, "n_estimators_range", (100, 600))
            lr_lo, lr_hi = ModelFactoryCLS._range(config, "learning_rate_range", (0.01, 0.3))
            md_lo, md_hi = ModelFactoryCLS._range(config, "max_depth_range", (3, 10))
            ss_lo, ss_hi = ModelFactoryCLS._range(config, "subsample_range", (0.6, 1.0))
            cs_lo, cs_hi = ModelFactoryCLS._range(config, "colsample_bytree_range", (0.6, 1.0))
            ra_lo, ra_hi = ModelFactoryCLS._range(config, "reg_alpha_range", (0.0, 5.0))
            rl_lo, rl_hi = ModelFactoryCLS._range(config, "reg_lambda_range", (0.0, 5.0))
            mcw_lo, mcw_hi = ModelFactoryCLS._range(config, "min_child_weight_range", (0.0, 10.0))
            gm_lo, gm_hi = ModelFactoryCLS._range(config, "gamma_range", (0.0, 5.0))

            return {
                "n_estimators": trial.suggest_int("n_estimators", int(ne_lo), int(ne_hi)),
                "learning_rate": trial.suggest_float("learning_rate", float(lr_lo), float(lr_hi), log=True),
                "max_depth": trial.suggest_int("max_depth", int(md_lo), int(md_hi)),
                "subsample": trial.suggest_float("subsample", float(ss_lo), float(ss_hi)),
                "colsample_bytree": trial.suggest_float("colsample_bytree", float(cs_lo), float(cs_hi)),
                "reg_alpha": trial.suggest_float("reg_alpha", float(ra_lo), float(ra_hi)),
                "reg_lambda": trial.suggest_float("reg_lambda", float(rl_lo), float(rl_hi)),
                "min_child_weight": trial.suggest_float("min_child_weight", float(mcw_lo), float(mcw_hi)),
                "gamma": trial.suggest_float("gamma", float(gm_lo), float(gm_hi)),
            }

        # ===== RandomForest =====
        if name == "random_forest":
            ne_lo, ne_hi = ModelFactoryCLS._range(config, "n_estimators_range", (200, 800))
            md_lo, md_hi = ModelFactoryCLS._range(config, "max_depth_range", (3, 20))
            mss_lo, mss_hi = ModelFactoryCLS._range(config, "min_samples_split_range", (2, 20))
            msl_lo, msl_hi = ModelFactoryCLS._range(config, "min_samples_leaf_range", (1, 20))
            mf_lo, mf_hi = ModelFactoryCLS._range(config, "max_features_range", (0.4, 1.0))

            return {
                "n_estimators": trial.suggest_int("n_estimators", int(ne_lo), int(ne_hi)),
                "max_depth": trial.suggest_int("max_depth", int(md_lo), int(md_hi)),
                "min_samples_split": trial.suggest_int("min_samples_split", int(mss_lo), int(mss_hi)),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", int(msl_lo), int(msl_hi)),
                "max_features": trial.suggest_float("max_features", float(mf_lo), float(mf_hi)),
            }

        # ===== Logistic（saga + elasticnet）=====
        if name == "logistic":
            c_lo, c_hi = ModelFactoryCLS._range(config, "C_range", (1e-3, 1e2))
            l1_lo, l1_hi = ModelFactoryCLS._range(config, "l1_ratio_range", (0.0, 1.0))
            return {
                "C": trial.suggest_float("C", float(c_lo), float(c_hi), log=True),
                "l1_ratio": trial.suggest_float("l1_ratio", float(l1_lo), float(l1_hi)),
            }

        # 未サポート or パッケージ無し → ロジスティックにフォールバック
        return {"C": 1.0, "l1_ratio": 0.0}

    @staticmethod
    def build(name: str, params: Dict[str, Any]):
        """name と params から学習器を構築"""
        if name == "lightgbm" and has_lgbm:
            # 公式パラメータ名のみ + 警告抑制パラメータ追加
            return lgb.LGBMClassifier(
                objective="binary",
                n_jobs=-1,
                random_state=42,
                verbosity=-1,           # 警告を抑制
                force_col_wise=True,    # 列方向処理を強制（オーバーヘッド警告対策）
                **params,
            )

        if name == "xgboost" and has_xgb:
            return xgb.XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                n_jobs=-1,
                random_state=42,
                verbosity=0,            # XGBoostも警告抑制
                **params,
            )

        if name == "random_forest":
            return RandomForestClassifier(
                n_jobs=-1,
                random_state=42,
                **params,
            )

        # Logistic（saga + elasticnet 固定）
        # l1_ratioが指定されていない場合は0.5をデフォルト値として使用
        if 'l1_ratio' not in params:
            params = params.copy()  # 元のdictを変更しないようにコピー
            params['l1_ratio'] = 0.5
        
        return LogisticRegression(
            solver="saga",
            penalty="elasticnet",
            max_iter=5000,
            n_jobs=-1,
            **params,
        )
