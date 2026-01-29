# -*- coding: utf-8 -*-
"""
1予測モデル構築 - 完全版パイプライン（安定化・バグ修正版）
- X: スケーリングは前処理内で一回のみ（外側 RobustScaler 撤去）
- Y: AdaptiveTargetTransformer で log/sqrt 変換+確実な逆変換
- DCV: 完全リーク対策、早期returnバグ修正
- import 両対応、SHAP フォールバック、clean_model_params徹底
"""

import sys
import os
from pathlib import Path
import gc
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import joblib
import optuna
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import json
from datetime import datetime

# ---------------- プロジェクトパス設定 ----------------
PROJECT_ROOT = Path.cwd()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PYTHON_CODE_FOLDER = PROJECT_ROOT / "00_Pythonコード"
if str(PYTHON_CODE_FOLDER) not in sys.path:
    sys.path.insert(0, str(PYTHON_CODE_FOLDER))

# ---------------- 設定・フォント ----------------
from config import Config
plt.rcParams['font.family'] = Config.JAPANESE_FONT_FAMILY
plt.rcParams['axes.unicode_minus'] = Config.JAPANESE_FONT_UNICODE_MINUS

# ---------------- モジュール import 両対応 ----------------
# 前処理・ユーティリティ
try:
    from core.preprocessing import EnhancedPreprocessor, AdvancedFeatureSelector
    from core.utils import fix_seed, choose_transform, clean_model_params
except Exception:
    from preprocessing import EnhancedPreprocessor, AdvancedFeatureSelector
    from utils import fix_seed, choose_transform, clean_model_params

# モデルファクトリ
try:
    from models.model_factory import ModelFactory
except Exception:
    from model_factory import ModelFactory

# データ拡張・分析
from feature_aware_augmentor import FeatureAwareAugmentor
from data_analyzer import DataAnalyzer

# SHAP フォールバック
try:
    from shap_analysis.complete_shap import CompleteSHAPAnalyzer
    _HAS_SHAP = True
except Exception:
    _HAS_SHAP = False
    class CompleteSHAPAnalyzer:
        def __init__(self, *args, **kwargs): pass
        def analyze(self, *args, **kwargs):
            print("⏭ SHAPは無効化（モジュール未導入）")
            
# === 01_model_builder.py（追記：小さなFail-Fast）===
def assert_config_minimum(Config):
    required = [
        "RESULT_FOLDER","MODEL_FOLDER","DATA_FOLDER","INPUT_FILE",
        "TARGET_COLUMNS","FEATURE_COLUMNS","OUTER_SPLITS","INNER_SPLITS",
        "N_TRIALS","MODELS_TO_USE","RANDOM_STATE","get_n_jobs"
    ]
    missing = [k for k in required if not hasattr(Config, k)]
    if missing:
        raise RuntimeError(f"Config不足: {missing}")
    n_jobs = int(Config.get_n_jobs("optuna"))
    if n_jobs < 1:
        raise RuntimeError("Config.get_n_jobs('optuna') は 1以上が必要です。")

def assert_some_features_exist(df, feature_cols):
    exists = [c for c in feature_cols if c in df.columns]
    if not exists:
        raise RuntimeError("FEATURE_COLUMNS に該当する列が1つもデータに存在しません。")
    return exists


# ---------------- 目的変数変換（Y専用） ----------------
class AdaptiveTargetTransformer:
    def __init__(self):
        self.transform_methods = {}  # {target: 'none'|'log'|'sqrt'}
        self.shifts = {}             # {target: float}
        self.analysis_results = {}   # 記録

    @staticmethod
    def _shift_for_log(y):
        m = np.min(y)
        return abs(m) + 1.0 if m <= 0 else 0.0

    @staticmethod
    def _shift_for_sqrt(y):
        m = np.min(y)
        return abs(m) if m < 0 else 0.0

    def analyze_and_fit(self, y, target_name):
        # --- SciPyが無い環境でも落ちないフォールバック ---
        try:
            from scipy import stats
            _has_scipy = True
        except Exception:
            _has_scipy = False
    
        y = np.asarray(y).astype(float)
    
        # 分布診断（ShapiroはSciPyが必要）
        if _has_scipy and len(y) >= 3:
            try:
                _, p = stats.shapiro(y[:min(5000, len(y))])
            except Exception:
                p = 0.0
        else:
            p = np.nan  # SciPy無のときはNaN（解析用途で明確化）
    
        # 歪度・尖度（SciPy無なら簡易代替 or NaN）
        if _has_scipy and len(y) > 1:
            skew = float(stats.skew(y))
            kurt = float(stats.kurtosis(y))
        else:
            # 単純な近似またはNaN（ここではNaNにしておくのがシンプル）
            skew = np.nan
            kurt = np.nan
    
        self.analysis_results[target_name] = {
            "shapiro_p_value": (float(p) if p == p else None),  # NaN→None
            "skewness": (float(skew) if skew == skew else None),
            "kurtosis": (float(kurt) if kurt == kurt else None),
            "mean": float(np.mean(y)), "std": float(np.std(y)),
            "min": float(np.min(y)), "max": float(np.max(y))
        }

        # 変換法
        method = Config.TRANSFORM_METHOD if Config.TRANSFORM_METHOD != "auto" \
                 else choose_transform(y, method="auto")
        self.transform_methods[target_name] = method
        # 変換
        shift = 0.0
        if method == "log":
            shift = self._shift_for_log(y)
            y_t = np.log(y + shift) if shift > 0 else np.log(y)
        elif method == "sqrt":
            shift = self._shift_for_sqrt(y)
            y_t = np.sqrt(y + shift) if shift > 0 else np.sqrt(y)
        else:
            y_t = y.copy()
        self.shifts[target_name] = float(shift)
        # 変換後検査
        try:
            _, p_after = stats.shapiro(y_t[:min(5000, len(y_t))])
        except Exception:
            p_after = 0.0
        self.analysis_results[target_name]["shapiro_p_value_after"] = float(p_after)
        return y_t

    def transform(self, y, target_name):
        if target_name not in self.transform_methods:
            raise ValueError(f"Target '{target_name}' not fitted")
        method = self.transform_methods[target_name]
        shift = self.shifts.get(target_name, 0.0)
        y = np.asarray(y).astype(float)
        if method == "log":
            return np.log(y + shift) if shift > 0 else np.log(y)
        elif method == "sqrt":
            return np.sqrt(y + shift) if shift > 0 else np.sqrt(y)
        else:
            return y

    def inverse_transform(self, y_t, target_name):
        if target_name not in self.transform_methods:
            raise ValueError(f"Target '{target_name}' not fitted")
        method = self.transform_methods[target_name]
        shift = self.shifts.get(target_name, 0.0)
        y_t = np.asarray(y_t).astype(float)
        if method == "log":
            y = np.exp(y_t)
            return y - shift if shift > 0 else y
        elif method == "sqrt":
            y = np.square(y_t)
            return y - shift if shift > 0 else y
        else:
            return y_t

    def save_metadata(self, path):
        meta = {
            "transform_methods": self.transform_methods,
            "shifts": self.shifts,
            "analysis_results": self.analysis_results,
            "schema_version": "2.1",
        }
        joblib.dump(meta, path)
        print(f"✅ 目的変数変換メタデータ保存: {path}")

# ---------------- 前処理パイプライン（Xは一回だけスケール） ----------------
class CompletePipeline:
    def __init__(self, config, best_params):
        self.config = config
        self.best_params = best_params
        self.preprocessor = None
        self.selector = None
        self.is_fitted = False

    def fit(self, X, y):
        # 外側スケール撤去。dtype整備のみ
        X32 = X.astype("float32", copy=False)
        self.preprocessor = EnhancedPreprocessor(
            use_interactions=self.best_params.get("use_interactions", False),
            use_polynomial=self.best_params.get("use_polynomial", False)
        )
        X_prep = self.preprocessor.fit_transform(X32, pd.Series(y))

        self.selector = AdvancedFeatureSelector(
            top_k=self.best_params.get("top_k", Config.DEFAULT_TOP_K),
            corr_threshold=self.best_params.get("corr_threshold", Config.DEFAULT_CORR_THRESHOLD),
            use_correlation_removal=Config.USE_CORRELATION_REMOVAL,
            mandatory_features=Config.MANDATORY_FEATURES,
            use_mandatory_features=Config.USE_MANDATORY_FEATURES,
            feature_names=X.columns.tolist() if hasattr(X, "columns") else None
        )
        X_sel = self.selector.fit_transform(X_prep, y)
        self.is_fitted = True
        return X_sel

    def transform(self, X):
        if not self.is_fitted:
            raise ValueError("Pipeline not fitted yet")
        X32 = X.astype("float32", copy=False)
        X_prep = self.preprocessor.transform(X32)
        X_sel = self.selector.transform(X_prep)
        return X_sel

    def save(self, path):
        from sklearn.pipeline import Pipeline
        # Pipeline(pre→sel) を構築
        sk_pipe = Pipeline([
            ("pre", self.preprocessor),
            ("sel", self.selector),
        ])
        payload = {
            "sk_pipeline": sk_pipe,                 # 方式B: 追加保存
            "preprocessor": self.preprocessor,      # 旧来の dict も温存（後方互換）
            "selector": self.selector,
            "best_params": self.best_params,
            "is_fitted": self.is_fitted,
            "schema_version": "2.1",                # 方式E: 薄いメタ
            "pipeline_format": "dict+sk_pipeline"   # 方式E
        }
        joblib.dump(payload, path)
        print(f"✅ 前処理パイプライン保存: {path} (format=dict+sk_pipeline, schema=2.1)")


# ---------------- 再現性管理 ----------------
class ReproducibilityManager:
    def __init__(self, seed=42):
        self.seed = seed

    def set_all_seeds(self):
        import random, os
        random.seed(self.seed)
        np.random.seed(self.seed)
        os.environ["PYTHONHASHSEED"] = str(self.seed)
        try:
            import torch
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            print("✅ PyTorch乱数シード設定")
        except ImportError:
            pass
        try:
            import tensorflow as tf
            tf.random.set_seed(self.seed)
            print("✅ TensorFlow乱数シード設定")
        except ImportError:
            pass
        print(f"✅ すべての乱数シード設定: {self.seed}")

    def save_environment_info(self, path):
        import platform, sys as _sys
        info = {
            "python_version": _sys.version,
            "platform": platform.platform(),
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "random_seed": self.seed,
            "timestamp": datetime.now().isoformat()
        }
        try:
            import sklearn
            info["sklearn_version"] = sklearn.__version__
        except Exception:
            pass
        with open(path, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2, ensure_ascii=False)
        print(f"✅ 実行環境情報保存: {path}")

# ---------------- DCV 本体 ----------------
class AdvancedDCV:
    def __init__(self, config):
        self.config = config
        self.augmentor = FeatureAwareAugmentor(config)
        self.shap_analyzer = CompleteSHAPAnalyzer(config)
        self.target_transformer = AdaptiveTargetTransformer()
        self.results = {}

    def _evaluate_params(self, X, y, groups, model_instance, params):
        # 内側CV：外側スケールなし。前処理に一本化
        if groups is None:
            groups = np.arange(len(y))
        groups = np.asarray(groups)
        cv = GroupKFold(n_splits=self.config.INNER_SPLITS)
        scores = []

        # 学習ループ
        for inner_fold, (tr_idx, va_idx) in enumerate(cv.split(X, y, groups)):
            X_tr, X_va = X.iloc[tr_idx].copy(), X.iloc[va_idx].copy()
            y_tr, y_va = y[tr_idx], y[va_idx]

            # Augment は学習折のみ
            X_tr_aug, y_tr_aug, _ = self.augmentor.augment(X_tr, y_tr)

            # 前処理 → 特徴選択
            preprocessor = EnhancedPreprocessor(
                use_interactions=params.get("use_interactions", False),
                use_polynomial=params.get("use_polynomial", False)
            )
            X_tr_prep = preprocessor.fit_transform(X_tr_aug.astype("float32", copy=False), pd.Series(y_tr_aug))
            X_va_prep = preprocessor.transform(X_va.astype("float32", copy=False))

            selector = AdvancedFeatureSelector(
                top_k=params.get("top_k", Config.DEFAULT_TOP_K),
                corr_threshold=params.get("corr_threshold", Config.DEFAULT_CORR_THRESHOLD),
                use_correlation_removal=Config.USE_CORRELATION_REMOVAL,
                mandatory_features=Config.MANDATORY_FEATURES,
                use_mandatory_features=Config.USE_MANDATORY_FEATURES,
                feature_names=X.columns.tolist() if hasattr(X, "columns") else None
            )
            X_tr_sel = selector.fit_transform(X_tr_prep, y_tr_aug)
            X_va_sel = selector.transform(X_va_prep)

            # モデル
            model_params = {k: v for k, v in params.items()
                            if k not in ["top_k", "corr_threshold", "use_interactions", "use_polynomial"]}
            model_params["random_state"] = self.config.RANDOM_STATE
            model_params = clean_model_params(model_params, model_instance.get_name() if hasattr(model_instance, "get_name") else "model")

            model = model_instance.build(**model_params)
            model.fit(X_tr_sel, y_tr_aug)
            y_hat = model.predict(X_va_sel)

            scores.append(mean_absolute_error(y_va, y_hat))

            # クリーニング
            del X_tr, X_va, X_tr_aug, X_tr_prep, X_va_prep, X_tr_sel, X_va_sel
            del preprocessor, selector, model, y_hat
            gc.collect()

        return scores  # ← fold ループの外で返す

    def optimize_model(self, X_train, y_train, groups_train):
        print("\n" + "="*60)
        print("モデル最適化開始（完全リーク対策）")
        print("="*60)

        best_score = float('inf')
        best_model_name = None
        best_params = None
        best_model_instance = None

        available_models = ModelFactory.get_available_models()
        models_to_use = {k: v for k, v in available_models.items() if k in self.config.MODELS_TO_USE}
        if not models_to_use:
            print("⚠ MODELS_TO_USE が空 → 全モデルを対象")
            models_to_use = available_models

        for model_name in models_to_use.keys():
            print(f"\n🔍 {model_name} 最適化中...")
            model_instance = ModelFactory.create_model(model_name)

            def objective(trial):
                try:
                    pp = {
                        "top_k": trial.suggest_int("top_k", 5, min(50, X_train.shape[1])),
                        "corr_threshold": trial.suggest_float("corr_threshold", 0.85, 0.99),
                        "use_interactions": trial.suggest_categorical("use_interactions", [True, False]),
                        "use_polynomial": trial.suggest_categorical("use_polynomial", [True, False]),
                    }
                    mp = model_instance.suggest_hyperparameters(trial)
                    params = {**pp, **mp}
                    scores = self._evaluate_params(X_train, y_train, groups_train, model_instance, params)
                    m = float(np.mean(scores))
                    s = float(np.std(scores))
                    if s > m * 0.5:
                        return float('inf')
                    return m
                except Exception as e:
                    print(f"  Trial失敗: {e}")
                    return float('inf')

            study = optuna.create_study(
                direction="minimize",
                sampler=optuna.samplers.TPESampler(seed=self.config.RANDOM_STATE)
            )
            study.optimize(objective,
                           n_trials=self.config.N_TRIALS,
                           n_jobs=self.config.get_n_jobs('optuna'),
                           show_progress_bar=Config.SHOW_OPTUNA_PROGRESS, catch=(Exception,))

            if study.best_value < float('inf'):
                y_std = np.std(y_train)
                if y_std > 0 and study.best_value > y_std * 0.5:
                    print(f"  {model_name} 性能基準未達: MAE={study.best_value:.4f} > {y_std*0.5:.4f}")
                else:
                    print(f"  {model_name} ベストスコア: {study.best_value:.4f}")
                    if study.best_value < best_score:
                        best_score = study.best_value
                        best_model_name = model_name
                        best_params = study.best_params
                        best_model_instance = model_instance
            else:
                print(f"  {model_name} 最適化失敗")

        if best_model_name is None:
            print("\n⚠ すべて基準未達 → フォールバック採用")
            for name in self.config.FALLBACK_MODEL_ORDER:
                if name in models_to_use:
                    best_model_name = name
                    best_model_instance = ModelFactory.create_model(name)
                    best_params = self.config.FALLBACK_DEFAULT_PARAMS.copy()
                    best_params["top_k"] = min(Config.DEFAULT_TOP_K, X_train.shape[1])
                    print(f"  → {best_model_name} を使用")
                    break

        print(f"\n🏆 内側CVベストモデル: {best_model_name} (平均MAE: {best_score:.4f})")
        return best_model_name, best_params, best_model_instance

    def run_dcv(self, X, y, target_name):
        print(f"\n{'='*60}\nDouble Cross-Validation: {target_name}\nPPM Levels: {self.config.PPM_LEVELS}\n{'='*60}")

        base_groups = getattr(self, "groups", None)
        if base_groups is None:
            base_groups = np.arange(len(y))
        base_groups = np.asarray(base_groups)

        outer_cv = GroupKFold(n_splits=self.config.OUTER_SPLITS)
        fold_results, all_predictions, all_true, oof_indices = [], [], [], []

        for fold, (tr_idx, te_idx) in enumerate(outer_cv.split(X, y, base_groups)):
            print(f"\n--- Outer Fold {fold+1}/{self.config.OUTER_SPLITS} ---")
            X_tr_base = X.iloc[tr_idx].copy()
            y_tr_base = y[tr_idx]
            g_tr_base = base_groups[tr_idx]
            X_te_base = X.iloc[te_idx].copy()
            y_te_base = y[te_idx]

            # Y変換決定/適用
            if fold == 0:
                y_tr_trans = self.target_transformer.analyze_and_fit(y_tr_base, target_name)
            else:
                y_tr_trans = self.target_transformer.transform(y_tr_base, target_name)
            y_te_trans = self.target_transformer.transform(y_te_base, target_name)
            transform_method = self.target_transformer.transform_methods[target_name]

            # 内側最適化
            best_name, best_params, best_inst = self.optimize_model(X_tr_base, y_tr_trans, g_tr_base)

            # 外側最終学習（学習データのみ拡張）
            X_tr_aug, y_tr_aug, _ = self.augmentor.augment(X_tr_base, y_tr_trans)
            X_tr_aug = X_tr_aug.astype("float32", copy=False)

            pipeline = CompletePipeline(self.config, best_params)
            X_tr_sel = pipeline.fit(X_tr_aug, y_tr_aug)
            X_te_sel = pipeline.transform(X_te_base)

            model_params = {k: v for k, v in best_params.items()
                            if k not in ["top_k", "corr_threshold", "use_interactions", "use_polynomial"]}
            model_params["random_state"] = self.config.RANDOM_STATE
            model_params = clean_model_params(model_params, best_name)

            model = best_inst.build(**model_params)
            model.fit(X_tr_sel, y_tr_aug)

            y_pred_t = model.predict(X_te_sel)
            y_pred = self.target_transformer.inverse_transform(y_pred_t, target_name)
            y_te_orig = self.target_transformer.inverse_transform(y_te_trans, target_name)

            mae = mean_absolute_error(y_te_orig, y_pred)
            rmse = np.sqrt(mean_squared_error(y_te_orig, y_pred))
            r2 = r2_score(y_te_orig, y_pred)
            print(f"  Model: {best_name}")
            print(f"  MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")

            fold_results.append({
                "fold": fold, "model_name": best_name, "params": best_params,
                "mae": mae, "rmse": rmse, "r2": r2, "transform_method": transform_method
            })
            all_predictions.extend(y_pred)
            all_true.extend(y_te_orig)
            oof_indices.extend(te_idx.tolist())

            # fold終端クリーンアップ（存在変数のみ）
            del X_tr_base, y_tr_base, g_tr_base, X_te_base, y_te_base
            del y_tr_trans, y_te_trans, y_pred_t, y_pred, y_te_orig
            del X_tr_aug, y_tr_aug, X_tr_sel, X_te_sel, pipeline, model
            gc.collect()

        # OOF 指標
        if len(all_true) > 0:
            all_true = np.array(all_true)
            all_predictions = np.array(all_predictions)
            cv_mae = mean_absolute_error(all_true, all_predictions)
            cv_rmse = np.sqrt(mean_squared_error(all_true, all_predictions))
            cv_r2 = r2_score(all_true, all_predictions)
        else:
            cv_mae = cv_rmse = cv_r2 = np.nan

        # 代表ハイパラ（覗き見排除）
        if len(fold_results) == 0:
            final_model_name = self.config.FALLBACK_FINAL_MODEL
            final_params = self.config.FALLBACK_DEFAULT_PARAMS.copy()
        else:
            names = [fr["model_name"] for fr in fold_results]
            final_model_name = Counter(names).most_common(1)[0][0]
            cands = [fr["params"] for fr in fold_results if fr["model_name"] == final_model_name]
            key_vals = defaultdict(list)
            for p in cands:
                for k, v in p.items():
                    key_vals[k].append(v)
            INT_KEYS = {
                "top_k", "max_iter", "n_neighbors", "n_estimators", "max_depth",
                "min_samples_split", "min_samples_leaf", "num_leaves", "bagging_freq",
                "min_child_samples", "subsample_for_bin", "min_data_in_bin",
                "max_bin", "max_bins", "n_components", "n_clusters"
            }
            BOOL_KEYS = {"use_interactions", "use_polynomial"}
            final_params = {}
            for k, vs in key_vals.items():
                if k in BOOL_KEYS or isinstance(vs[0], bool):
                    final_params[k] = Counter(vs).most_common(1)[0][0]
                elif k in INT_KEYS:
                    final_params[k] = int(round(np.median([float(v) for v in vs])))
                elif all(isinstance(v, (int, float, np.integer, np.floating)) for v in vs):
                    final_params[k] = float(np.median(vs))
                else:
                    final_params[k] = Counter(vs).most_common(1)[0][0]
            final_params["top_k"] = int(final_params.get("top_k", 20))
            final_params["corr_threshold"] = float(final_params.get("corr_threshold", 0.95))
            final_params["use_interactions"] = bool(final_params.get("use_interactions", False))
            final_params["use_polynomial"] = bool(final_params.get("use_polynomial", False))

        # 最終学習（全データ）
        print(f"\n{'='*60}\n最終モデル訓練（全データ、評価はOOFのみ）\n{'='*60}")
        final_inst = ModelFactory.create_model(final_model_name)

        y_all_t = self.target_transformer.transform(y, target_name)
        X_all_aug, y_all_aug, _ = self.augmentor.augment(X, y_all_t)
        final_pipeline = CompletePipeline(self.config, final_params)
        X_all_sel = final_pipeline.fit(X_all_aug.astype("float32", copy=False), y_all_aug)

        # 保存（パイプライン）
        os.makedirs(self.config.MODEL_FOLDER, exist_ok=True)
        pipe_path = os.path.join(self.config.MODEL_FOLDER, f"pipeline_{target_name}.pkl")
        final_pipeline.save(pipe_path)

        feature_names = self._get_feature_names(X, final_pipeline.preprocessor, final_pipeline.selector)

        fm_params = {k: v for k, v in final_params.items()
                     if k not in ["top_k", "corr_threshold", "use_interactions", "use_polynomial"]}
        fm_params["random_state"] = self.config.RANDOM_STATE
        fm_params = clean_model_params(fm_params, final_model_name)

        final_model = final_inst.build(**fm_params)
        final_model.fit(X_all_sel, y_all_aug)

        print(f"最終モデル: {final_model_name}")
        print(f"特徴量数: {X_all_sel.shape[1]}")

        if _HAS_SHAP and self.config.SHAP_MODE != "none":
            shap_folder = os.path.join(self.config.RESULT_FOLDER, f"shap_{target_name}_final")
            self.shap_analyzer.analyze(
                model=final_model, X=X_all_sel, y=y_all_aug,
                feature_names=feature_names, target_name=target_name,
                model_type=final_inst.get_model_type(), output_folder=shap_folder
            )

        pack = {
            "fold_results": fold_results,
            "predictions": np.array(all_predictions),
            "true_values": np.array(all_true),
            "oof_indices": np.array(oof_indices, dtype=int),
            "final_model": final_model,
            "final_model_name": final_model_name,
            "final_model_instance": final_inst,
            "preprocessor": final_pipeline.preprocessor,
            "selector": final_pipeline.selector,
            "transform_method": self.target_transformer.transform_methods[target_name],
            "feature_names": feature_names,
            "best_params": final_params,
            "cv_mae": float(cv_mae) if cv_mae == cv_mae else None,
            "cv_rmse": float(cv_rmse) if cv_rmse == cv_rmse else None,
            "cv_r2": float(cv_r2) if cv_r2 == cv_r2 else None,
            "schema_version": "2.1",               # 方式E: バンドル側にも簡易スキーマ
            "pipeline_format": "dict+sk_pipeline" # 方式E: 将来の読み分け用ヒント
        }
        self.results[target_name] = pack
        return pack

    def _get_feature_names(self, X0, preprocessor, selector):
        orig = X0.columns.tolist()
        if hasattr(preprocessor, "get_feature_names"):
            proc = preprocessor.get_feature_names()
        else:
            proc = orig
        if hasattr(selector, "selected_features_") and selector.selected_features_ is not None:
            idx = selector.selected_features_
            if len(idx) <= len(proc):
                names = [proc[i] if i < len(proc) else f"Feature_{i}" for i in idx]
            else:
                names = [f"Feature_{i}" for i in range(len(idx))]
        else:
            names = proc
        return names

# ---------------- データ分析（任意） ----------------
def perform_data_analysis(df):
    print("\n" + "="*60)
    print("📊 データ分析開始")
    print("="*60)
    out = os.path.join(Config.RESULT_FOLDER, "data_analysis")
    analyzer = DataAnalyzer(output_folder=out)
    res = analyzer.analyze_dataframe(
        df=df,
        target_columns=Config.TARGET_COLUMNS,
        feature_columns=Config.FEATURE_COLUMNS,
        show_plots=False, save_plots=True
    )
    print("\n" + "="*60)
    print("✅ データ分析完了")
    print(f"結果保存先: {out}")
    print("="*60)
    return res

def _assert_cv_splits(n_samples: int, outer: int, inner: int):
    if n_samples < outer:
        raise ValueError(f"OUTER_SPLITS={outer} はサンプル数 {n_samples} を超えています。")
    if n_samples < inner:
        raise ValueError(f"INNER_SPLITS={inner} はサンプル数 {n_samples} を超えています。")

def _downcast_df(df: pd.DataFrame) -> pd.DataFrame:
    fcols = df.select_dtypes(include=["float64"]).columns
    icols = df.select_dtypes(include=["int64"]).columns
    # ES: DEBUG: inspeccionar tipos antes de downcast
    # EN: DEBUG: inspect dtypes before downcast
    # JA: DEBUG: downcast前に型を確認
    try:
        print("🔍 DEBUG _downcast_df: float64 cols =", list(fcols))
        print("🔍 DEBUG _downcast_df: int64 cols   =", list(icols))
        if len(icols):
            print("🔍 DEBUG _downcast_df: dtypes int64 cols =", df[icols].dtypes.to_dict())
            print("🔍 DEBUG _downcast_df: head int64 cols =", df[icols].head(3).to_dict(orient="list"))
    except Exception as _e:
        # ES: No bloquear ejecución si el debug falla
        # EN: Do not block execution if debug fails
        # JA: デバッグが失敗しても実行を止めない
        print(f"⚠️ デバッグ: _downcast_df に失敗: {_e}")
    if len(fcols):
        df[fcols] = df[fcols].astype("float32")
    if len(icols):
        # ES: pd.to_numeric no acepta DataFrames completos; aplicar columna a columna
        # EN: pd.to_numeric does not accept full DataFrames; apply column-by-column
        # JA: pd.to_numeric はDataFrame全体に使えないため列ごとに適用
        df[icols] = df[icols].apply(pd.to_numeric, downcast="integer")
    return df

# ---------------- メイン ----------------
def main():
    # 再現性
    repro = ReproducibilityManager(seed=Config.RANDOM_STATE)
    repro.set_all_seeds()

    # 【追加】Fail-Fast：Configの最低限チェック
    assert_config_minimum(Config)

    os.makedirs(Config.RESULT_FOLDER, exist_ok=True)
    repro.save_environment_info(os.path.join(Config.RESULT_FOLDER, "environment_info.json"))

    # 設定検証
    try:
        Config.validate()
        print("✅ 設定検証成功")
        finfo = Config.get_feature_info()
        print(f"特徴量構成: 連続{len(finfo['continuous'])}個, 離散{len(finfo['discrete'])}個, "
              f"バイナリ{len(finfo['binary'])}個, 整数{len(finfo['integer'])}個")
    except ValueError as e:
        print(f"❌ 設定エラー: {e}")
        sys.exit(1)

    # モデル可用性
    avail = ModelFactory.get_available_models()
    use = {k: v for k, v in avail.items() if k in Config.MODELS_TO_USE}
    print(f"利用可能なモデル: {list(avail.keys())}")
    print(f"使用するモデル: {list(use.keys())}")

    # 読み込み
    try:
        path = os.path.join(Config.DATA_FOLDER, Config.INPUT_FILE)
        df = pd.read_excel(path)
        df = _downcast_df(df)
        # 読み込み後
        print(f"\n✅ データ読み込み成功: {df.shape}")
        _assert_cv_splits(len(df), Config.OUTER_SPLITS, Config.INNER_SPLITS)

    except FileNotFoundError:
        print(f"⚠ ファイル未検出: {os.path.join(Config.DATA_FOLDER, Config.INPUT_FILE)}")
        print("デモデータを生成します...")
        np.random.seed(42)
        n = 500
        df = pd.DataFrame()
        for c in Config.CONTINUOUS_FEATURES: df[c] = np.random.randn(n)*10 + 50
        for c in Config.DISCRETE_FEATURES:   df[c] = np.random.choice([0,1,2], n)
        for c in Config.BINARY_FEATURES:     df[c] = np.random.choice([0,1], n)
        for c in Config.INTEGER_FEATURES:    df[c] = np.random.randint(1,10, n)
        for t in Config.TARGET_COLUMNS:      df[t] = np.random.randn(n)*5 + 30

    # 【追加】Fail-Fast：特徴量が1つも無い事故を入口で阻止
    _ = assert_some_features_exist(df, Config.FEATURE_COLUMNS)

    if Config.RUN_DATA_ANALYSIS:
        perform_data_analysis(df)
    else:
        print("\n⏭ データ分析スキップ（Config.RUN_DATA_ANALYSIS=False）")

    dcv = AdvancedDCV(Config)
    all_results = {}

    for target in Config.TARGET_COLUMNS:
        if target not in df.columns:
            print(f"⚠ Target '{target}' not found")
            continue
        df_task = df.dropna(subset=[target])
        feat_cols = [c for c in Config.FEATURE_COLUMNS if c in df_task.columns]
        X = df_task[feat_cols]
        y = df_task[target].values

        print(f"\n処理中: {target}")
        print(f"  データサイズ: {X.shape}")

        result = dcv.run_dcv(X, y, target)
        all_results[target] = result

        print(f"\n{'='*60}\n結果サマリー（外側OOF）: {target}\n{'='*60}")
        print(f"CV MAE:  {result['cv_mae']:.4f}" if result['cv_mae'] is not None else "CV MAE:  n/a")
        print(f"CV RMSE: {result['cv_rmse']:.4f}" if result['cv_rmse'] is not None else "CV RMSE: n/a")
        print(f"CV R2:   {result['cv_r2']:.4f}" if result['cv_r2'] is not None else "CV R2:   n/a")
        print(f"最終モデル: {result['final_model_name']}")

        # 推論用バンドル（scalerは持たない）
        bundle = {
            "model_name": result["final_model_name"],
            "final_model": result["final_model"],
            "preprocessor": result["preprocessor"],
            "selector": result["selector"],
            "transform_method": result["transform_method"],
            "transform_shift": dcv.target_transformer.shifts.get(target, 0.0),
            "transform_analysis": dcv.target_transformer.analysis_results.get(target, {}),
            "feature_names": result["feature_names"],
            "feature_columns": feat_cols,
            "mandatory_features": Config.MANDATORY_FEATURES,
            "continuous_features": Config.CONTINUOUS_FEATURES,
            "discrete_features": Config.DISCRETE_FEATURES,
            "binary_features": Config.BINARY_FEATURES,
            "integer_features": Config.INTEGER_FEATURES,
            "best_params": result["best_params"],
            "cv_mae": result["cv_mae"],
            "cv_rmse": result["cv_rmse"],
            "cv_r2": result["cv_r2"],
            "created_at": datetime.now().isoformat(),
            "target_name": target,
            "config_version": "2.0",
            "augmentation_ratio": Config.AUGMENT_RATIO,
            "ppm_levels": Config.PPM_LEVELS,
            "random_state": Config.RANDOM_STATE,
            # （任意だが推奨）学習側のスキーマラベルを合わせて入れておく
            "schema_version": Config.PIPELINE_SCHEMA_VERSION,
            "pipeline_format": Config.PIPELINE_FORMAT_LABEL,
        }
        os.makedirs(Config.MODEL_FOLDER, exist_ok=True)
        bundle_path = os.path.join(Config.MODEL_FOLDER, f"{Config.FINAL_MODEL_PREFIX}_{target}.pkl")
        joblib.dump(bundle, bundle_path)
        print(f"✅ 推論用バンドル保存: {bundle_path}")

        # 変換メタ
        tmeta_path = os.path.join(Config.MODEL_FOLDER, f"transform_metadata_{target}.pkl")
        dcv.target_transformer.save_metadata(tmeta_path)

    save_path = os.path.join(Config.RESULT_FOLDER, "dcv_results.pkl")
    joblib.dump(all_results, save_path)
    print(f"\n✅ 結果保存: {save_path}")

    return all_results


# ---------------- 実行 & OOF 可視化 ----------------
if __name__ == "__main__":
    results = main()
    if results:
        for target, result in results.items():
            y_true = result.get("true_values")
            y_pred = result.get("predictions")
            if y_true is None or y_pred is None or len(y_pred) == 0:
                continue
            y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
            residuals = y_true - y_pred
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            r2 = r2_score(y_true, y_pred)
            sigma = residuals.std(ddof=1) if len(residuals) > 1 else 0.0
            thr = 2.0 * sigma
            out_mask = (np.abs(residuals) >= thr) if sigma > 0 else np.zeros_like(residuals, dtype=bool)
            oof_idx = result.get("oof_indices", np.arange(len(y_true)))

            plt.figure(figsize=(12,5))
            # True vs Pred
            plt.subplot(1,2,1)
            plt.scatter(y_true, y_pred, alpha=0.5)
            lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
            plt.plot([lo,hi],[lo,hi], "r--")
            plt.xlabel("True Values"); plt.ylabel("Predictions")
            plt.title(f"{target}: Predictions vs True"); plt.grid(True, alpha=0.3)
            ax1 = plt.gca()
            tmethod = result.get("transform_method", "none")
            tinfo = f"Y変換: {tmethod}" if tmethod != "none" else "Y変換: なし"
            txt = (f"Model: {result['final_model_name']}\n{tinfo}\n"
                   f"MAE = {mae:.3f}\nRMSE = {rmse:.3f}\nR² = {r2:.3f}\n"
                   f"σ = {sigma:.3f}（外れ値: |res|≥2σ → {int(out_mask.sum())}点）")
            ax1.text(0.02, 0.98, txt, transform=ax1.transAxes, va='top', ha='left',
                     bbox=dict(boxstyle='round', fc='white', alpha=0.85))
            for i in np.where(out_mask)[0]:
                ax1.annotate(str(oof_idx[i]), (y_true[i], y_pred[i]),
                             textcoords='offset points', xytext=(3,3), fontsize=9)

            # Residuals
            plt.subplot(1,2,2)
            plt.scatter(y_pred, residuals, alpha=0.5)
            plt.axhline(0, color='r', linestyle='--')
            if sigma > 0:
                plt.axhline(+thr, color='gray', linestyle='--')
                plt.axhline(-thr, color='gray', linestyle='--')
                plt.scatter(y_pred[out_mask], residuals[out_mask],
                            facecolors='none', edgecolors='r', s=64, linewidths=1.5)
            plt.xlabel("Predictions"); plt.ylabel("Residuals")
            plt.title("Residual Plot"); plt.grid(True, alpha=0.3)

            out_path = os.path.join(Config.RESULT_FOLDER, f"{target}_results.png")
            plt.tight_layout(); plt.savefig(out_path)
            print(f"🖼 可視化保存: {out_path}")

    print("\n" + "="*60)
    print("パイプライン完了（安定化・バグ修正版：Xは前処理内一回スケール）")
    print("="*60)

    # __pycache__ の整理（移動）
    import shutil, glob
    def move_pycache_to_temp():
        temp = Path("99_Temp"); temp.mkdir(exist_ok=True)
        for p in glob.glob("**/__pycache__", recursive=True):
            pth = Path(p)
            if pth.exists() and pth.is_dir():
                dest = temp / f"{pth.parent.name}__pycache__"
                if dest.exists(): shutil.rmtree(dest)
                shutil.move(str(pth), str(dest))
                print(f"✅ {pth} → {dest}")
    move_pycache_to_temp()
