"""
多目的最適化版トレーナー
- NP_ALPHAを探索パラメータとして組み込み
- FP率最小化とカバレッジ最大化の多目的最適化
"""
import os
import json
import warnings
import optuna
from optuna.samplers import NSGAIISampler
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from collections import Counter
import joblib

from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_auc_score

from config_cls import ConfigCLS
from schema_detect import detect_schema
from feature_noise import add_noise_columns
from feature_select import select_features_by_importance
from models_cls import ModelFactoryCLS
from calibration import select_and_fit_calibrator
from thresholds import np_tau_pos_from_neg_scores, adjust_tau_pos_by_ci
from ood_mahalanobis import MahalanobisGate

warnings.filterwarnings("ignore")


def _prepare_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, list]:
    """データ準備（既存のコードと同じ）"""
    schema = detect_schema(
        df,
        target_column=ConfigCLS.TARGET_COLUMN,
        positive_labels=ConfigCLS.POSITIVE_LABELS,
        negative_labels=ConfigCLS.NEGATIVE_LABELS,
        id_candidates=ConfigCLS.ID_COLUMNS_CANDIDATES,
        date_candidates=ConfigCLS.DATE_COLUMNS_CANDIDATES,
        group_column=ConfigCLS.GROUP_COLUMN,
    )
    feat_cols = schema["feature_columns"]
    hard_exclude = set(ConfigCLS.EXCLUDE_COLUMNS) | {ConfigCLS.TARGET_COLUMN}
    feat_cols = [c for c in feat_cols if c not in hard_exclude]
    
    X = df[feat_cols].copy()
    y = schema["y"]
    g = df[schema["group_column"]].values if schema["group_column"] else np.arange(len(df))
    return X, y, g, feat_cols


def calculate_coverage_and_fp_rate(p_cal, y_true, tau_pos, tau_neg):
    """カバレッジとFP率を計算"""
    pred = np.full_like(y_true, fill_value=-1)
    pred[p_cal >= tau_pos] = 1
    pred[p_cal <= tau_neg] = 0
    
    # カバレッジ（Gray以外の割合）
    coverage = 1.0 - np.mean(pred == -1)
    
    # FP率（実際のNegativeをPositiveと判定する率）
    neg_mask = (y_true == 0)
    if neg_mask.sum() > 0:
        fp_rate = np.mean((pred == 1) & neg_mask) / neg_mask.mean()
    else:
        fp_rate = 0.0
    
    return coverage, fp_rate


def multi_objective_optimization(X_train, y_train, groups_train, n_trials=50):
    """
    多目的最適化でNP_ALPHAを含むハイパーパラメータを探索
    目的：
    1. FP率最小化（NP_ALPHA制約の遵守）
    2. カバレッジ最大化
    3. AUC最大化
    """
    
    def objective(trial):
        # NP_ALPHAを探索対象に含める
        np_alpha = trial.suggest_float("np_alpha", 0.001, 0.1, log=True)
        
        # モデル関連のハイパーパラメータ
        model_name = trial.suggest_categorical("model_name", ConfigCLS.MODELS_TO_USE)
        top_k = trial.suggest_int("top_k", *ConfigCLS.SELECT_TOP_K_RANGE)
        corr_threshold = trial.suggest_float("corr_threshold", *ConfigCLS.CORR_THRESHOLD_RANGE)
        
        # モデル固有のハイパーパラメータ
        mconf = ConfigCLS.MODEL_CONFIGS.get(model_name, {})
        params = ModelFactoryCLS.suggest_hyperparams(model_name, trial, mconf)
        
        # Inner CV での評価
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=ConfigCLS.RANDOM_STATE)
        
        fp_rates = []
        coverages = []
        aucs = []
        
        for tr_idx, va_idx in cv.split(X_train, y_train):
            X_tr = X_train.iloc[tr_idx].copy()
            y_tr = y_train[tr_idx]
            X_va = X_train.iloc[va_idx].copy()
            y_va = y_train[va_idx]
            
            # 前処理
            X_tr_num = X_tr.apply(pd.to_numeric, errors="coerce").fillna(0)
            X_va_num = X_va.apply(pd.to_numeric, errors="coerce").fillna(0)
            
            scaler = RobustScaler()
            X_tr_scaled = pd.DataFrame(
                scaler.fit_transform(X_tr_num), 
                columns=X_tr_num.columns, 
                index=X_tr_num.index
            )
            X_va_scaled = pd.DataFrame(
                scaler.transform(X_va_num), 
                columns=X_va_num.columns, 
                index=X_va_num.index
            )
            
            # モデル構築と特徴選択
            model = ModelFactoryCLS.build(model_name, params)
            X_tr_sel = select_features_by_importance(
                model, X_tr_scaled, y_tr,
                top_k=top_k,
                corr_threshold=corr_threshold,
                forced_keep=[c for c in ConfigCLS.MUST_KEEP_FEATURES if c in X_tr_scaled.columns]
            )
            sel_cols = X_tr_sel.columns.tolist()
            
            # 学習と予測
            model.fit(X_tr_sel, y_tr)
            proba_va = model.predict_proba(X_va_scaled[sel_cols])[:, 1]
            
            # τ+計算（動的なnp_alpha使用）
            neg_scores = proba_va[y_va == 0]
            if len(neg_scores) > 0:
                tau_pos_cv = np.quantile(neg_scores, 1 - np_alpha)
            else:
                tau_pos_cv = 0.5
            
            # τ-探索（簡易版）
            tau_neg_cv = 0.3  # 固定値または簡易探索
            
            # 評価指標計算
            coverage, fp_rate = calculate_coverage_and_fp_rate(proba_va, y_va, tau_pos_cv, tau_neg_cv)
            auc = roc_auc_score(y_va, proba_va)
            
            fp_rates.append(fp_rate)
            coverages.append(coverage)
            aucs.append(auc)
        
        # 3つの目的を返す
        return [
            -np.mean(fp_rates),  # FP率最小化（負値で返す）
            np.mean(coverages),   # カバレッジ最大化
            np.mean(aucs)        # AUC最大化
        ]
    
    # 多目的最適化の実行
    study = optuna.create_study(
        study_name="multi_objective_cls",
        directions=["maximize", "maximize", "maximize"],  # 全て最大化
        sampler=NSGAIISampler(seed=ConfigCLS.RANDOM_STATE)
    )
    
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    return study


def select_best_trial(study, fp_weight=0.3, coverage_weight=0.5, auc_weight=0.2):
    """
    Pareto最適解から重み付けスコアで最良のトライアルを選択
    """
    best_score = -float('inf')
    best_trial = None
    
    for trial in study.best_trials:
        # 目的値を取得（FP率は負値なので反転）
        fp_score = -trial.values[0]  # 小さいほど良い
        coverage_score = trial.values[1]  # 大きいほど良い
        auc_score = trial.values[2]  # 大きいほど良い
        
        # 正規化（0-1スケール）
        fp_norm = 1.0 - fp_score  # FP率が低いほど高スコア
        
        # 重み付けスコア
        total_score = (
            fp_weight * fp_norm + 
            coverage_weight * coverage_score + 
            auc_weight * auc_score
        )
        
        if total_score > best_score:
            best_score = total_score
            best_trial = trial
    
    return best_trial


def train_and_bundle_multiobjective(input_excel: Optional[str] = None) -> Dict[str, Any]:
    """多目的最適化版の学習"""
    
    # データ準備（既存コードと同じ）
    df = pd.read_excel(input_excel or ConfigCLS.INPUT_FILE)
    os.makedirs(ConfigCLS.RESULT_FOLDER, exist_ok=True)
    
    tcol = ConfigCLS.TARGET_COLUMN
    df[tcol] = df[tcol].replace(r"^\s*$", np.nan, regex=True)
    df = df.dropna(subset=[tcol]).copy()
    
    X, y, groups, feature_cols = _prepare_data(df)
    
    # 多目的最適化の実行
    print("=== 多目的最適化開始 ===")
    study = multi_objective_optimization(X, y, groups, n_trials=ConfigCLS.N_TRIALS_MULTI_OBJECTIVE)
    
    # Pareto最適解の表示
    print(f"\nPareto最適解の数: {len(study.best_trials)}")
    for i, trial in enumerate(study.best_trials[:5]):  # 上位5件表示
        print(f"  解{i+1}: FP率={-trial.values[0]:.3f}, カバレッジ={trial.values[1]:.3f}, AUC={trial.values[2]:.3f}")
        print(f"    NP_ALPHA={trial.params['np_alpha']:.4f}")
    
    # 最良のトライアルを選択
    best_trial = select_best_trial(study, 
                                   fp_weight=ConfigCLS.FP_WEIGHT,
                                   coverage_weight=ConfigCLS.COVERAGE_WEIGHT,
                                   auc_weight=ConfigCLS.AUC_WEIGHT)
    
    print(f"\n選択された最適解:")
    print(f"  NP_ALPHA: {best_trial.params['np_alpha']:.4f}")
    print(f"  モデル: {best_trial.params['model_name']}")
    print(f"  FP率: {-best_trial.values[0]:.3f}")
    print(f"  カバレッジ: {best_trial.values[1]:.3f}")
    print(f"  AUC: {best_trial.values[2]:.3f}")
    
    # 選択されたパラメータで最終モデルを学習
    best_np_alpha = best_trial.params['np_alpha']
    best_model_name = best_trial.params['model_name']
    best_params = {k: v for k, v in best_trial.params.items() 
                   if k not in ['np_alpha', 'model_name', 'top_k', 'corr_threshold']}
    
    # 以降は既存のコードと同様に最終モデルを構築...
    # （省略：既存のtrain_and_bundle関数の後半部分と同じ処理）
    
    # バンドル作成
    bundle = {
        "np_alpha": float(best_np_alpha),
        "model_name": best_model_name,
        "multi_objective_results": {
            "pareto_solutions": [
                {
                    "np_alpha": t.params['np_alpha'],
                    "fp_rate": -t.values[0],
                    "coverage": t.values[1],
                    "auc": t.values[2]
                }
                for t in study.best_trials[:10]
            ],
            "selected_solution": {
                "np_alpha": best_np_alpha,
                "fp_rate": -best_trial.values[0],
                "coverage": best_trial.values[1],
                "auc": best_trial.values[2]
            }
        },
        # ... 既存のバンドル内容
    }
    
    # 結果保存
    out_path = os.path.join(ConfigCLS.RESULT_FOLDER, "final_bundle_multiobjective.pkl")
    joblib.dump(bundle, out_path)
    
    # Pareto前線の可視化用データも保存
    pareto_df = pd.DataFrame([
        {
            "np_alpha": t.params['np_alpha'],
            "fp_rate": -t.values[0],
            "coverage": t.values[1],
            "auc": t.values[2]
        }
        for t in study.best_trials
    ])
    pareto_df.to_csv(os.path.join(ConfigCLS.RESULT_FOLDER, "pareto_front.csv"), index=False)
    
    print(f"\n✅ 多目的最適化完了: {out_path}")
    
    return bundle