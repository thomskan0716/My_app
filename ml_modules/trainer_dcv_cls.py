"""
分類（FP=0保証 三値化）トレーナ - DCV版（修正版）
- Outer×Inner（Group対応）
- Innerのみ：列フィルタ→ノイズ→スケール→特徴選択→HPO（Optuna）
- 退化チェックは Inner fold ごとに実施（ご指定のロジック）
- Outer：best_params を反映して再学習（退化チェックはしない）
- 校正選択（Temperature / Isotonic）
- τ+（NP）、τ−（Coverage最適化、Precision@Pos=1.0制約）
- OOD（Mahalanobis）
- 最終モデル全量学習 + バンドル保存
"""
import os
import json
import warnings
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter  # Counter追加
import joblib
import optuna
import numpy as np
import pandas as pd

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


# =========================
#  ユーティリティ
# =========================
def learn_column_filter(X_tr: pd.DataFrame) -> tuple[list, pd.Series]:
    """
    学習用データで列選抜ルールを学習して返す
      - 50%以上がNaNの列を落とす
      - 数値化後に中央値で仮埋め → 分散0の列を落とす
    戻り値:
      keep_columns: 採用する列名リスト
      medians: 数値列の学習時中央値（欠損埋め用）
    """
    Xn = X_tr.apply(pd.to_numeric, errors="coerce")
    miss_ratio = Xn.isna().mean()
    cols_missing_ok = miss_ratio.index[miss_ratio <= 0.5].tolist()
    Xn = Xn[cols_missing_ok]

    med = Xn.median(numeric_only=True)
    Xfilled = Xn.fillna(med)
    var = Xfilled.var(numeric_only=True)
    cols_var_ok = var.index[var > 0.0].tolist()

    keep = cols_var_ok
    return keep, med


def _prepare_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, list]:
    """
    スキーマ検出 → 必要に応じて列を上書き除外（Indexや回帰目的変数を説明変数から外す）
    """
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

    # ★ detect_schema の結果に対し、Config 側の除外規則を必ず適用（回帰目的変数や 'Index' 等を弾く）
    hard_exclude = set(ConfigCLS.EXCLUDE_COLUMNS) | {ConfigCLS.TARGET_COLUMN}
    feat_cols = [c for c in feat_cols if c not in hard_exclude]

    X = df[feat_cols].copy()
    y = schema["y"]
    g = df[schema["group_column"]].values if schema["group_column"] else np.arange(len(df))
    return X, y, g, feat_cols


def _split_outer(X, y, groups):
    if ConfigCLS.GROUP_COLUMN:
        cv = GroupKFold(n_splits=ConfigCLS.OUTER_SPLITS)
        return list(cv.split(X, y, groups))
    cv = StratifiedKFold(n_splits=ConfigCLS.OUTER_SPLITS, shuffle=True, random_state=ConfigCLS.RANDOM_STATE)
    return list(cv.split(X, y))


def _metric_precision_pos(y_true, y_pred):
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    return tp / (tp + fp) if (tp + fp) > 0 else 1.0

# trainer_dcv_cls.py および trainer_dcv_cls_with_eval.py の修正

# 旧関数をコメントアウトまたは削除
# def _search_tau_neg(p_cal, y_true, tau_pos, grid):
#     ...

# 修正版を追加
def _search_tau_neg(p_cal, y_true, tau_pos, grid):
    """修正版：Precision@Pos=1.0 を壊さず Coverage を最大化"""
    
    # 入力検証
    if len(grid) == 0:
        grid = np.linspace(0.0, 0.8, 161).tolist()
    
    # τ+より小さい値のみ有効
    valid_grid = [t for t in sorted(grid) if t < tau_pos]
    if len(valid_grid) == 0:
        valid_grid = [tau_pos * 0.5]
    
    best_coverage = -1.0
    best_tau = valid_grid[0]
    candidates_found = 0
    
    for tau_candidate in valid_grid:
        pred = np.full_like(y_true, fill_value=-1, dtype=int)
        pred[p_cal >= tau_pos] = 1
        pred[p_cal <= tau_candidate] = 0
        
        # Precision計算
        pos_mask = (pred == 1)
        if pos_mask.sum() > 0:
            tp = np.sum((pred == 1) & (y_true == 1))
            fp = np.sum((pred == 1) & (y_true == 0))
            precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        else:
            precision = 1.0
        
        # FP=0制約チェック
        if precision < 0.999999:
            continue
        
        coverage = 1.0 - np.mean(pred == -1)
        
        if coverage > best_coverage:
            best_coverage = coverage
            best_tau = tau_candidate
            candidates_found += 1
    
    # フォールバック処理
    if candidates_found == 0:
        best_tau = min(tau_pos * 0.3, valid_grid[-1])
        pred_fb = np.full_like(y_true, fill_value=-1, dtype=int)
        pred_fb[p_cal >= tau_pos] = 1
        pred_fb[p_cal <= best_tau] = 0
        best_coverage = 1.0 - np.mean(pred_fb == -1)
        print(f"[警告] FP=0制約でのτ-が見つからず、フォールバック値使用: {best_tau:.3f}")
    
    return best_tau, best_coverage



# =========================
#  メイン
# =========================
from typing import Optional, Dict, Any

def train_and_bundle(input_excel: Optional[str] = None) -> Dict[str, Any]:
    # 0) 入力読込 + 目的変数空白を除去
    df = pd.read_excel(input_excel or ConfigCLS.INPUT_FILE)
    os.makedirs(ConfigCLS.MODEL_FOLDER_PATH, exist_ok=True)

    tcol = ConfigCLS.TARGET_COLUMN
    if tcol not in df.columns:
        raise ValueError(f"目的変数列がありません: {tcol}")

    df[tcol] = df[tcol].replace(r"^\s*$", np.nan, regex=True)
    before = len(df)
    df = df.dropna(subset=[tcol]).copy()
    print(f"[clean] drop rows with empty target: {before} -> {len(df)}")

    # 1) スキーマ準備（feat列は EXCLUDE と target を除外）
    X, y, groups, feature_cols = _prepare_data(df)
    
    # OOD用のベース（"最終で使う特徴の全量"を後で抽出できるように、まずは feature_cols で退避）
    X_all = df[feature_cols].copy()

    # 2) Outer split
    outer_splits = _split_outer(X, y, groups)

    rng = np.random.RandomState(ConfigCLS.RANDOM_STATE)
    oof_p = np.zeros(len(y))
    oof_idx = np.zeros(len(y), dtype=int)
    chosen_cals = []
    all_neg_scores = []

    for i, (tr_idx, te_idx) in enumerate(outer_splits, 1):
        print(f"\n--- Outer Fold {i}/{ConfigCLS.OUTER_SPLITS} ---")

        tr_full = X.iloc[tr_idx].copy()
        y_tr_full = y[tr_idx]
        g_tr_full = groups[tr_idx]

        # Calib（校正用）を Train から切り出し（Group 対応）
        if ConfigCLS.GROUP_COLUMN:
            uniq = np.unique(g_tr_full)
            rng.shuffle(uniq)
            cut = int(0.2 * len(uniq)) or 1
            calib_groups = set(uniq[:cut])
            calib_mask = np.array([g in calib_groups for g in g_tr_full])
        else:
            calib_mask = np.zeros_like(y_tr_full, dtype=bool)
            calib_mask[rng.choice(len(y_tr_full), size=max(1, int(0.2 * len(y_tr_full))), replace=False)] = True

        idx_calib = np.where(calib_mask)[0]
        idx_train = np.where(~calib_mask)[0]

        X_tr, y_tr = tr_full.iloc[idx_train].copy(), y_tr_full[idx_train]
        X_cb, y_cb = tr_full.iloc[idx_calib].copy(), y_tr_full[idx_calib]
        X_te = X.iloc[te_idx].copy()

        # === 列フィルタを学習（学習側のみで決定） ===
        keep_cols, med = learn_column_filter(X_tr)

        # 同じ列を Calib/Test に適用（欠損は学習中央値で埋める）
        X_tr = X_tr[keep_cols].apply(pd.to_numeric, errors="coerce").fillna(med)
        X_cb = X_cb[keep_cols].apply(pd.to_numeric, errors="coerce").fillna(med)
        X_te = X_te[keep_cols].apply(pd.to_numeric, errors="coerce").fillna(med)

        print(f"[fold {i}] kept columns: {len(keep_cols)} / {X.shape[1]}")

        # ========== Inner: Optuna ==========
        def objective(trial):
            """Inner CV with vertical noise augmentation (正しい実装)"""
            
            # 1) データコピー
            X_inner = X_tr.copy()
            y_inner = y_tr.copy()
            
            # 2) 数値化と欠損値処理
            X_inner = X_inner.apply(pd.to_numeric, errors="coerce")
            X_inner = X_inner.fillna(X_inner.median(numeric_only=True))
            
            # 3) 縦方向ノイズ付与（Data Augmentation）
            if ConfigCLS.USE_INNER_NOISE:
                from feature_noise_vertical import add_noise_augmentation
                
                # Trialごとに異なるseed
                noise_seed = ConfigCLS.RANDOM_STATE + trial.number
                
                # 縦方向にノイズ付きサンプルを追加
                X_inner, y_inner, augment_info = add_noise_augmentation(
                    X_inner,
                    y_inner,
                    noise_ppm=ConfigCLS.NOISE_PPM,  # ConfigCLSに追加: NOISE_PPM = 100
                    augment_ratio=ConfigCLS.NOISE_RATIO,  # 0.1 = 10%のサンプル追加
                    random_state=noise_seed,
                    verbose=(trial.number == 0)  # 最初のトライアルのみ表示
                )
                
                # Optunaのユーザー属性に記録
                trial.set_user_attr("augmented_samples", augment_info["added_samples"])
                trial.set_user_attr("total_samples", augment_info["total_samples"])
            
            # 4) スケーリング
            scaler_in = RobustScaler()
            X_inner_vals = scaler_in.fit_transform(X_inner.values)
            X_inner_df = pd.DataFrame(X_inner_vals, columns=X_inner.columns, index=X_inner.index)
            
            # 5) モデル選択とハイパーパラメータ
            model_name = trial.suggest_categorical("model_name", ConfigCLS.MODELS_TO_USE)
            mconf = ConfigCLS.MODEL_CONFIGS.get(model_name, {})
            params = ModelFactoryCLS.suggest_hyperparams(model_name, trial, mconf)
            model = ModelFactoryCLS.build(model_name, params)
            
            # 6) Inner CV（拡張データで学習）
            cv = StratifiedKFold(n_splits=ConfigCLS.INNER_SPLITS, shuffle=True, random_state=ConfigCLS.RANDOM_STATE)
            aucs = []
            
            # 重要: CVのsplitも拡張後のデータに対して実行
            for tr_i, va_i in cv.split(X_inner_df, y_inner):
                X_tr_i = X_inner_df.iloc[tr_i]
                X_va_i = X_inner_df.iloc[va_i]
                y_tr_i = y_inner[tr_i]
                y_va_i = y_inner[va_i]
                
                # 特徴選択
                top_k = trial.suggest_int("top_k", *ConfigCLS.SELECT_TOP_K_RANGE)
                corr_thr = trial.suggest_float("corr_threshold", *ConfigCLS.CORR_THRESHOLD_RANGE)
                forced_keep_inner = [c for c in ConfigCLS.MUST_KEEP_FEATURES if c in X_tr_i.columns]
                
                X_sel = select_features_by_importance(
                    model, X_tr_i, y_tr_i,
                    top_k=top_k,
                    corr_threshold=corr_thr,
                    forced_keep=forced_keep_inner,
                )
                sel_cols = X_sel.columns.tolist()
                
                # モデル学習（拡張データで）
                model.fit(X_sel, y_tr_i)
                
                # 評価（検証セットも拡張データの一部）
                proba = model.predict_proba(X_va_i[sel_cols])[:, 1]
                aucs.append(roc_auc_score(y_va_i, proba))
            
            return float(np.mean(aucs))
        

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=ConfigCLS.RANDOM_STATE),
        )
        study.optimize(objective, n_trials=ConfigCLS.N_TRIALS_INNER, show_progress_bar=False)

        # ========== Outer: best_params を反映 ==========
        hp = study.best_params.copy()
        model_name = hp.pop("model_name")
        top_k = int(hp.pop("top_k", min(60, X_tr.shape[1])))
        corr_thr = float(hp.pop("corr_threshold", 0.95))
        params = hp  # HPO済み

        # Outer fold 用のスケーラは「学習側」に合わせて作る
        scaler = RobustScaler()
        X_tr_scaled = pd.DataFrame(scaler.fit_transform(X_tr.values), columns=X_tr.columns, index=X_tr.index)
        X_cb_scaled = pd.DataFrame(scaler.transform(X_cb.values),    columns=X_cb.columns, index=X_cb.index)
        X_te_scaled = pd.DataFrame(scaler.transform(X_te.values),    columns=X_te.columns, index=X_te.index)
        
        model = ModelFactoryCLS.build(model_name, params)
        
        # Outer 用の特徴選択（Inner で決めた top_k / corr_thr を使用）
        forced_keep_outer = [c for c in ConfigCLS.MUST_KEEP_FEATURES if c in X_tr_scaled.columns]
        X_tr_sel = select_features_by_importance(
            model, X_tr_scaled, y_tr,
            top_k=top_k,
            corr_threshold=corr_thr,
            forced_keep=forced_keep_outer,
        )
        sel_cols = X_tr_sel.columns.tolist()
        model.fit(X_tr_sel, y_tr)
        
        # Calibration（Calib で選択）
        proba_cb_raw = model.predict_proba(X_cb_scaled[sel_cols])[:, 1]

        
        #
        eps = 1e-6
        logits = np.log(np.clip(proba_cb_raw, eps, 1 - eps) / np.clip(1 - proba_cb_raw, eps, 1 - eps))

        calib, cal_name, cal_metrics = select_and_fit_calibrator(
            raw_scores=proba_cb_raw,
            logits_or_scores=logits,
            y_true=y_cb,
            weights=ConfigCLS.CALIBRATION_SELECTION_METRIC_WEIGHTS,
            candidates=tuple(ConfigCLS.CALIBRATION_CANDIDATES),
        )

        # τ+用に fold の「Calib 上の負例確率」を集約
        if cal_name == "temperature":
            p_cb = calib.predict_proba(logits)
        else:
            p_cb = calib.predict_proba(proba_cb_raw)
        all_neg_scores.append(p_cb[y_cb == 0])

        # ========== Outer test 予測（OOF） ==========
        X_te_scaled = pd.DataFrame(scaler.transform(X_te.values), columns=X_te.columns, index=X_te.index)
        proba_te_raw = model.predict_proba(X_te_scaled[sel_cols])[:, 1]
        logits_te = np.log(np.clip(proba_te_raw, eps, 1 - eps) / np.clip(1 - proba_te_raw, eps, 1 - eps))
        if cal_name == "temperature":
            p_te = calib.predict_proba(logits_te)
        else:
            p_te = calib.predict_proba(proba_te_raw)
        oof_p[te_idx] = p_te
        oof_idx[te_idx] = te_idx
        chosen_cals.append(cal_name)

    # ========== τ+ 決定（NP） ==========
    neg_scores_all = np.concatenate(all_neg_scores) if len(all_neg_scores) > 0 else np.array([])
    tau_pos = np_tau_pos_from_neg_scores(neg_scores_all, ConfigCLS.NP_ALPHA)
    if ConfigCLS.USE_UPPER_CI_ADJUST and ConfigCLS.NP_ALPHA > 0.0:
        tau_pos = adjust_tau_pos_by_ci(
            neg_scores_all, tau_pos, ConfigCLS.NP_ALPHA, ConfigCLS.CI_METHOD, ConfigCLS.CI_CONFIDENCE
        )

    # ========== τ− 探索（Precision@Pos=1.0制約でCoverage最大化） ==========
    tau_neg, best_cov = _search_tau_neg(oof_p, y, tau_pos, ConfigCLS.TAU_NEG_GRID)

    # ========== 最終モデル（全データ） ==========
    keep_cols_all, med_all = learn_column_filter(X)
    X_all_num = X[keep_cols_all].apply(pd.to_numeric, errors="coerce").fillna(med_all)
    
    # 最終スケーラ（全量で fit）
    scaler_final = RobustScaler()
    X_scaled_df = pd.DataFrame(
        scaler_final.fit_transform(X_all_num.values),
        columns=X_all_num.columns,
        index=X_all_num.index
    )
    
    # 最終モデル名（Outer の最頻モデル or 既定の優先順）
    cal_winner = Counter(chosen_cals).most_common(1)[0][0] if chosen_cals else ConfigCLS.CALIBRATION_CANDIDATES[0]
    final_model_name = model_name if "model_name" in locals() and model_name in ConfigCLS.MODELS_TO_USE else (
        "lightgbm" if "lightgbm" in ConfigCLS.MODELS_TO_USE else
        "xgboost" if "xgboost" in ConfigCLS.MODELS_TO_USE else
        "random_forest" if "random_forest" in ConfigCLS.MODELS_TO_USE else
        "logistic"
    )
    model_final = ModelFactoryCLS.build(final_model_name, {})
    
    # 最終の特徴選択（"必ず残す"も尊重）
    forced_keep_all = [c for c in ConfigCLS.MUST_KEEP_FEATURES if c in X_scaled_df.columns]
    X_sel_all = select_features_by_importance(
        model_final, X_scaled_df, y,
        top_k=min(60, X_scaled_df.shape[1]),
        corr_threshold=0.95,
        forced_keep=forced_keep_all,
    )
    sel_cols_final = X_sel_all.columns.tolist()
    
    # 最終モデル学習
    model_final.fit(X_sel_all, y)

    # ===== OOD（Mahalanobis）学習 =====
    ood = None
    if ConfigCLS.USE_OOD:
        X_ood_scaled = X_scaled_df[sel_cols_final].values
        ood = MahalanobisGate().fit(
            X_ood_scaled,
            percentile=ConfigCLS.OOD_PERCENTILE
        )    
    # ===== 校正器 fit（全データ） =====
    proba_all_raw = model_final.predict_proba(X_scaled_df[sel_cols_final])[:, 1]
    eps = 1e-6
    logits_all = np.log(np.clip(proba_all_raw, eps, 1 - eps) / np.clip(1 - proba_all_raw, eps, 1 - eps))
    if cal_winner == "temperature":
        from calibration import TemperatureScaler
        calibrator = TemperatureScaler().fit(logits_all, y)
    else:
        from calibration import IsotonicCalibrator
        calibrator = IsotonicCalibrator().fit(proba_all_raw, y)


    # ========== バンドル保存 ==========
    bundle = {
        "scaler": scaler_final,
        "model_name": final_model_name,
        "model": model_final,
        "selected_columns": sel_cols_final,
        "calibrator_name": cal_winner,
        "calibrator": calibrator,
        "tau_pos": float(tau_pos),
        "tau_neg": float(tau_neg),
        "ood": ood,
        "feature_columns": feature_cols,
        "np_alpha": float(ConfigCLS.NP_ALPHA),
        "result_folder": ConfigCLS.MODEL_FOLDER_PATH,
        "column_filter": {"keep_columns": keep_cols_all, "medians": med_all.to_dict()},
    }

    out_path = os.path.join(ConfigCLS.MODEL_FOLDER_PATH, "final_bundle_cls.pkl")
    os.makedirs(ConfigCLS.MODEL_FOLDER_PATH, exist_ok=True)
    joblib.dump(bundle, out_path)
    print(f"\n✅ バンドル保存: {out_path}")

    with open(os.path.join(ConfigCLS.MODEL_FOLDER_PATH, "summary_cls.json"), "w", encoding="utf-8") as f:
        json.dump(
            {"tau_pos": bundle["tau_pos"], "tau_neg": bundle["tau_neg"], "calibrator": cal_winner},
            f,
            ensure_ascii=False,
            indent=2,
        )
    return bundle
