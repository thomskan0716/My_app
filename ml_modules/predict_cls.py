"""
学習済みバンドル（final_bundle_cls.pkl）を使って、新規Excelに三値判定を出力するスクリプト
- 校正確率 p_cal の計算
- τ+ / τ− による三値化（P / G / N）
- OODゲート（Mahalanobis）で G 強制
"""
import os
import joblib
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any
from config_cls import ConfigCLS
from schema_detect import detect_schema

def _ensure_columns(df: pd.DataFrame, needed_cols):
    missing = [c for c in needed_cols if c not in df.columns]
    if missing:
        raise ValueError(f"予測必要列が見つかりません: {missing}")
    return df[needed_cols].copy()

def _calibrate_probs(calibrator_name, calibrator, raw_scores: np.ndarray) -> np.ndarray:
    eps = 1e-6
    if calibrator_name == "temperature":
        logits = np.log(np.clip(raw_scores, eps, 1-eps) / np.clip(1-raw_scores, eps, 1-eps))
        return calibrator.predict_proba(logits)
    else:
        return calibrator.predict_proba(raw_scores)

def predict_excel(
    bundle_path: Optional[str]=None,
    input_excel: Optional[str]=None,
    output_excel: Optional[str]=None,
) -> str:
    # 0) バンドル読み込み
    bundle_path = bundle_path or os.path.join(ConfigCLS.MODEL_FOLDER_PATH, "final_bundle_cls.pkl")
    if not os.path.exists(bundle_path):
        raise FileNotFoundError(f"バンドルが見つかりません: {bundle_path}")
    bundle: Dict[str, Any] = joblib.load(bundle_path)

    input_excel = input_excel or ConfigCLS.PREDICT_INPUT_FILE
    if input_excel is None:
        raise ValueError("予測用Excelのパスを指定してください（ConfigCLS.PREDICT_INPUT_FILE または引数）。")
    df_new = pd.read_excel(input_excel)

    # 1) 特徴の整合性確保（学習時の feature_columns を基準）
    feature_cols = bundle.get("feature_columns", None)
    if feature_cols is None:
        # 互換: schema_detect で除外済みの列を推定
        schema = detect_schema(df_new, target_column=ConfigCLS.TARGET_COLUMN,
                               positive_labels=ConfigCLS.POSITIVE_LABELS,
                               negative_labels=ConfigCLS.NEGATIVE_LABELS,
                               id_candidates=ConfigCLS.ID_COLUMNS_CANDIDATES,
                               date_candidates=ConfigCLS.DATE_COLUMNS_CANDIDATES,
                               group_column=ConfigCLS.GROUP_COLUMN)
        feature_cols = schema["feature_columns"]
    X = _ensure_columns(df_new, feature_cols)

    # ★ 追加：列フィルタ（keep + median埋め）
    cf = bundle.get("column_filter")
    if cf is not None:
        keep_cols = cf["keep_columns"]
        med_all = pd.Series(cf["medians"])
        # ない列は作る（全部NaN→この後のfillnaで中央値）
        for c in keep_cols:
            if c not in X.columns:
                X[c] = np.nan
        X = X[keep_cols].apply(pd.to_numeric, errors="coerce").fillna(med_all)

    # 2) スケーリング＆選抜列
    scaler = bundle["scaler"]
    X_scaled = pd.DataFrame(scaler.transform(X.values), columns=X.columns, index=X.index)
    sel_cols = bundle["selected_columns"]
    X_sel = X_scaled[sel_cols].copy()

    # 3) モデル確率 → 校正確率 p_cal
    model = bundle["model"]
    raw_scores = model.predict_proba(X_sel)[:,1]
    p_cal = _calibrate_probs(bundle["calibrator_name"], bundle["calibrator"], raw_scores)

    # 4) OOD（あれば）
    ood_flag = np.zeros(len(X_sel), dtype=int)
    maha = np.zeros(len(X_sel))
    
    if bundle.get("ood", None) is not None:
        ood = bundle["ood"]
        # ★ 修正：OOD用の列は selected_columns と同じにする
        # （学習時にOODはX_scaled_df[sel_cols_final]で学習されているため）
        ood_cols = bundle["selected_columns"]
        
        # X_selは既にスケール済み＆選択済みなのでそのまま使用
        X_ood_scaled = X_sel.values
        
        maha = ood.score(X_ood_scaled)
        ood_flag = (maha >= ood.threshold_).astype(int)

    # 5) 三値化
    tau_pos = float(bundle["tau_pos"])
    tau_neg = float(bundle["tau_neg"])
    pred = np.full(len(p_cal), fill_value=-1)  # -1=G, 0=N, 1=P
    pred[p_cal >= tau_pos] = 1
    pred[p_cal <= tau_neg] = 0
    # OODはGに強制
    pred[ood_flag == 1] = -1

    # 6) 出力
    # OUTPUT_COLUMNS_BASEが定義されていない場合の対処
    result = pd.DataFrame({
        "pred_label": pred,
        "p_cal": p_cal,
        "tau_pos": tau_pos,
        "tau_neg": tau_neg,
        "ood_flag": ood_flag,
        "maha_dist": maha,
    }, index=df_new.index)

    out_df = pd.concat([df_new, result], axis=1)

    # 出力パスの決定
    if output_excel is None:
        # PREDICT_OUTPUT_FILEが定義されていない場合の処理
        if hasattr(ConfigCLS, 'PREDICT_OUTPUT_FILE') and ConfigCLS.PREDICT_OUTPUT_FILE:
            output_excel = ConfigCLS.PREDICT_OUTPUT_FILE
        else:
            base = os.path.splitext(os.path.basename(input_excel))[0]
            output_excel = os.path.join(ConfigCLS.PREDICTION_FOLDER_PATH, f"predictions_{base}.xlsx")
    
    os.makedirs(os.path.dirname(output_excel) or '.', exist_ok=True)
    out_df.to_excel(output_excel, index=False)
    return output_excel

if __name__ == "__main__":
    path = predict_excel()
    print(f"✅ 予測Excelを保存: {path}")