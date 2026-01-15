from __future__ import annotations
import re
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Set
from config_cls import ConfigCLS

def _normalize_columns(cols: List[str]) -> List[str]:
    # 余計な空白やBOM/不可視文字の除去
    return [re.sub(r'\s+', '', str(c)).strip() for c in cols]

def detect_schema(
    df: pd.DataFrame,
    target_column: Optional[str] = None,
    positive_labels: Optional[List] = None,
    negative_labels: Optional[List] = None,
    id_candidates: Optional[List[str]] = None,
    date_candidates: Optional[List[str]] = None,
    group_column: Optional[str] = None,
) -> Dict:
    cols_original = list(df.columns)
    # 列名クリーニング（必要なら）
    # df.columns = _normalize_columns(list(df.columns))  # ← 有効化したい場合のみ

    # 目的変数
    t_col = target_column or ConfigCLS.TARGET_COLUMN
    if t_col not in df.columns:
        raise ValueError(f"目的変数列が見つかりません: {t_col}")

    # 除外候補の集合を構成
    exclude: Set[str] = set(ConfigCLS.EXCLUDE_COLUMNS)
    exclude.add(t_col)  # 分類の目的変数を除外

    # ID/日時の候補も除外（実際に存在するもののみ）
    for c in (id_candidates or ConfigCLS.ID_COLUMNS_CANDIDATES or []):
        if c in df.columns:
            exclude.add(c)
    for c in (date_candidates or ConfigCLS.DATE_COLUMNS_CANDIDATES or []):
        if c in df.columns:
            exclude.add(c)

    # Excel由来の「Unnamed: x」は全部除外
    for c in df.columns:
        if isinstance(c, str) and c.startswith("Unnamed:"):
            exclude.add(c)

    # group_column は除外
    if group_column and group_column in df.columns:
        exclude.add(group_column)

    # --- 説明変数候補（ALLOWED_FEATURESに含まれるもののみ） ---
    # まず除外対象でない列を取得
    potential_candidates = [c for c in df.columns if c not in exclude]
    
    # ALLOWED_FEATURESに含まれるもののみを候補とする
    feature_candidates = [c for c in potential_candidates if c in ConfigCLS.ALLOWED_FEATURES]

    # 「必ず残す」列が候補に無ければ、存在する分だけ追加
    must_keep = [c for c in ConfigCLS.MUST_KEEP_FEATURES if c in df.columns]
    for c in must_keep:
        if c not in feature_candidates:
            feature_candidates.append(c)

    # 最終：数値化できる列のみを説明変数に（学習直前にさらに to_numeric/median 埋めを実施）
    Xnum = df[feature_candidates].apply(pd.to_numeric, errors="coerce")
    feature_columns = list(Xnum.columns)

    # y と group
    y_raw = df[t_col].values
    
    # ラベル変換（分類タスク用）
    y = np.full_like(y_raw, np.nan, dtype=float)
    for i, val in enumerate(y_raw):
        if val in ConfigCLS.POSITIVE_LABELS:
            y[i] = 1.0
        elif val in ConfigCLS.NEGATIVE_LABELS:
            y[i] = 0.0
        # マッチしない値はNaNのまま（後で除去）
    
    g = df[group_column].values if group_column and group_column in df.columns else None

    return {
        "feature_columns": feature_columns,
        "y": y,
        "group_column": group_column if group_column in df.columns else None,
        "excluded": sorted(list(exclude)),
        "must_keep_effective": must_keep,
        "columns_original": cols_original,
    }
