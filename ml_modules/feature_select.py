from typing import Iterable, Optional, Sequence, List
import numpy as np
import pandas as pd
from sklearn.base import clone

def _infer_importance(fitted_model, columns: List[str]) -> pd.Series:
    """学習済みモデルから重要度を推定して Series で返す。
    - tree 系: feature_importances_
    - 線形系: coef_ の絶対値
    - どれも無ければ一様重み
    """
    if hasattr(fitted_model, "feature_importances_"):
        imp = getattr(fitted_model, "feature_importances_")
        return pd.Series(np.asarray(imp, dtype=float), index=columns).fillna(0.0)

    if hasattr(fitted_model, "coef_"):
        coef = getattr(fitted_model, "coef_")
        coef = np.asarray(coef, dtype=float)
        if coef.ndim == 2:  # (1, n_features) の想定
            coef = coef[0]
        coef = np.abs(coef)
        return pd.Series(coef, index=columns).fillna(0.0)

    # フォールバック：一様
    return pd.Series(np.ones(len(columns), dtype=float), index=columns)

def _corr_prune(X: pd.DataFrame, candidates: List[str], kept: List[str], thr: float) -> List[str]:
    """相関フィルタ（累積で |r| >= thr を避ける）。kept は固定で先に確保しておく。"""
    selected = list(kept)  # すでに確保済み（forced_keep）
    for col in candidates:
        if col in selected:
            continue
        if not selected:
            selected.append(col)
            continue
        # ★ 修正点：DataFrame.corrwith(Series) を使って最大相関を求める
        r = np.abs(X[selected].corrwith(X[col])).max()
        if not np.isfinite(r):
            r = 0.0
        if r < thr:
            selected.append(col)
    return selected

def select_features_by_importance(
    model,
    X: pd.DataFrame,
    y: Sequence,
    *,
    top_k: int = 60,
    corr_threshold: float = 0.95,
    forced_keep: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """重要度→相関フィルタで列選択。forced_keep は常に残す。
    - model は原本を汚さないよう clone() でフィット
    - 相関フィルタは forced_keep を事前固定してから適用
    """
    Xn = X.apply(pd.to_numeric, errors="coerce")
    Xn = Xn.fillna(Xn.median(numeric_only=True))

    # 強制保持（X に存在するものだけ有効）
    keep_set = set([c for c in (forced_keep or []) if c in Xn.columns])

    # モデルをクローンして学習（原本は汚さない）
    m = clone(model)
    m.fit(Xn, y)
    imp = _infer_importance(m, list(Xn.columns))

    # forced_keep を先に確保し、それ以外の候補を重要度降順に
    non_keep = [c for c in imp.sort_values(ascending=False).index if c not in keep_set]

    # top_k は「forced_keep 以外」での上限
    if top_k is None or top_k <= 0:
        top_k = min(60, Xn.shape[1])
    non_keep = non_keep[:top_k]

    # 相関フィルタ：keep は固定、non_keep から追加
    selected_cols = _corr_prune(Xn, non_keep, kept=list(keep_set), thr=corr_threshold)

    # 最終的な列に整形
    selected_cols = [c for c in Xn.columns if c in selected_cols]
    return Xn[selected_cols]
