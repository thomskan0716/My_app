# -*- coding: utf-8 -*-
"""
Pareto frontier calculation & plotting (multi-objective, with min/max per feature)
Refactored + Extended:
  - 高速パレート抽出（チャンク分割＋逐次マージ：順序非依存）
  - 可視化の納得感向上（2Dペア前面の重ね描き, ε-支配の図用間引き, 代表サンプル化）
  - 階層化（lexicographic）：優先セット→残りで二段目の前面
  - 3パターン一括算出（4目的/3→2/2→2、必須項の組合せ制御）
  - 3項目・2項目パターンの図を自動保存
  - 安全ガードと詳細ログ
"""

import os
import sys
import glob
import shutil
import warnings
from pathlib import Path
import itertools as it

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ================= プロジェクトルート/パス =================
PROJECT_ROOT = Path.cwd()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PYTHON_CODE_FOLDER = PROJECT_ROOT / "00_Pythonコード"
if str(PYTHON_CODE_FOLDER) not in sys.path:
    sys.path.insert(0, str(PYTHON_CODE_FOLDER))

# ================= 設定読み込み =================
from config import Config

# 日本語フォント設定
import matplotlib.font_manager as fm  # noqa: F401
plt.rcParams["font.family"] = getattr(Config, "JAPANESE_FONT_FAMILY", "Yu Gothic")
plt.rcParams["axes.unicode_minus"] = getattr(Config, "JAPANESE_FONT_UNICODE_MINUS", False)

# ================= ユーザ設定 =================
INPUT_FOLDER   = getattr(Config, "PREDICTION_FOLDER", ".")
INPUT_FILE     = getattr(Config, "PREDICTION_OUTPUT_FILE", "Prediction_output.xlsx")
OUTPUT_FOLDER  = os.path.join(".", getattr(Config, "PARETO_OUTPUT_FOLDER", "04_pareto"))
PARETO_OBJECTIVES = getattr(Config, "PARETO_OBJECTIVES", {})  # {"摩耗量":"min", ...}

# === パフォーマンス・可視化オプション（Configから読み込み） ===
PARETO_CHUNK_SIZE          = int(getattr(Config, "PARETO_CHUNK_SIZE", 10000))
PARETO_VERBOSE             = bool(getattr(Config, "PARETO_VERBOSE", True))
PARETO_MAX_ROWS_FOR_STRICT = getattr(Config, "PARETO_MAX_ROWS_FOR_STRICT", None)
PARETO_ENABLE_PLOTTING     = bool(getattr(Config, "PARETO_ENABLE_PLOTTING", True))
PARETO_PLOT_SHOW_PAIRWISE_FRONT = bool(getattr(Config, "PARETO_PLOT_SHOW_PAIRWISE_FRONT", True))
PARETO_USE_EPSILON_PLOT = bool(getattr(Config, "PARETO_USE_EPSILON_PLOT", True))
PARETO_EPSILON_REL      = float(getattr(Config, "PARETO_EPSILON_REL", 0.02))
PARETO_PLOT_MAX_POINTS  = int(getattr(Config, "PARETO_PLOT_MAX_POINTS", 1500))

# === プロット既定（Configから読み込み） ===
def _cfg(name, default): return getattr(Config, name, default)
_PLOT_FIGSIZE      = _cfg("PARETO_PLOT_FIGSIZE", (7, 6))
_PLOT_ALPHA_ALL    = _cfg("PARETO_PLOT_ALPHA_ALL", 0.35)
_PLOT_SIZE_ALL     = _cfg("PARETO_PLOT_SIZE_ALL", 18)
_PLOT_SIZE_PARETO  = _cfg("PARETO_PLOT_SIZE_PARETO", 36)
_PLOT_EDGECOLORS   = _cfg("PARETO_PLOT_EDGECOLORS", "r")
_PLOT_FACECOLORS   = _cfg("PARETO_PLOT_FACECOLORS", "none")
_PLOT_LINEWIDTHS   = _cfg("PARETO_PLOT_LINEWIDTHS", 1.2)
_PLOT_GRID_ALPHA   = _cfg("PARETO_PLOT_GRID_ALPHA", 0.3)
_PLOT_DPI          = _cfg("PARETO_PLOT_DPI", 300)
_PLOTS_FOLDER_NAME = _cfg("PARETO_PLOTS_FOLDER", "pareto_plots")
_PLOT_FN_FMT       = _cfg("PARETO_PLOT_FILENAME_FORMAT", "pareto_{x_logical}__vs__{y_logical}.png")
_LABEL_MIN_SFX     = _cfg("PARETO_LABEL_MIN_SUFFIX", "（↓良い）")
_LABEL_MAX_SFX     = _cfg("PARETO_LABEL_MAX_SUFFIX", "（↑良い）")
_XLSX_NAME         = _cfg("PARETO_EXCEL_FILENAME", "pareto_frontier.xlsx")
_SHEET_PARETO_ONLY = _cfg("PARETO_SHEET_PARETO_ONLY", "pareto_only")
_SHEET_PARETO_ONLY_PLOT = _cfg("PARETO_SHEET_PARETO_ONLY_PLOT", "pareto_only_plotmask")
_SHEET_META        = _cfg("PARETO_SHEET_META", "meta")

# === 階層化ケース（論理名で指定：Configから読み込み） ===
LEXI_CASES = getattr(Config, "LEXI_CASES", [])  # Config.LEXI_CASESで設定（未定義時は空リスト）

# ================= ユーティリティ =================
def resolve_column(df: pd.DataFrame, name: str) -> str:
    """論理名（prediction_は付けない）→ 実列名を解決"""
    cut_name = getattr(Config, "CUTTING_TIME_COLUMN_NAME", "切削時間")
    if name == cut_name:
        if name in df.columns: return name
        raise KeyError(f"計算列が見つかりません: '{name}'")
    pref = getattr(Config, "PREDICTION_COLUMN_PREFIX", "prediction")
    pred_name = f"{pref}_{name}"
    if pred_name in df.columns: return pred_name
    if name in df.columns: return name
    raise KeyError(f"必要列が見つかりません: '{pred_name}' も '{name}' もありません。")

def logical_name_from_resolved(resolved_col: str) -> str:
    pref = f"{getattr(Config, 'PREDICTION_COLUMN_PREFIX', 'prediction')}_"
    return resolved_col.replace(pref, "") if resolved_col.startswith(pref) else resolved_col

def to_minimization_matrix(df_eval: pd.DataFrame, cols: list, obj_dirs: dict) -> np.ndarray:
    """最小化問題に変換（maxは符号反転）"""
    vals = df_eval[cols].to_numpy(dtype=np.float32, copy=False)
    for j, col in enumerate(cols):
        logical_name = logical_name_from_resolved(col)
        if obj_dirs.get(logical_name, "min").lower() == "max":
            vals[:, j] *= -1.0
    return vals

def pretty_axis_label(logical_name: str, direction: str) -> str:
    return f"{logical_name}{_LABEL_MIN_SFX}" if direction.lower() == "min" else f"{logical_name}{_LABEL_MAX_SFX}"

# ---- 2目的(最小化)の高速前面（O(M log M)） ----
def pareto_2d_mask_min(values_2d: np.ndarray) -> np.ndarray:
    x = values_2d[:, 0]; y = values_2d[:, 1]
    order = np.argsort(x, kind="mergesort")
    y_sorted = y[order]
    keep_sorted = np.zeros_like(y_sorted, dtype=bool)
    best = np.inf
    for i in range(len(y_sorted)):
        if y_sorted[i] < best:
            keep_sorted[i] = True
            best = y_sorted[i]
    keep = np.zeros_like(keep_sorted); keep[order] = keep_sorted
    return keep

def resolve_cols_from_logicals(df: pd.DataFrame, logical_names: list) -> list:
    return [resolve_column(df, nm) for nm in logical_names]

def make_2d_values(df: pd.DataFrame, x_col: str, y_col: str,
                   x_dir: str, y_dir: str) -> np.ndarray:
    xv = pd.to_numeric(df[x_col], errors="coerce").to_numpy(dtype=np.float32)
    yv = pd.to_numeric(df[y_col], errors="coerce").to_numpy(dtype=np.float32)
    if str(x_dir).lower() == "max": xv *= -1
    if str(y_dir).lower() == "max": yv *= -1
    return np.c_[xv, yv]

# ---- 2Dプロット共通関数（Pattern B/C 用） ----
def plot_2d_front(df_universe: pd.DataFrame, df_front: pd.DataFrame,
                  x_col: str, y_col: str,
                  x_logical: str, y_logical: str,
                  x_dir: str, y_dir: str,
                  out_png: str):
    """Universe上の散布＋frontのハイライト＋2Dペア前面（黒★）を保存"""
    x_all = pd.to_numeric(df_universe[x_col], errors="coerce")
    y_all = pd.to_numeric(df_universe[y_col], errors="coerce")
    mask_all = np.isfinite(x_all) & np.isfinite(y_all)
    if not mask_all.any() or df_front.empty:
        print(f"⚠ プロットをスキップ（有効データ or front が空）：{out_png}")
        return

    fig, ax = plt.subplots(figsize=_PLOT_FIGSIZE)

    # All
    ax.scatter(x_all[mask_all], y_all[mask_all],
               alpha=_PLOT_ALPHA_ALL, label="All", s=_PLOT_SIZE_ALL)

    # Front
    ax.scatter(df_front[x_col], df_front[y_col],
               label="Pareto front", s=_PLOT_SIZE_PARETO,
               edgecolors=_PLOT_EDGECOLORS, facecolors=_PLOT_FACECOLORS,
               linewidths=_PLOT_LINEWIDTHS)

    # 2Dペア専用前面（黒★）
    if PARETO_PLOT_SHOW_PAIRWISE_FRONT:
        vals_pair = make_2d_values(df_universe.loc[mask_all], x_col, y_col, x_dir, y_dir)
        mask_pair = pareto_2d_mask_min(vals_pair)
        df_pair = df_universe.loc[mask_all, [x_col, y_col]]
        ax.scatter(df_pair[x_col].to_numpy()[mask_pair],
                   df_pair[y_col].to_numpy()[mask_pair],
                   marker="*", s=40, label="2D pairwise front")

    ax.set_xlabel(pretty_axis_label(x_logical, x_dir))
    ax.set_ylabel(pretty_axis_label(y_logical, y_dir))
    ax.set_title(f"Pareto: {x_logical} vs {y_logical}  (front={len(df_front):,}/{int(mask_all.sum()):,})")
    ax.grid(True, alpha=_PLOT_GRID_ALPHA)
    if ax.get_legend_handles_labels()[0]:
        ax.legend()

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=_PLOT_DPI)
    plt.close(fig)
    print(f"🖼 保存: {out_png}")

# ======== 高速パレート抽出：チャンク分割 + 逐次マージ（順序非依存） ========
def _dominated_mask_min(values: np.ndarray) -> np.ndarray:
    """小規模集合に対する厳密支配判定（最小化, O(M^2)）"""
    n = values.shape[0]
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        if dominated[i]: continue
        v = values[i]
        le = (values <= v).all(axis=1)
        lt = (values <  v).any(axis=1)
        d = le & lt
        d[i] = False
        dominated |= d
    return dominated

def pareto_front_mask_chunked(values: np.ndarray, chunk_size: int = 10_000,
                              verbose: bool = True) -> np.ndarray:
    n = values.shape[0]
    idx_global = np.arange(n)
    pf_vals = None
    pf_idx  = None

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        block = values[start:end]
        bidx  = idx_global[start:end]

        if pf_vals is not None and len(pf_vals):
            le = (pf_vals[:, None, :] <= block[None, :, :]).all(axis=2)
            lt = (pf_vals[:, None, :] <  block[None, :, :]).any(axis=2)
            dom = (le & lt).any(axis=0)
            if dom.any():
                block = block[~dom]; bidx = bidx[~dom]

        if block.size == 0:
            if verbose: print(f"[Pareto] processed {end}/{n} (block fully dominated)")
            continue

        if pf_vals is None:
            comb = block; cidx = bidx
        else:
            comb = np.vstack([pf_vals, block]); cidx = np.concatenate([pf_idx, bidx])

        d = _dominated_mask_min(comb)
        keep = ~d
        pf_vals = comb[keep]
        pf_idx  = cidx[keep]

        if verbose:
            print(f"[Pareto] processed {end}/{n}, current frontier size = {len(pf_vals)}")

    mask = np.zeros(n, dtype=bool)
    if pf_idx is not None and len(pf_idx):
        mask[pf_idx] = True
    return mask

# ---- 図用：ε-支配で“ほぼ同等”をまとめて間引き（任意） ----
def pareto_front_mask_chunked_eps(values: np.ndarray, rel_eps: float = 0.02,
                                  chunk_size: int = 10_000, verbose: bool = False) -> np.ndarray:
    vmin = np.nanmin(values, axis=0)
    vmax = np.nanmax(values, axis=0)
    span = np.maximum(vmax - vmin, 1e-12)
    norm = (values - vmin) / span
    eps = float(rel_eps)

    def _dominated_mask_eps(vals: np.ndarray) -> np.ndarray:
        n = vals.shape[0]
        dominated = np.zeros(n, dtype=bool)
        for i in range(n):
            if dominated[i]: continue
            v = vals[i]
            le = (vals <= (v + eps)).all(axis=1)
            lt = (vals <  (v + eps)).any(axis=1)
            d = le & lt
            d[i] = False
            dominated |= d
        return dominated

    n = norm.shape[0]
    idx_global = np.arange(n)
    pf_vals = None; pf_idx = None

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        block = norm[start:end]
        bidx  = idx_global[start:end]

        if pf_vals is not None and len(pf_vals):
            le = (pf_vals[:, None, :] <= (block[None, :, :] + eps)).all(axis=2)
            lt = (pf_vals[:, None, :] <  (block[None, :, :] + eps)).any(axis=2)
            dom = (le & lt).any(axis=0)
            block = block[~dom]; bidx = bidx[~dom]

        if block.size == 0:
            if verbose: print(f"[Pareto ε] processed {end}/{n} (block fully dominated)")
            continue

        if pf_vals is None:
            comb = block; cidx = bidx
        else:
            comb = np.vstack([pf_vals, block]); cidx = np.concatenate([pf_idx, bidx])

        d = _dominated_mask_eps(comb)
        keep = ~d
        pf_vals = comb[keep]; pf_idx = cidx[keep]
        if verbose:
            print(f"[Pareto ε] processed {end}/{n}, current frontier size = {len(pf_vals)}")

    mask = np.zeros(n, dtype=bool)
    if pf_idx is not None and len(pf_idx):
        mask[pf_idx] = True
    return mask

# ======== サブ空間・階層化 ========
def pareto_mask_on_subset(df_eval: pd.DataFrame, logicals: list, obj_dirs: dict,
                          chunk_size: int, verbose: bool) -> np.ndarray:
    cols = resolve_cols_from_logicals(df_eval, logicals)
    vals = to_minimization_matrix(df_eval, cols, obj_dirs)
    return pareto_front_mask_chunked(vals, chunk_size=chunk_size, verbose=verbose)

def lexicographic_front_mask(df_eval: pd.DataFrame,
                             primary_logicals: list, secondary_logicals: list,
                             obj_dirs: dict, chunk_size: int, verbose: bool) -> np.ndarray:
    # 1段目：primaryで前面
    mask1 = pareto_mask_on_subset(df_eval, primary_logicals, obj_dirs, chunk_size, verbose=False)
    idx1 = df_eval.index[mask1]

    if len(secondary_logicals) == 0:
        mask_final = np.zeros(len(df_eval), dtype=bool)
        mask_final[idx1] = True
        return mask_final

    # 2段目：候補上でsecondary前面
    df_cand = df_eval.loc[idx1].copy()
    mask2 = pareto_mask_on_subset(df_cand, secondary_logicals, obj_dirs, chunk_size, verbose=False)

    mask_final = np.zeros(len(df_eval), dtype=bool)
    mask_final[df_cand.index[mask2]] = True
    return mask_final

# ================= 実行本体 =================
def main():
    # 入力
    input_path = os.path.join(INPUT_FOLDER, INPUT_FILE)
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"予測結果が見つかりません: {input_path}")

    df = pd.read_excel(input_path)
    print(f"✅ データ読み込み: {df.shape}")
    print(f"   列名: {df.columns.tolist()[:10]}...")

    # 目的列の定義
    if not isinstance(PARETO_OBJECTIVES, dict):
        print("⚠ PARETO_OBJECTIVES が dict ではありません。全て min として扱います。")
        obj_dirs = {}
    else:
        obj_dirs = {k: str(v).lower() for k, v in PARETO_OBJECTIVES.items()}

    # 目的の実列解決
    eff_cols_resolved, logical_names, missing = [], [], []
    for name in obj_dirs.keys():
        try:
            col = resolve_column(df, name)
            eff_cols_resolved.append(col)
            logical_names.append(logical_name_from_resolved(col))
            print(f"  ✓ {name} → {col}")
        except KeyError:
            missing.append(name)
            print(f"  ✗ {name}: 列が見つかりません")

    if len(eff_cols_resolved) < 2:
        if missing:
            raise ValueError(f"パレート分析には少なくとも2つの目的が必要です。不足: {missing}")
        raise ValueError("パレート分析に使用できる目的変数が2つ未満です。")

    # 全min前提の安全ガード（必要に応じて外してください）
    assert all(v == "min" for v in obj_dirs.values()), "今回の実行は全min前提です。max指定が含まれています。"

    # 欠損/非有限の除外
    work = df.copy()
    for c in eff_cols_resolved:
        work[c] = pd.to_numeric(work[c], errors="coerce")
    work[eff_cols_resolved] = work[eff_cols_resolved].replace([np.inf, -np.inf], np.nan)
    df_eval = work.dropna(subset=eff_cols_resolved)
    if len(df_eval) == 0:
        raise ValueError("目的列に欠損/非有限値が多すぎて評価できません。")

    # 行数上限（任意）
    if PARETO_MAX_ROWS_FOR_STRICT is not None and len(df_eval) > int(PARETO_MAX_ROWS_FOR_STRICT):
        print(f"⚠ 行数が上限({int(PARETO_MAX_ROWS_FOR_STRICT):,})を超えています。先頭から上限件のみで判定します。")
        df_eval = df_eval.iloc[:int(PARETO_MAX_ROWS_FOR_STRICT)].copy()

    print(f"\n評価対象: {len(df_eval):,}件（欠損・非有限除外後）")

    # ---- 厳密フロンティア（保存用） ----
    vals_min = to_minimization_matrix(df_eval, eff_cols_resolved, obj_dirs)
    strict_mask = pareto_front_mask_chunked(vals_min, chunk_size=PARETO_CHUNK_SIZE, verbose=PARETO_VERBOSE)
    n_pf = int(strict_mask.sum())
    ratio = n_pf / len(df_eval)
    print(f"パレートフロンティア（厳密）: {n_pf:,} / {len(df_eval):,}  (比率={ratio:.3%})")

    # ---- 図用フロンティア（ε-支配で間引き可能） ----
    if PARETO_USE_EPSILON_PLOT:
        plot_mask = pareto_front_mask_chunked_eps(vals_min, rel_eps=PARETO_EPSILON_REL,
                                                  chunk_size=PARETO_CHUNK_SIZE, verbose=False)
        print(f"  └ 図用（ε={PARETO_EPSILON_REL:.3f}）: {int(plot_mask.sum()):,} 点")
    else:
        plot_mask = strict_mask

    # 出力フォルダ
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    plot_dir = os.path.join(OUTPUT_FOLDER, _PLOTS_FOLDER_NAME)
    os.makedirs(plot_dir, exist_ok=True)

    # ---- Excel保存（厳密＆図用マスクの両方） ----
    xlsx_path = os.path.join(OUTPUT_FOLDER, _XLSX_NAME)
    with pd.ExcelWriter(xlsx_path) as writer:
        df_eval.loc[df_eval.index[strict_mask]].to_excel(writer, sheet_name=_SHEET_PARETO_ONLY, index=False)
        df_eval.loc[df_eval.index[plot_mask]].to_excel(writer, sheet_name=_SHEET_PARETO_ONLY_PLOT, index=False)
        used_pairs = [(logical_names[i], eff_cols_resolved[i]) for i in range(len(eff_cols_resolved))]
        meta = pd.DataFrame({
            "objective": [name for name, _ in used_pairs],
            "direction": [obj_dirs.get(name, "min") for name, _ in used_pairs],
            "resolved_column_used": [col for _, col in used_pairs],
            "n_eval": [len(df_eval)]*len(used_pairs),
            "n_front_strict": [n_pf]*len(used_pairs),
            "front_ratio": [ratio]*len(used_pairs),
        })
        meta.to_excel(writer, sheet_name=_SHEET_META, index=False)
    print(f"✅ パレート結果を保存: {xlsx_path}")

    # ---- 可視化（4目的の2Dペア） ----
    if PARETO_ENABLE_PLOTTING and len(eff_cols_resolved) >= 2:
        df_out = df_eval.copy()
        df_out["is_pareto_plot"] = False
        df_out.loc[df_eval.index[plot_mask], "is_pareto_plot"] = True

        pairs = list(it.combinations(range(len(eff_cols_resolved)), 2))
        for (i, j) in pairs:
            x_logical, y_logical = logical_names[i], logical_names[j]
            x_dir = obj_dirs.get(x_logical, "min"); y_dir = obj_dirs.get(y_logical, "min")
            x_col, y_col = eff_cols_resolved[i], eff_cols_resolved[j]

            # 全点（有限のみ）
            x_vals_all = pd.to_numeric(df_out[x_col], errors="coerce")
            y_vals_all = pd.to_numeric(df_out[y_col], errors="coerce")
            mask_all = np.isfinite(x_vals_all) & np.isfinite(y_vals_all)
            n_all = int(mask_all.sum())

            # 図用フロント点（必要なら代表サンプル化）
            pareto_df = df_out[df_out["is_pareto_plot"] & mask_all].copy()
            if len(pareto_df) > PARETO_PLOT_MAX_POINTS:
                try:
                    from sklearn.cluster import KMeans
                    X = df_eval.loc[pareto_df.index, eff_cols_resolved].to_numpy(dtype=np.float32)
                    k = min(PARETO_PLOT_MAX_POINTS, len(X))
                    km = KMeans(n_clusters=k, n_init="auto", random_state=0).fit(X)
                    centers_idx = []
                    for ci in range(k):
                        m = np.where(km.labels_ == ci)[0]
                        if m.size == 0: continue
                        mX = X[m]; ctr = km.cluster_centers_[ci]
                        jmin = m[np.argmin(((mX - ctr) ** 2).sum(axis=1))]
                        centers_idx.append(df_eval.index[pareto_df.index[jmin]])
                    keep_idx = set(centers_idx)
                    pareto_df = pareto_df.loc[pareto_df.index.intersection(keep_idx)]
                except Exception:
                    pass  # k-meansが無ければそのまま

            n_pf_plot = len(pareto_df)

            if n_all == 0 and n_pf_plot == 0:
                print(f"⚠ プロットをスキップ: {x_logical} vs {y_logical}（有効データ0件）")
                continue

            fig, ax = plt.subplots(figsize=_PLOT_FIGSIZE)

            if n_all > 0:
                ax.scatter(x_vals_all[mask_all], y_vals_all[mask_all],
                           alpha=_PLOT_ALPHA_ALL, label="All", s=_PLOT_SIZE_ALL)
            if n_pf_plot > 0:
                ax.scatter(pareto_df[x_col], pareto_df[y_col],
                           label="Pareto front", s=_PLOT_SIZE_PARETO,
                           edgecolors=_PLOT_EDGECOLORS, facecolors=_PLOT_FACECOLORS,
                           linewidths=_PLOT_LINEWIDTHS)

            # 2Dペア専用前面（黒★）
            if PARETO_PLOT_SHOW_PAIRWISE_FRONT and n_all > 0:
                df_pair = df_out.loc[mask_all, [x_col, y_col]].copy()
                vals_pair = make_2d_values(df_out.loc[mask_all], x_col, y_col, x_dir, y_dir)
                mask_pair = pareto_2d_mask_min(vals_pair)
                ax.scatter(df_pair[x_col].to_numpy()[mask_pair],
                           df_pair[y_col].to_numpy()[mask_pair],
                           marker="*", s=40, label="2D pairwise front")

            ax.set_xlabel(pretty_axis_label(x_logical, x_dir))
            ax.set_ylabel(pretty_axis_label(y_logical, y_dir))
            ax.set_title(f"Pareto: {x_logical} vs {y_logical}  (front={n_pf_plot:,}/{n_all:,})")
            ax.grid(True, alpha=_PLOT_GRID_ALPHA)
            if ax.get_legend_handles_labels()[0]:
                ax.legend()

            fn = _PLOT_FN_FMT.format(x_logical=x_logical, y_logical=y_logical)
            save_path = os.path.join(plot_dir, fn)
            plt.tight_layout()
            plt.savefig(save_path, dpi=_PLOT_DPI)
            plt.close(fig)
            print(f"🖼 保存: {save_path}")

    # =========================
    # Pattern A: 4目的フル（上で保存済み）
    # =========================

    base_dir = Path(OUTPUT_FOLDER)

    # ===========================================
    # 階層化（Lexicographic）front：優先→残り（二段目）
    # ===========================================
    for case in LEXI_CASES:
        pri = list(case.get("primary", []))
        sec = list(case.get("secondary", []))
        tag = str(case.get("tag", "lexi"))
        print(f"\n[LEXI] primary={pri}, secondary={sec}")

        m_lexi = lexicographic_front_mask(df_eval, pri, sec, obj_dirs,
                                          chunk_size=PARETO_CHUNK_SIZE, verbose=False)
        df_lexi = df_eval.loc[df_eval.index[m_lexi]].copy()
        out_dir = base_dir / f"pattern_lexi_{tag}"
        out_dir.mkdir(parents=True, exist_ok=True)
        xlsx_path2 = out_dir / f"lexi_{tag}.xlsx"
        with pd.ExcelWriter(xlsx_path2) as writer:
            df_lexi.to_excel(writer, sheet_name="lexi_front", index=False)
            pd.DataFrame({"primary": [", ".join(pri)], "secondary": [", ".join(sec)]}).to_excel(
                writer, sheet_name="meta", index=False
            )
        print(f"✅ [LEXI] 保存: {xlsx_path2}")

    # ===========================================================
    # Pattern B: 4項目中3項目→2Dフロント
    #   - B1: 「切削時間」「摩耗量」必須 + 残り1つ → (切削時間, 摩耗量) の2Dフロント
    #   - B2: 「摩耗量」必須 + 残り2つ選択 → (摩耗量, X) の2Dフロント（重複なし）
    # ===========================================================
    ln = logical_names  # 例: ["摩耗量","切削時間","上面ダレ量","側面ダレ量"]

    # ----- B1 -----
    must = ["切削時間", "摩耗量"]
    others = [x for x in ln if x not in must]
    out_dir_b1 = base_dir / "pattern_B1_3to2_must_切削時間_摩耗量"
    out_dir_b1.mkdir(parents=True, exist_ok=True)

    for x in others:
        triplet = must + [x]
        print(f"\n[Pattern B1] triplet={triplet} -> 2D ({must[0]}, {must[1]})")
        mask3 = pareto_mask_on_subset(df_eval, triplet, obj_dirs, PARETO_CHUNK_SIZE, False)
        cand = df_eval.loc[df_eval.index[mask3]]

        cols = resolve_cols_from_logicals(df_eval, must)
        x_col, y_col = cols[0], cols[1]
        x_dir = obj_dirs.get(must[0], "min"); y_dir = obj_dirs.get(must[1], "min")
        v2 = make_2d_values(cand, x_col, y_col, x_dir, y_dir)
        pm = pareto_2d_mask_min(v2)
        front = cand.iloc[np.where(pm)[0]].copy()

        xlsx_b1 = out_dir_b1 / f"triplet_{triplet[2]}__2D_{must[0]}_{must[1]}.xlsx"
        with pd.ExcelWriter(xlsx_b1) as writer:
            front.to_excel(writer, sheet_name="front_2D", index=False)
            pd.DataFrame({
                "triplet": [", ".join(triplet)],
                "pair2D":  [f"{must[0]} & {must[1]}"]
            }).to_excel(writer, sheet_name="meta", index=False)
        print(f"✅ [B1] 保存: {xlsx_b1}")

        # --- 図の保存 ---
        png_b1 = out_dir_b1 / f"triplet_{triplet[2]}__2D_{must[0]}_{must[1]}.png"
        plot_2d_front(cand, front, x_col, y_col, must[0], must[1], x_dir, y_dir, str(png_b1))

    # ----- B2 -----
    must = ["摩耗量"]
    others = [x for x in ln if x not in must]
    out_dir_b2 = base_dir / "pattern_B2_3to2_must_摩耗量"
    out_dir_b2.mkdir(parents=True, exist_ok=True)

    for x, y in it.combinations(others, 2):
        triplet = must + [x, y]
        for pair in [(must[0], x), (must[0], y)]:
            pair_name = f"{pair[0]}_{pair[1]}"
            print(f"\n[Pattern B2] triplet={triplet} -> 2D ({pair[0]}, {pair[1]})")
            mask3 = pareto_mask_on_subset(df_eval, triplet, obj_dirs, PARETO_CHUNK_SIZE, False)
            cand = df_eval.loc[df_eval.index[mask3]]

            cols = resolve_cols_from_logicals(df_eval, list(pair))
            x_col, y_col = cols[0], cols[1]
            x_dir = obj_dirs.get(pair[0], "min"); y_dir = obj_dirs.get(pair[1], "min")
            v2 = make_2d_values(cand, x_col, y_col, x_dir, y_dir)
            pm = pareto_2d_mask_min(v2)
            front = cand.iloc[np.where(pm)[0]].copy()

            xlsx_b2 = out_dir_b2 / f"triplet_{x}_{y}__2D_{pair_name}.xlsx"
            with pd.ExcelWriter(xlsx_b2) as writer:
                front.to_excel(writer, sheet_name="front_2D", index=False)
                pd.DataFrame({
                    "triplet": [", ".join(triplet)],
                    "pair2D":  [f"{pair[0]} & {pair[1]}"]
                }).to_excel(writer, sheet_name="meta", index=False)
            print(f"✅ [B2] 保存: {xlsx_b2}")

            # --- 図の保存 ---
            png_b2 = out_dir_b2 / f"triplet_{x}_{y}__2D_{pair_name}.png"
            plot_2d_front(cand, front, x_col, y_col, pair[0], pair[1], x_dir, y_dir, str(png_b2))

    # ==========================================
    # Pattern C: 4項目中2項目→2Dフロント
    #   - C1: 「切削時間」必須 + 残りすべてとペア
    #   - C2: 「摩耗量」必須 + 残りすべてとペア（C1と重複なし）
    # ==========================================
    out_dir_c = base_dir / "pattern_C_2to2_pairs"
    out_dir_c.mkdir(parents=True, exist_ok=True)

    used_pairs = set()
    # C1
    must = "切削時間"
    for x in [n for n in ln if n != must]:
        pair = tuple(sorted([must, x]))
        if pair in used_pairs: continue
        used_pairs.add(pair)

        print(f"\n[Pattern C1] 2D ({pair[0]}, {pair[1]})")
        cols = resolve_cols_from_logicals(df_eval, list(pair))
        x_col, y_col = cols[0], cols[1]
        x_dir = obj_dirs.get(pair[0], "min"); y_dir = obj_dirs.get(pair[1], "min")
        v2 = make_2d_values(df_eval, x_col, y_col, x_dir, y_dir)
        pm = pareto_2d_mask_min(v2)
        front = df_eval.iloc[np.where(pm)[0]].copy()

        xlsx_c1 = out_dir_c / f"2D_{pair[0]}_{pair[1]}.xlsx"
        with pd.ExcelWriter(xlsx_c1) as writer:
            front.to_excel(writer, sheet_name="front_2D", index=False)
            pd.DataFrame({"pair2D": [f"{pair[0]} & {pair[1]}"]}).to_excel(writer, sheet_name="meta", index=False)
        print(f"✅ [C1] 保存: {xlsx_c1}")

        # --- 図の保存 ---
        png_c1 = out_dir_c / f"2D_{pair[0]}_{pair[1]}.png"
        plot_2d_front(df_eval, front, x_col, y_col, pair[0], pair[1], x_dir, y_dir, str(png_c1))

    # C2
    must = "摩耗量"
    for x in [n for n in ln if n != must]:
        pair = tuple(sorted([must, x]))
        if pair in used_pairs: continue
        used_pairs.add(pair)

        print(f"\n[Pattern C2] 2D ({pair[0]}, {pair[1]})")
        cols = resolve_cols_from_logicals(df_eval, list(pair))
        x_col, y_col = cols[0], cols[1]
        x_dir = obj_dirs.get(pair[0], "min"); y_dir = obj_dirs.get(pair[1], "min")
        v2 = make_2d_values(df_eval, x_col, y_col, x_dir, y_dir)
        pm = pareto_2d_mask_min(v2)
        front = df_eval.iloc[np.where(pm)[0]].copy()

        xlsx_c2 = out_dir_c / f"2D_{pair[0]}_{pair[1]}.xlsx"
        with pd.ExcelWriter(xlsx_c2) as writer:
            front.to_excel(writer, sheet_name="front_2D", index=False)
            pd.DataFrame({"pair2D": [f"{pair[0]} & {pair[1]}"]}).to_excel(writer, sheet_name="meta", index=False)
        print(f"✅ [C2] 保存: {xlsx_c2}")

        # --- 図の保存 ---
        png_c2 = out_dir_c / f"2D_{pair[0]}_{pair[1]}.png"
        plot_2d_front(df_eval, front, x_col, y_col, pair[0], pair[1], x_dir, y_dir, str(png_c2))

    print("🎯 完了：パレート分析（階層化 + 3パターン一括＋図生成）を作成しました。")

# ===== __pycache__ を 99_Temp へ移動（従来仕様を継承） =====
def move_pycache_to_temp():
    temp_folder = Path("99_Temp")
    temp_folder.mkdir(exist_ok=True)
    pycache_folders = glob.glob("**/__pycache__", recursive=True)
    for pycache_path in pycache_folders:
        p = Path(pycache_path)
        if p.exists() and p.is_dir():
            try:
                relative_path = p.relative_to(Path.cwd())
                temp_dest = temp_folder / f"{relative_path.parent.name}__pycache__"
            except ValueError:
                temp_dest = temp_folder / f"{p.parent.name}__pycache__"
            if temp_dest.exists():
                shutil.rmtree(temp_dest)
            shutil.move(str(p), str(temp_dest))
            print(f"✅ {p} → {temp_dest}")

if __name__ == "__main__":
    move_pycache_to_temp()
    main()

