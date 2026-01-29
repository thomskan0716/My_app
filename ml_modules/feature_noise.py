"""
特徴量にノイズを付与するモジュール（改善版）
- 追加されたノイズ列の詳細情報を返す
- ログ出力機能を追加
"""
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any


def add_noise_columns(
    X: pd.DataFrame,
    mode: str = "gaussian",
    ratio: float = 0.1,
    gaussian_std_factor: float = 0.1,
    random_state: int = 42,
    verbose: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    特徴量にノイズ列を追加
    
    Parameters:
    -----------
    X : pd.DataFrame
        入力特徴量
    mode : str
        ノイズモード ("gaussian" or "permute")
    ratio : float
        追加するノイズ列の割合
    gaussian_std_factor : float
        ガウシアンノイズの標準偏差係数
    random_state : int
        乱数シード
    verbose : bool
        詳細情報を出力するか
    
    Returns:
    --------
    X_with_noise : pd.DataFrame
        ノイズ列が追加されたデータ
    noise_info : Dict[str, Any]
        ノイズ追加の詳細情報
    """
    
    rng = np.random.RandomState(random_state)
    X_copy = X.copy()
    
    # 元の列数
    original_cols = X.shape[1]
    
    # 追加するノイズ列数
    n_noise_cols = max(1, int(original_cols * ratio))
    
    # 数値列のみを対象
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) == 0:
        if verbose:
            print("[WARNING] 数値列が存在しないため、ノイズ追加をスキップ")
        return X_copy, {"added_cols": 0, "noise_columns": []}
    
    noise_columns = []
    
    if mode == "gaussian":
        # ガウシアンノイズの追加
        for i in range(n_noise_cols):
            # ランダムに元の列を選択
            base_col = rng.choice(numeric_cols)
            base_data = X[base_col].values
            
            # 標準偏差を計算
            std = np.nanstd(base_data)
            if std == 0:
                std = 1.0
            
            # ガウシアンノイズを生成
            noise = rng.normal(0, std * gaussian_std_factor, size=len(base_data))
            noise_col_name = f"noise_gaussian_{i}"
            X_copy[noise_col_name] = base_data + noise
            noise_columns.append(noise_col_name)
            
    elif mode == "permute":
        # 順列ノイズの追加
        for i in range(n_noise_cols):
            # ランダムに元の列を選択
            base_col = rng.choice(numeric_cols)
            base_data = X[base_col].values.copy()
            
            # シャッフル
            rng.shuffle(base_data)
            noise_col_name = f"noise_permute_{i}"
            X_copy[noise_col_name] = base_data
            noise_columns.append(noise_col_name)
    
    else:
        raise ValueError(f"不明なノイズモード: {mode}")
    
    # 詳細情報の作成
    noise_info = {
        "original_cols": original_cols,
        "added_cols": n_noise_cols,
        "total_cols": X_copy.shape[1],
        "noise_columns": noise_columns,
        "mode": mode,
        "ratio": ratio,
        "numeric_cols_used": len(numeric_cols)
    }
    
    if verbose:
        print(f"[ノイズ付与] モード: {mode}")
        print(f"  元の列数: {original_cols}")
        print(f"  追加列数: {n_noise_cols} (比率: {ratio:.1%})")
        print(f"  合計列数: {X_copy.shape[1]}")
        if mode == "gaussian":
            print(f"  ガウシアンノイズ std係数: {gaussian_std_factor}")
    
    return X_copy, noise_info


# 後方互換性のためのラッパー関数
def add_noise_columns_legacy(
    X: pd.DataFrame,
    mode: str = "gaussian",
    ratio: float = 0.1,
    gaussian_std_factor: float = 0.1,
    random_state: int = 42
) -> pd.DataFrame:
    """
    後方互換性のための旧インターフェース
    （情報を返さずDataFrameのみ返す）
    """
    X_with_noise, _ = add_noise_columns(
        X, mode, ratio, gaussian_std_factor, random_state, verbose=False
    )
    return X_with_noise