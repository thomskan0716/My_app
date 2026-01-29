"""
正しいノイズ付与モジュール（縦方向：Data Augmentation）
連続数値特徴量にppmレベルのノイズを加えてサンプルを増やす
"""
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List, Optional
from config_cls import ConfigCLS


def add_noise_augmentation(
    X: pd.DataFrame,
    y: np.ndarray,
    noise_ppm: float = 100.0,
    augment_ratio: float = 0.1,
    continuous_features: Optional[List[str]] = None,
    random_state: int = 42,
    verbose: bool = True
) -> Tuple[pd.DataFrame, np.ndarray, Dict[str, Any]]:
    """
    連続数値特徴量にppmレベルのノイズを加えたサンプルを追加（縦方向）
    
    Parameters:
    -----------
    X : pd.DataFrame
        元の特徴量 (n_samples, n_features)
    y : np.ndarray
        ラベル
    noise_ppm : float
        ノイズレベル（parts per million）
        例: 100ppm = 0.01%の変動
    augment_ratio : float
        追加するサンプルの割合（0.1 = 10%追加）
    continuous_features : List[str], optional
        ノイズを付与する連続値列のリスト
        Noneの場合はConfigCLSから取得または自動検出
    random_state : int
        乱数シード
    verbose : bool
        詳細情報を出力
        
    Returns:
    --------
    X_augmented : pd.DataFrame
        ノイズ付与されたサンプルが追加されたデータ
    y_augmented : np.ndarray
        対応するラベル
    augment_info : Dict[str, Any]
        拡張の詳細情報
    """
    
    np.random.seed(random_state)
    
    # 元のサイズ
    n_original = len(X)
    
    # 追加するサンプル数
    n_augment = int(n_original * augment_ratio)
    
    if n_augment == 0:
        if verbose:
            print("[警告] augment_ratio が小さすぎて追加サンプル数が0")
        return X.copy(), y.copy(), {"added_samples": 0}
    
    # 連続値列の特定
    if continuous_features is None:
        # ConfigCLSから取得を試みる
        if hasattr(ConfigCLS, 'CONTINUOUS_FEATURES'):
            continuous_features = [col for col in ConfigCLS.CONTINUOUS_FEATURES if col in X.columns]
        else:
            # 自動検出（数値型かつユニーク値が多い列）
            continuous_features = []
            for col in X.select_dtypes(include=[np.number]).columns:
                if X[col].nunique() > 10:  # 離散値でない
                    continuous_features.append(col)
    
    if not continuous_features:
        if verbose:
            print("[警告] 連続値特徴量が見つからない")
        return X.copy(), y.copy(), {"added_samples": 0}
    
    # 拡張するサンプルをランダムに選択（層化サンプリング）
    if len(np.unique(y)) == 2:  # 二値分類
        # クラスバランスを保持
        indices_to_augment = []
        for class_label in np.unique(y):
            class_indices = np.where(y == class_label)[0]
            n_class_augment = int(len(class_indices) * augment_ratio)
            if n_class_augment > 0:
                selected = np.random.choice(class_indices, n_class_augment, replace=True)
                indices_to_augment.extend(selected)
        indices_to_augment = np.array(indices_to_augment)
    else:
        # ランダムサンプリング
        indices_to_augment = np.random.choice(n_original, n_augment, replace=True)
    
    # 選択されたサンプルをコピー
    X_to_augment = X.iloc[indices_to_augment].copy()
    y_to_augment = y[indices_to_augment].copy()
    
    # 連続値特徴量にノイズを付与
    noise_stats = {}
    for col in continuous_features:
        values = X_to_augment[col].values
        
        # 各値に対してppmレベルのノイズを生成
        # ノイズ = 値 × (ppm / 1,000,000) × 正規分布
        noise_std = np.abs(values) * (noise_ppm / 1e6)
        noise = np.random.normal(0, noise_std, len(values))
        
        # ノイズを付与
        X_to_augment[col] = values + noise
        
        # 統計情報を記録
        noise_stats[col] = {
            'mean_noise': np.mean(np.abs(noise)),
            'max_noise': np.max(np.abs(noise)),
            'relative_noise': np.mean(np.abs(noise) / (np.abs(values) + 1e-10))
        }
    
    # 元データと結合
    X_augmented = pd.concat([X, X_to_augment], ignore_index=True)
    y_augmented = np.concatenate([y, y_to_augment])
    
    # 詳細情報
    augment_info = {
        'original_samples': n_original,
        'added_samples': n_augment,
        'total_samples': len(X_augmented),
        'augment_ratio': augment_ratio,
        'noise_ppm': noise_ppm,
        'continuous_features_used': continuous_features,
        'noise_statistics': noise_stats
    }
    
    if verbose:
        print(f"[Data Augmentation] ノイズ付与（縦方向）")
        print(f"  元のサンプル数: {n_original}")
        print(f"  追加サンプル数: {n_augment} (比率: {augment_ratio:.1%})")
        print(f"  合計サンプル数: {len(X_augmented)}")
        print(f"  ノイズレベル: {noise_ppm} ppm")
        print(f"  対象特徴量数: {len(continuous_features)}")
        
        # クラスバランスの確認
        if len(np.unique(y)) == 2:
            orig_balance = np.mean(y == 1)
            aug_balance = np.mean(y_augmented == 1)
            print(f"  Positiveクラス比率: {orig_balance:.1%} → {aug_balance:.1%}")
    
    return X_augmented, y_augmented, augment_info


def validate_augmentation(
    X_original: pd.DataFrame,
    X_augmented: pd.DataFrame,
    augment_info: Dict[str, Any]
) -> bool:
    """
    拡張データの妥当性を検証
    
    Returns:
    --------
    is_valid : bool
        検証に合格したか
    """
    
    # 列数が同じか確認
    if X_original.shape[1] != X_augmented.shape[1]:
        print("[エラー] 列数が変更されています（横方向の変更は不正）")
        return False
    
    # 列名が同じか確認
    if list(X_original.columns) != list(X_augmented.columns):
        print("[エラー] 列名が変更されています")
        return False
    
    # サンプル数が増えているか確認
    expected_rows = augment_info['original_samples'] + augment_info['added_samples']
    if len(X_augmented) != expected_rows:
        print("[エラー] サンプル数が期待値と一致しません")
        return False
    
    print("[検証OK] 正しく縦方向にデータが拡張されています")
    return True