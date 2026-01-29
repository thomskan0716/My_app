"""PPMノイズ拡張モジュール"""
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple

class PPMNoiseAugmentor:
    def __init__(self, ppm_levels: List[int] = None, 
                 n_augment_per_level: int = 2,
                 random_state: int = None,
                 continuous_feature_names: List[str] = None):  # ★追加パラメータ
        """
        PPMノイズベースのデータ拡張
        
        Args:
            ppm_levels: PPMレベルのリスト
            n_augment_per_level: 各レベルでの拡張回数
            random_state: 乱数シード
            continuous_feature_names: 連続値特徴量の名前リスト
        """
        self.ppm_levels = ppm_levels if ppm_levels else [10, 50, 100]
        self.n_augment_per_level = n_augment_per_level
        self.random_state = random_state
        self.continuous_feature_names = continuous_feature_names or []
    
    def add_ppm_noise(self, X: pd.DataFrame, ppm_level: int, seed: int = None) -> pd.DataFrame:
        """
        指定PPMレベルでノイズを追加（feature_aware_augmentor.pyで使用）
        
        Args:
            X: 入力DataFrame
            ppm_level: PPMレベル
            seed: 乱数シード
            
        Returns:
            ノイズが追加されたDataFrame
        """
        if seed is not None:
            np.random.seed(seed)
            
        X_noisy = X.copy()
        
        # 連続値特徴量が指定されている場合はそれを使用
        if self.continuous_feature_names:
            numeric_cols = [col for col in self.continuous_feature_names if col in X.columns]
        else:
            # 指定がない場合は数値列を自動検出
            numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in numeric_cols:
            values = X[col].values
            # PPMベースのノイズスケール計算
            noise_scale = np.abs(values) * ppm_level / 1e6
            # ゼロ値の場合の処理
            noise_scale = np.where(noise_scale == 0, 
                                  np.abs(values.mean()) * ppm_level / 1e6, 
                                  noise_scale)
            noise = np.random.normal(0, noise_scale)
            X_noisy[col] = values + noise
        
        return X_noisy
    
    def augment(self, X: pd.DataFrame, y: np.ndarray, 
                existing_groups: Optional[np.ndarray] = None) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """
        データセット全体の拡張（元のメソッド）
        
        Args:
            X: 特徴量DataFrame
            y: ターゲット配列
            existing_groups: 既存のグループ配列
            
        Returns:
            X_augmented: 拡張された特徴量
            y_augmented: 拡張されたターゲット
            groups: グループ配列
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
            
        X_list = [X.copy()]
        y_list = [y.copy()]
        
        aug_count = 0
        for ppm in self.ppm_levels:
            for i in range(self.n_augment_per_level):
                # add_ppm_noiseメソッドを使用
                seed = (self.random_state or 0) + aug_count if self.random_state else None
                X_noisy = self.add_ppm_noise(X, ppm, seed=seed)
                
                X_list.append(X_noisy)
                y_list.append(y.copy())
                aug_count += 1
        
        X_augmented = pd.concat(X_list, ignore_index=True)
        y_augmented = np.concatenate(y_list)
        
        # グループ作成
        n_copies = len(X_list)
        if existing_groups is not None:
            groups = np.tile(existing_groups, n_copies)
        else:
            groups = np.repeat(np.arange(len(X)), n_copies)
        
        return X_augmented, y_augmented, groups
    
    def augment_with_ppm_noise(self, X: pd.DataFrame, y: np.ndarray, 
                               ppm_level: int, config) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        指定PPMレベルでデータ拡張（互換性のため）
        
        Args:
            X: 特徴量DataFrame
            y: ターゲット配列
            ppm_level: PPMレベル
            config: 設定オブジェクト
            
        Returns:
            X_augmented: ノイズ追加された特徴量
            y_augmented: コピーされたターゲット
        """
        seed = config.RANDOM_STATE if hasattr(config, 'RANDOM_STATE') else None
        X_augmented = self.add_ppm_noise(X, ppm_level, seed=seed)
        y_augmented = y.copy()
        
        return X_augmented, y_augmented