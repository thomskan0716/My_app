"""
特徴量タイプを考慮したデータ拡張モジュール（安定化・バグ修正）
- 未使用インポート削除
- PPM_LEVELS が空/未設定の場合は安全にスキップ
- augment_ratio を [0,1] にクリップ
- 乱数は局所 RNG（np.random.default_rng）で再現性と副作用なし
- existing_groups があれば元サンプルのグループを拡張データへ継承
- 出力Xは float32 に統一
"""
import numpy as np
import pandas as pd
from typing import Optional

class FeatureAwareAugmentor:
    """特徴量タイプを考慮したPPMノイズ拡張"""

    def __init__(self, config):
        self.config = config
        self.ppm_levels = getattr(config, 'PPM_LEVELS', None)
        # 0〜1 にクリップ（負値/1超えの事故防止）
        raw_ratio = getattr(config, 'AUGMENT_RATIO', 0.2)
        self.augment_ratio = float(np.clip(raw_ratio, 0.0, 1.0))

    def _add_noise_to_continuous(self, X, ppm_level, seed=None):
        """連続変数のみにノイズを追加"""
        if seed is not None:
            np.random.seed(seed)
            
        X_noisy = X.copy()
        
        for col in self.config.CONTINUOUS_FEATURES:
            if col in X_noisy.columns:
                # ★ Saneo: Asegurar copia propia y float64 para evitar vistas compartidas
                # Evita que el GC/hilos encuentren vistas compartidas medio muertas
                col_values = X_noisy[col].values
                x = np.asarray(col_values, dtype=np.float64).copy(order="C")
                
                # Calcular ruido
                noise = np.random.randn(x.size) * (ppm_level / 1e6) * x
                x += noise
                
                # Asignar de vuelta
                X_noisy[col] = x
                
        return X_noisy

    def augment(self, X: pd.DataFrame, y: np.ndarray, existing_groups: Optional[np.ndarray] = None):
        """データ拡張"""
        if not self.config.USE_PPM_AUGMENTATION:
            groups = np.asarray(existing_groups) if existing_groups is not None else np.arange(len(X))
            return X, y, groups
    
        X = X.reset_index(drop=True)
        y = np.asarray(y)
        n_original = len(X)
        
        # 生成するデータ数を計算
        n_to_generate = int(n_original * self.augment_ratio)
        if hasattr(self.config, 'SHOW_AUGMENTATION_LOGS') and self.config.SHOW_AUGMENTATION_LOGS:
            print(f"生成するデータ数 :   {n_to_generate}")
        
        if n_to_generate == 0:
            groups = np.asarray(existing_groups) if existing_groups is not None else np.arange(len(X))
            return X, y, groups
        
        # データ生成（サンプルごとに1個ずつ生成）
        X_list = [X]  # 元データ
        y_list = [y]
        
        # n_to_generate個のサンプルを生成
        ppm_cycle = np.tile(self.ppm_levels, n_to_generate // len(self.ppm_levels) + 1)[:n_to_generate]
        
        for i in range(n_to_generate):
            # どの元サンプルから生成するか
            source_idx = i % n_original
            X_single = X.iloc[[source_idx]]
            
            # ノイズ追加
            ppm = ppm_cycle[i]
            seed = (self.config.RANDOM_STATE or 0) + i
            X_noisy = self._add_noise_to_continuous(X_single, ppm, seed)
            
            X_list.append(X_noisy)
            y_list.append(y[[source_idx]])
        
        # 結合
        X_augmented = pd.concat(X_list, ignore_index=True)
        y_augmented = np.concatenate(y_list)
        
        # グループ作成（正しい方法）
        # 元データ: 0,1,2,...,429
        # 拡張データ: 最初の86個は 0,1,2,...,85,0,1,2...
        groups = np.arange(n_original)  # 元データのグループ
        aug_groups = np.arange(n_to_generate) % n_original  # 拡張データのグループ
        groups = np.concatenate([groups, aug_groups])
        
        # ログ（拡張ログ表示が有効な場合のみ）
        if hasattr(self.config, 'SHOW_AUGMENTATION_LOGS') and self.config.SHOW_AUGMENTATION_LOGS:
            print(f"\n拡張結果:")
            print(f"  元データ: {n_original} samples")
            print(f"  拡張後: {len(X_augmented)} samples")
            print(f"  拡張率: {len(X_augmented)/n_original:.1f}倍")
            print(f"  設定比率: {self.augment_ratio} ({self.augment_ratio*100:.0f}%)")
        
        return X_augmented, y_augmented, groups
