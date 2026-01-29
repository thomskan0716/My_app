# ood_mahalanobis.py
"""
Mahalanobis 距離に基づく簡易 OOD ゲート
- 標準化後の特徴空間で Empirical Covariance を使って距離（二乗）を計算
- しきい値は学習データの距離分布パーセンタイルで決定
"""
from __future__ import annotations

from typing import Optional
import numpy as np
from sklearn.covariance import EmpiricalCovariance


class MahalanobisGate:
    def __init__(self) -> None:
        self.mean_: Optional[np.ndarray] = None       # shape (1, n_features)
        self.cov_: EmpiricalCovariance = EmpiricalCovariance()
        self.threshold_: Optional[float] = None
        self.n_features_: Optional[int] = None

    def _check_2d(self, X: np.ndarray, name: str) -> np.ndarray:
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"{name} は2次元配列である必要があります。got shape={X.shape}")
        return X

    def fit(self, X: np.ndarray, percentile: float = 99.0) -> "MahalanobisGate":
        """
        X : (n_samples, n_features) すでにスケーリング済みの特徴行列
        percentile : 距離（二乗）の上位パーセンタイルを OOD しきい値に設定
        """
        X = self._check_2d(X, "X")
        if not (0.0 < percentile < 100.0):
            raise ValueError(f"percentile は (0,100) で指定してください。got {percentile}")

        # 学習
        self.cov_.fit(X)
        self.mean_ = np.mean(X, axis=0, keepdims=True)     # (1, d)
        self.n_features_ = X.shape[1]

        # EmpiricalCovariance.mahalanobis は「Mahalanobis距離の二乗」を返す
        d2 = self.cov_.mahalanobis(X - self.mean_)
        self.threshold_ = float(np.percentile(d2, percentile))
        return self

    def _ensure_fitted(self) -> None:
        if self.mean_ is None or self.n_features_ is None or self.threshold_ is None:
            raise RuntimeError("MahalanobisGate は未学習です。まず fit(X) を呼んでください。")

    def score(self, X: np.ndarray) -> np.ndarray:
        # 予測時の次元不一致を即座に検出
        if X.shape[1] != self.mean_.shape[1]:
            raise ValueError(
                f"Mahalanobis: 次元不一致 X:{X.shape[1]} vs mean_:{self.mean_.shape[1]}"
            )
        d2 = self.cov_.mahalanobis(X - self.mean_)
        return d2


    def is_ood(self, X: np.ndarray) -> np.ndarray:
        """
        しきい値以上（大きいほど遠い）を OOD=1 とするフラグを返す。
        """
        d2 = self.score(X)
        return (d2 >= float(self.threshold_)).astype(int)
