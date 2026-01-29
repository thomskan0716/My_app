
# models/base_model.py
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseModel(ABC):
    """
    各モデル実装の共通インターフェース。

    - ModelFactory からインスタンス化される前提
    - build(**params) で実体(self.model)を構築
    - fit/predict は sklearn 互換インターフェースへ委譲
    """

    def __init__(self) -> None:
        # 実体（sklearn/LightGBM/XGBoost/CatBoost 等の学習器）が入る
        self.model: Any = None
        # 表示/識別用のモデル名（派生クラスで上書き可）
        self.name: str = getattr(self, "name", self.__class__.__name__.lower())
        # 外部ライブラリ依存モデル(CatBoost等)の可用フラグ（派生側で設定）
        self.available: bool = getattr(self, "available", True)

    def get_name(self) -> str:
        """モデルの表示名/キー名"""
        return getattr(self, "name", self.__class__.__name__.lower())

    @property
    def is_built(self) -> bool:
        """モデルが build() 済みかどうか"""
        return self.model is not None

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__} "
            f"name={self.get_name()} built={self.is_built} available={self.available}>"
        )

    # ===== 抽象I/F（派生クラスで実装） =====
    @abstractmethod
    def suggest_hyperparameters(self, trial: Any) -> Dict[str, Any]:
        """
        Optuna の Trial を受け取り、ハイパーパラメータ候補を dict で返す。
        使用しないモデルでも空 dict を返せばよい。
        """
        raise NotImplementedError

    @abstractmethod
    def build(self, **params: Any) -> Any:
        """
        実際の学習器(Estimator)を構築し self.model にセットして返す。
        ここでパラメータの正規化（int化など）も行う。
        """
        raise NotImplementedError

    # ===== 既定実装（任意で派生側が上書き可） =====
    def get_model_type(self) -> str:
        """
        SHAP 等の分岐で利用するモデルタイプのヒント。
        例: 'tree' | 'linear' | 'gaussian_process' | 'unknown'
        """
        return "unknown"

    def fit(self, X: Any, y: Any, **kwargs: Any) -> Any:
        """
        学習器の学習。派生側が未実装でも sklearn 互換 Estimator に委譲。
        戻り値は慣例として self.model を返す（sklearn 互換）。
        """
        if self.model is None:
            self.build()
        return self.model.fit(X, y, **kwargs)

    def predict(self, X: Any, **kwargs: Any) -> Any:
        """
        学習器の予測。派生側が未実装でも sklearn 互換 Estimator に委譲。
        戻り値は通常 numpy.ndarray を想定（型は Any として緩く表現）。
        """
        if self.model is None:
            raise RuntimeError(
                f"{self.get_name()} is not built. Call build() then fit() first."
            )
        return self.model.predict(X, **kwargs)
