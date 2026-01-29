# models/gaussian_process_model.py
"""
ガウス過程回帰モデル（GaussianProcessRegressor）
- Optuna の gp_* パラメータを正式キーへ正規化
- n_features 未確定でも落ちない length_scale 初期化
- 整数/型の厳密化
- fit 時に次元を確定し、必要なら安全に再 build
"""

from __future__ import annotations
import numpy as np

try:
    from .base_model import BaseModel
except Exception:
    from base_model import BaseModel  # 単体配布時のフォールバック

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel,
    DotProduct,
    RBF,
    Matern,
    WhiteKernel,
)


class GaussianProcessModel(BaseModel):
    """ガウス過程回帰モデル"""

    def __init__(self) -> None:
        self.model = None
        self.name = "gaussian_process"
        self.available = True
        self.n_features: int | None = None
        # ユーザが直前の kernel_type を見たい場合に備え保存
        self._last_kernel_type: str = "rbf_only"

    # ---------------- Optuna param space ----------------
    def suggest_hyperparameters(self, trial):
        """
        Optuna 用のハイパーパラメータ提案。
        ※ 'rbf_const_with_dot' を使いたい場合はリストのコメントを外す。
        """
        return {
            "gp_alpha": trial.suggest_float("gp_alpha", 1e-6, 1e-1, log=True),
            "gp_n_restarts_optimizer": trial.suggest_int("gp_n_restarts_optimizer", 0, 10),
            "gp_normalize_y": trial.suggest_categorical("gp_normalize_y", [True, False]),
            "gp_kernel_type": trial.suggest_categorical(
                "gp_kernel_type",
                [
                    "rbf_only",
                    "rbf_with_dot",
                    "rbf_with_const",
                    # "rbf_const_with_dot",
                    "matern_15_only",
                    "matern_15_with_dot",
                    "matern_05_only",
                    "matern_05_with_dot",
                    "matern_25_only",
                    "matern_25_with_dot",
                    "dot_only",
                ],
            ),
        }

    # ---------------- Kernel factory ----------------
    def _get_kernel(self, kernel_type: str, n_features: int | None):
        """
        n_features が未確定(None)でも落ちないように、length_scale を
        ベクトルではなくスカラーで初期化可能にする。
        """
        has_dim = (n_features is not None) and (n_features > 0)
        # RBF/Matern にベクトル長スケールを与えたい場合は np.ones(n_features)、
        # 未確定時はスカラー 1.0 を使う。
        one = np.ones(n_features) if has_dim else 1.0

        if kernel_type == "rbf_only":
            return ConstantKernel() * RBF() + WhiteKernel()

        elif kernel_type == "rbf_with_dot":
            return ConstantKernel() * RBF() + WhiteKernel() + ConstantKernel() * DotProduct()

        elif kernel_type == "rbf_with_const":
            return ConstantKernel() * RBF(one) + WhiteKernel()

        elif kernel_type == "rbf_const_with_dot":
            return ConstantKernel() * RBF(one) + WhiteKernel() + ConstantKernel() * DotProduct()

        elif kernel_type == "matern_15_only":
            return ConstantKernel() * Matern(nu=1.5) + WhiteKernel()

        elif kernel_type == "matern_15_with_dot":
            return ConstantKernel() * Matern(nu=1.5) + WhiteKernel() + ConstantKernel() * DotProduct()

        elif kernel_type == "matern_05_only":
            return ConstantKernel() * Matern(nu=0.5) + WhiteKernel()

        elif kernel_type == "matern_05_with_dot":
            return ConstantKernel() * Matern(nu=0.5) + WhiteKernel() + ConstantKernel() * DotProduct()

        elif kernel_type == "matern_25_only":
            return ConstantKernel() * Matern(nu=2.5) + WhiteKernel()

        elif kernel_type == "matern_25_with_dot":
            return ConstantKernel() * Matern(nu=2.5) + WhiteKernel() + ConstantKernel() * DotProduct()

        elif kernel_type == "dot_only":
            return ConstantKernel() * DotProduct() + WhiteKernel()

        # default
        return ConstantKernel() * RBF() + WhiteKernel()

    # ---------------- Build ----------------
    @staticmethod
    def _normalize_params(params: dict) -> dict:
        """gp_* 接頭辞を正式キーへ正規化"""
        mp = {
            "gp_alpha": "alpha",
            "gp_n_restarts_optimizer": "n_restarts_optimizer",
            "gp_normalize_y": "normalize_y",
            "gp_kernel_type": "kernel_type",
        }
        out = {}
        for k, v in params.items():
            out[mp.get(k, k)] = v
        return out

    def build(self, **params):
        """
        モデル構築（gp_* 正規化 / int化 / n_features 未確定でも安全）
        """
        clean = self._normalize_params(params.copy())

        # 整数パラメータの厳密化
        for key in ("n_restarts_optimizer",):
            if key in clean:
                val = clean[key]
                if isinstance(val, (float, np.floating)):
                    clean[key] = int(round(val))
                elif isinstance(val, np.integer):
                    clean[key] = int(val)

        # kernel_type と n_features
        kernel_type = clean.pop("kernel_type", "rbf_only")
        self._last_kernel_type = kernel_type  # 参考用に保持
        kernel = self._get_kernel(kernel_type, self.n_features)

        # GPR 構築
        self.model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=clean.get("alpha", 1e-6),
            optimizer=clean.get("optimizer", "fmin_l_bfgs_b"),
            n_restarts_optimizer=clean.get("n_restarts_optimizer", 0),
            normalize_y=clean.get("normalize_y", False),
            copy_X_train=True,
            random_state=clean.get("random_state", 42),
        )
        return self.model

    # ---------------- Fit ----------------
    def fit(self, X, y, **kwargs):
        """
        build 前に n_features を記録し、既存モデルの kernel 次元が合わなければ再 build。
        """
        # 先に次元を確定
        if hasattr(X, "shape"):
            self.n_features = int(X.shape[1])

        # モデル未構築なら通常 build
        if self.model is None:
            self.build()
        else:
            # 既存 kernel がベクトル length_scale を持ち、次元がズレるなら再 build
            needs_rebuild = False
            k = getattr(self.model, "kernel_", getattr(self.model, "kernel", None))
            try:
                if isinstance(k, (RBF, Matern)) and hasattr(k, "length_scale"):
                    ls = np.asarray(k.length_scale)
                    if ls.ndim == 1 and (self.n_features is not None) and (ls.size != self.n_features):
                        needs_rebuild = True
            except Exception:
                pass

            if needs_rebuild:
                # 直前に使った kernel_type を使って再生成
                self.model = None
                self.build(kernel_type=self._last_kernel_type)

        return self.model.fit(X, y, **kwargs)

    # ---------------- Predict ----------------
    def predict(self, X, return_std: bool = False):
        if return_std:
            return self.model.predict(X, return_std=True)
        return self.model.predict(X)

    # ---------------- Type hint for SHAP etc. ----------------
    def get_model_type(self) -> str:
        return "gaussian_process"

