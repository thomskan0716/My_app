"""
ユーティリティ関数モジュール（安定化・バグ修正）
- choose_transform: NaN/Inf耐性・安全化
- apply_transform/inverse_transform: 非推奨（AdaptiveTargetTransformerを使用推奨）
- clean_model_params: モデル名正規化、型整形の厳密化、モデル別の安全キャスト
"""
import numpy as np
import random
import os
import warnings
warnings.filterwarnings('ignore')

from scipy import stats
from scipy.stats import shapiro, skew

# 旧API用のグローバル（互換維持のため残置：ターゲット混在時は不適）
_transform_shifts = {}

# ---------------- 乱数シード ----------------
def fix_seed(seed=42):
    """乱数シードを固定（再現性）"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass

    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

# ---------------- 目的変数の変換選択 ----------------
def choose_transform(y, method='auto'):
    """
    目的変数の変換方法を選択
    method: 'auto' | 'none' | 'log' | 'sqrt'
    """
    y = np.asarray(y).astype(float)

    # NaN/Inf を除去
    y = y[np.isfinite(y)]
    if y.size == 0:
        return 'none'

    if method != 'auto':
        return method

    # 負値があるなら log/sqrt は基本的に安全でない（シフトを入れない方針のため）
    if (y <= 0).any():
        return "none"

    methods = []
    try:
        _, p_none = shapiro(y[:min(5000, len(y))])
        methods.append(("none", float(p_none)))
    except Exception:
        methods.append(("none", 0.0))

    try:
        _, p_log = shapiro(np.log(y)[:min(5000, len(y))])
        methods.append(("log", float(p_log)))
    except Exception:
        pass

    try:
        _, p_sqrt = shapiro(np.sqrt(y)[:min(5000, len(y))])
        methods.append(("sqrt", float(p_sqrt)))
    except Exception:
        pass

    if methods:
        best_method, best_p = max(methods, key=lambda x: x[1])
        if best_p > 0.05:
            return best_method

    # フォールバック：歪度
    try:
        sk = float(skew(y))
    except Exception:
        sk = 0.0
    if abs(sk) < 0.5:
        return "none"
    elif sk > 1.0:
        return "log"
    elif sk > 0.5:
        return "sqrt"
    else:
        return "none"

# ---------------- 旧API（非推奨） ----------------
def apply_transform(y, method):
    """
    旧API: 目的変数の変換を適用（非推奨）
    * 複数ターゲット混在時に shift が上書きされ逆変換破綻の恐れあり。
    * 現行は AdaptiveTargetTransformer を使用してください。
    """
    warnings.warn(
        "apply_transform()/inverse_transform() は非推奨です。"
        "AdaptiveTargetTransformer を使用してください。",
        DeprecationWarning
    )
    global _transform_shifts

    y = np.asarray(y).astype(float)
    if method == "log":
        if (y <= 0).any():
            min_val = np.min(y)
            shift = abs(min_val) + 1.0
            _transform_shifts[method] = shift
            return np.log(y + shift)
        else:
            _transform_shifts[method] = 0.0
            return np.log(y)

    elif method == "sqrt":
        if (y < 0).any():
            min_val = np.min(y)
            shift = abs(min_val)
            _transform_shifts[method] = shift
            return np.sqrt(y + shift)
        else:
            _transform_shifts[method] = 0.0
            return np.sqrt(y)

    else:
        _transform_shifts[method] = 0.0
        return y

def inverse_transform(y_transformed, method):
    """
    旧API: 逆変換（非推奨）
    * 複数ターゲットでの shift 共有に注意。AdaptiveTargetTransformer を使用推奨。
    """
    warnings.warn(
        "apply_transform()/inverse_transform() は非推奨です。"
        "AdaptiveTargetTransformer を使用してください。",
        DeprecationWarning
    )
    y_transformed = np.asarray(y_transformed).astype(float)
    shift = float(_transform_shifts.get(method, 0.0))

    if method == "log":
        y_original = np.exp(y_transformed)
        return y_original - shift if shift > 0 else y_original
    elif method == "sqrt":
        y_original = np.square(y_transformed)
        return y_original - shift if shift > 0 else y_original
    else:
        return y_transformed

# ---------------- モデルパラメータの型整形 ----------------
def _int_if_intlike(val):
    """安全に int 化：整数型 or ほぼ整数の非負値のみ int にする。割合(0<val<1)は保持。"""
    if val is None:
        return val
    if isinstance(val, (np.integer, int)):
        return int(val)
    if isinstance(val, (float, np.floating)):
        if val >= 0 and abs(val - round(val)) < 1e-9:
            return int(round(val))
    return val

def _float_if_number(val):
    """numpyスカラーを素のfloatへ（型の揺れ抑制）"""
    if isinstance(val, (np.floating, float, np.integer, int)):
        return float(val)
    return val

def clean_model_params(params: dict, model_name: str) -> dict:
    """
    モデルパラメータの型を適切に変換（安定化）
    - モデル名は小文字に正規化
    - 既知の整数パラメータのみ安全に int 化
    - 既知の浮動小数パラメータは float 化
    """
    cleaned = dict(params) if params else {}
    model_key = (model_name or '').lower()

    # モデル別：整数にしたいキー集合（慎重に設定）
    INT_PARAMS_BY_MODEL = {
        'lightgbm': {
            'n_estimators', 'num_leaves', 'bagging_freq',
            'min_child_samples', 'subsample_for_bin',
            'min_data_in_bin', 'max_bin', 'max_depth'
        },
        'xgboost': {
            'n_estimators', 'max_depth', 'max_leaves', 'max_bin'
            # NOTE: min_child_weight は float なので int 化しない
        },
        'random_forest': {
            'n_estimators', 'max_depth', 'max_leaf_nodes'
            # NOTE: min_samples_split / min_samples_leaf は割合(float)も取りうるので一律int化しない
        },
        'randomforest': {  # エイリアス
            'n_estimators', 'max_depth', 'max_leaf_nodes'
        },
        'gradient_boost': {
            'n_estimators', 'max_depth', 'max_leaf_nodes'
            # 同上：min_samples_* は割合対応のため除外
        },
        'gradientboost': {  # エイリアス
            'n_estimators', 'max_depth', 'max_leaf_nodes'
        },
        'elastic_net': set(),
        'ridge': set(),
        'lasso': set(),
        'catboost': {
            'iterations', 'depth', 'border_count', 'random_state', 'thread_count'
        },
        'gaussian_process': {
            'n_restarts_optimizer'
        },
        'gp': {
            'n_restarts_optimizer'
        }
    }

    # モデル別：float に寄せたいキー（学習率・正則化など）
    FLOAT_PARAMS_COMMON = {
        'learning_rate', 'subsample', 'colsample_bytree',
        'reg_alpha', 'reg_lambda', 'gamma', 'min_child_weight'
    }

    # 1) モデル固有の整数パラメータを安全に int 化
    int_keys = INT_PARAMS_BY_MODEL.get(model_key, set())
    for k in list(cleaned.keys()):
        if k in int_keys:
            cleaned[k] = _int_if_intlike(cleaned[k])

    # 2) 共通の「確実に整数」系（本パイプライン側で使いうる）
    for k in ('top_k', 'max_iter', 'n_neighbors'):
        if k in cleaned:
            cleaned[k] = _int_if_intlike(cleaned[k])

    # 3) 浮動小数へ寄せるキー
    for k in list(cleaned.keys()):
        if k in FLOAT_PARAMS_COMMON:
            cleaned[k] = _float_if_number(cleaned[k])

    return cleaned
