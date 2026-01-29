"""
校正器（Temperature Scaling / Isotonic）と選択器（ECE + Brier + NLL）
"""
from typing import Dict, Tuple
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, brier_score_loss

def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

class TemperatureScaler:
    """
    ロジット z を入力に取り、温度 T を最適化して
    p = sigmoid(z / T) によるキャリブレーションを行う簡易実装。
    """
    def __init__(self):
        self.T_ = 1.0
        self.fitted_ = False

    def fit(self, logits: np.ndarray, y_true: np.ndarray):
        # ロジット z を特徴、y を目的として、係数 w を学び、p = sigmoid(w * z) と見なす。
        # T = 1/|w| とすれば、p = sigmoid(z / T) となる。
        z = logits.reshape(-1,1)
        lr = LogisticRegression(penalty="none", solver="lbfgs", max_iter=2000)
        lr.fit(z, y_true)
        w = lr.coef_[0,0]
        self.T_ = 1.0/abs(w) if w != 0 else 1.0
        self.fitted_ = True
        return self

    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        if not self.fitted_:
            raise RuntimeError("TemperatureScaler is not fitted")
        z = logits / max(self.T_, 1e-6)
        p = _sigmoid(z)
        return np.clip(p, 1e-7, 1-1e-7)

class IsotonicCalibrator:
    """
    モノトン写像でスコア→確率に補正する古典的な手法。
    """
    def __init__(self):
        self.iso_ = None

    def fit(self, scores: np.ndarray, y_true: np.ndarray):
        self.iso_ = IsotonicRegression(out_of_bounds="clip")
        self.iso_.fit(scores, y_true)
        return self

    def predict_proba(self, scores: np.ndarray) -> np.ndarray:
        return np.clip(self.iso_.predict(scores), 1e-7, 1-1e-7)

def expected_calibration_error(p: np.ndarray, y: np.ndarray, n_bins: int=15) -> float:
    bins = np.linspace(0.0, 1.0, n_bins+1)
    idx = np.digitize(p, bins) - 1
    ece = 0.0
    for b in range(n_bins):
        mask = (idx==b)
        if not np.any(mask):
            continue
        conf = p[mask].mean()
        acc  = y[mask].mean()
        ece += (np.sum(mask) / len(p)) * abs(conf - acc)
    return float(ece)

def select_and_fit_calibrator(
    raw_scores: np.ndarray,
    logits_or_scores: np.ndarray,
    y_true: np.ndarray,
    weights: Dict[str,float] = {"ece":0.5, "brier":0.25, "nll":0.25},
    candidates=("temperature","isotonic"),
) -> Tuple[object, str, Dict[str,float]]:
    """
    候補から最良の校正器を選択して返す
    - temperature: ロジットを受け取る
    - isotonic: スコア（確率相当）を受け取る
    """
    best_name, best_score, best_cal = None, 1e9, None
    metrics = {}
    for name in candidates:
        if name == "temperature":
            cal = TemperatureScaler().fit(logits_or_scores, y_true)
            p = cal.predict_proba(logits_or_scores)
        elif name == "isotonic":
            cal = IsotonicCalibrator().fit(raw_scores, y_true)
            p = cal.predict_proba(raw_scores)
        else:
            continue
        ece = expected_calibration_error(p, y_true)
        brier = brier_score_loss(y_true, p)
        # NLL は2クラスの log_loss を使用
        nll = log_loss(y_true, np.vstack([1-p, p]).T, labels=[0,1])
        score = weights.get("ece",0.5)*ece + weights.get("brier",0.25)*brier + weights.get("nll",0.25)*nll
        metrics[name] = {"ece":ece, "brier":brier, "nll":nll, "composite":score}
        if score < best_score:
            best_score, best_name, best_cal = score, name, cal
    return best_cal, best_name, metrics.get(best_name, {})