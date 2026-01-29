"""
Neyman–Pearson型の陽性しきい値 τ+ と、Clopper–Pearson/Wilson の上側信頼上限による微調整
"""
from typing import Tuple, Literal
import numpy as np
from math import sqrt
from scipy.stats import beta, norm

def np_tau_pos_from_neg_scores(neg_scores: np.ndarray, alpha: float) -> float:
    """
    負例スコアの分位点から τ+ を決める
    alpha=0.0 → 負例最大（FP=0運用）
    alpha>0   → 1-alpha 分位
    """
    neg_scores = np.asarray(neg_scores)
    if neg_scores.size == 0:
        return 1.0
    if alpha <= 0.0:
        return float(np.max(neg_scores))
    q = np.quantile(neg_scores, 1.0 - alpha, method="linear")
    return float(q)

def clopper_pearson_upper(n_success: int, n_total: int, conf: float=0.95) -> float:
    """
    成功率（ここではFPR）のClopper–Pearson上側信頼限界（exact binomial）
    """
    if n_total == 0:
        return 1.0
    # 上側限界 → Beta(1 + x, n - x) の上側 (conf) 分位
    return float(beta.ppf(conf, n_success + 1, n_total - n_success))

def wilson_upper(n_success: int, n_total: int, conf: float=0.95) -> float:
    """
    Wilsonスコア区間の上側限界
    """
    if n_total == 0:
        return 1.0
    p_hat = n_success / n_total
    z = norm.ppf((1 + conf)/2.0)
    denom = 1 + z*z/n_total
    center = (p_hat + z*z/(2*n_total)) / denom
    half_width = (z/denom) * np.sqrt((p_hat*(1-p_hat)/n_total) + (z*z/(4*n_total*n_total)))
    return float(center + half_width)

def adjust_tau_pos_by_ci(
    neg_scores: np.ndarray,
    tau_pos: float,
    alpha_target: float,
    method: Literal["clopper_pearson","wilson"]="clopper_pearson",
    conf: float=0.95,
) -> float:
    """
    Calibration負例に対して τ+ を当て、観測FPRの上側信頼上限が alpha_target を超える場合は τ+ をわずかに引き上げる
    """
    neg_scores = np.asarray(neg_scores)
    if neg_scores.size == 0:
        return tau_pos
    # 観測FPR
    m = int(np.sum(neg_scores >= tau_pos))
    n = int(neg_scores.size)
    if method == "wilson":
        upper = wilson_upper(m, n, conf)
    else:
        upper = clopper_pearson_upper(m, n, conf)
    if upper <= alpha_target:
        return tau_pos
    # 超過している分、tau_pos を少し上げる（分位を1-α'まで動かす簡易実装）
    new_q = 1.0 - max(1e-6, alpha_target*0.9)
    new_tau = float(np.quantile(neg_scores, new_q, method="linear"))
    return max(tau_pos, new_tau)