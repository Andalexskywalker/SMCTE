# metrics.py
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import fbeta_score, mean_squared_error, make_scorer

def cac_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    thr_log: float = -6.0,
    beta: float = 2.0,
    eps: float = 1e-6
) -> tuple[float, float, float]:
    """
    ESA Collision Avoidance Challenge loss:
      L = weighted_MSE_HR / F2_clipped

    - weighted_MSE_HR: mean squared error on high-risk events, weighted by (risk - thr_log)
    - F2_clipped: F-beta (beta=2) on the binarized predictions, clipped to avoid division by zero
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    # mask high-risk events
    mask_hr = y_true >= thr_log
    if mask_hr.any():
        errors = (y_true[mask_hr] - y_pred[mask_hr]) ** 2
        # weight errors by proximity to threshold (more weight for more critical events)
        weights = (y_true[mask_hr] - thr_log) + eps
        weighted_mse = np.average(errors, weights=weights)
    else:
        weighted_mse = 0.0

    # binarize at threshold
    y_true_bin = (y_true >= thr_log).astype(int)
    y_pred_bin = (y_pred >= thr_log).astype(int)

    # compute F2, force a minimal floor to avoid huge loss
    f2 = fbeta_score(y_true_bin, y_pred_bin, beta=beta, zero_division=0)
    f2_clipped = max(f2, eps)

    # final loss
    L = weighted_mse / f2_clipped
    return L, weighted_mse, f2

def cac_scorer(thr_log: float = -6.0, beta: float = 2.0):
    """ sklearn scorer for GridSearchCV, returns -L (lower is better) """
    def _score(y_true, y_pred):
        L, _, _ = cac_loss(y_true, y_pred, thr_log=thr_log, beta=beta)
        return -L
    return make_scorer(_score, greater_is_better=False)

def f2_at_threshold(
    y_true_log: np.ndarray,
    y_pred_log: np.ndarray,
    thr_log: float = -6.0,
    beta: float = 2.0
) -> float:
    """Compute F2 at a fixed threshold."""
    y_true_bin = (np.asarray(y_true_log) >= thr_log).astype(int)
    y_pred_bin = (np.asarray(y_pred_log) >= thr_log).astype(int)
    return fbeta_score(y_true_bin, y_pred_bin, beta=beta, zero_division=0)

def find_best_threshold(
    y_true_log: np.ndarray,
    y_pred_log: np.ndarray,
    grid: np.ndarray | None = None,
    beta: float = 2.0
) -> tuple[float, float]:
    """
    Search for threshold that maximizes F2, ignoring thresholds
    that yield zero positive predictions.
    """
    if grid is None:
        grid = np.linspace(-7.0, -5.0, 101)
    best_thr, best_f2 = grid[0], 0.0
    y_true = np.asarray(y_true_log).ravel()
    y_pred = np.asarray(y_pred_log).ravel()

    for t in grid:
        y_pred_bin = (y_pred >= t).astype(int)
        if y_pred_bin.sum() == 0:
            continue
        y_true_bin = (y_true >= t).astype(int)
        f2 = fbeta_score(y_true_bin, y_pred_bin, beta=beta, zero_division=0)
        if f2 > best_f2:
            best_f2, best_thr = f2, t

    return best_thr, best_f2

def build_report(
    name: str,
    y_true_log: np.ndarray,
    y_pred_log: np.ndarray,
    thr_log: float = -6.0,
    beta: float = 2.0
) -> pd.DataFrame:
    """Generate a report DataFrame with name, L, weighted MSE_HR, F2, thr_log, beta."""
    L, mse_hr, f2 = cac_loss(y_true_log, y_pred_log, thr_log=thr_log, beta=beta)
    return pd.DataFrame([{
        "name": name,
        "L": L,
        "weighted_MSE_HR": mse_hr,
        "F2": f2,
        "thr_log": thr_log,
        "beta": beta
    }])
from metrics import build_report
import pandas as pd

# exemplo: para cada modelo finalista
rows = []
for name, (y_true_log, y_pred_log) in resultados_por_modelo.items():
    df = build_report(name=name, y_true_log=y_true_log, y_pred_log=y_pred_log, thr_log=-6.0, beta=2.0)
    df["particao"] = "test"  # ou "val"
    rows.append(df)

out = pd.concat(rows, ignore_index=True).rename(columns={"L":"CAC"})
out.to_csv("reports/models_quick/cac_results.csv", index=False)
