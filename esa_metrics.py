#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
esa_metrics.py — Métricas do ESA Collision Avoidance (versão prática para o projeto)
------------------------------------------------------------------------------------
Fornece métricas alinhadas com o que a prof pediu e com o que usámos no relatório:
- Classificação binária no limiar de risco (por defeito, log-risk >= -6 é positivo)
- F2, precisão, recall, balanced accuracy e PR-AUC (AP)
- MSE global e MSE em alto-risco (HR: y_true >= thr_log)
- "CAC loss" prática: L = MSE_HR / max(F2, eps)  (↓ melhor)

Também inclui:
- Procura de threshold ótimo por F2 (grid em log-risk)
- Relatórios em CSV/JSON e PR-curve em PNG
- (Opcional) métricas ao nível do evento se existir 'event_id' (via max por evento)

Uso 1 (biblioteca):
    from esa_metrics import build_report
    df = build_report("modeloX", y_true_log, y_pred_log, thr_log=-6.0)

Uso 2 (CLI sobre CSV com colunas):
    python esa_metrics.py --csv preds.csv --y_true_col y_true_log --y_pred_col y_pred_log \
                          --event_col event_id --thr -6 --out reports/metrics_esa

O CSV deve ter pelo menos as colunas y_true_log e y_pred_log (em log-risk coerente com o limiar).
"""
from __future__ import annotations
import argparse, json, math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    average_precision_score, roc_auc_score, precision_recall_curve,
    precision_score, recall_score, fbeta_score, balanced_accuracy_score,
    confusion_matrix
)

EPS = 1e-12

# ------------------------------ utilitários ------------------------------
def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True); return p

def _pr_curve(y_true_bin: np.ndarray, scores: np.ndarray, title: str, out_png: Path):
    p, r, _ = precision_recall_curve(y_true_bin, scores)
    plt.figure(figsize=(5,4))
    plt.plot(r, p)
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(title)
    plt.grid(True, ls=":")
    plt.tight_layout(); plt.savefig(out_png, dpi=180); plt.close()

def _conf_counts(y_true_bin: np.ndarray, y_pred_bin: np.ndarray) -> Dict[str,int]:
    tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred_bin, labels=[0,1]).ravel()
    return {"tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)}

def _mse(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0:
        return float("nan")
    d = (a - b)**2
    return float(np.mean(d))

def best_threshold_by_f2(
    y_true_log: np.ndarray,
    y_pred_log: np.ndarray,
    beta: float = 2.0,
    grid: Optional[np.ndarray] = None
) -> Tuple[float, float]:
    """
    Busca o limiar em log-risk que maximiza F_beta, comparando excedência do limiar em y_true e y_pred.
    Por defeito, grid uniforme entre [-9, -2] com passo 0.05.
    """
    if grid is None:
        grid = np.arange(-9.0, -2.0 + 1e-9, 0.05, dtype=float)
    best_thr, best_f = None, -1.0
    for t in grid:
        ytb = (y_true_log >= t).astype(int)
        ypb = (y_pred_log >= t).astype(int)
        if ytb.max() == 0 or ypb.max() == 0:
            # sem positivos numa das séries → F_beta = 0
            f = 0.0
        else:
            f = fbeta_score(ytb, ypb, beta=beta, zero_division=0)
        if f > best_f:
            best_f, best_thr = f, float(t)
    return (float(best_thr) if best_thr is not None else float(-6.0), float(best_f))

# ------------------------------ núcleo das métricas ------------------------------
@dataclass
class EsaMetrics:
    name: str
    thr_log: float
    beta: float
    n: int
    pos_ratio: float
    ap: float
    roc_auc: float
    f2: float
    precision: float
    recall: float
    balanced_acc: float
    tp: int; fp: int; tn: int; fn: int
    mse_all: float
    mse_hr: float
    cac_loss: float
    thr_opt: float
    f2_opt: float

def compute_esa_metrics(
    y_true_log: np.ndarray,
    y_pred_log: np.ndarray,
    thr_log: float = -6.0,
    beta: float = 2.0,
    name: str = "model",
) -> EsaMetrics:
    y_true_log = np.asarray(y_true_log, dtype=float).ravel()
    y_pred_log = np.asarray(y_pred_log, dtype=float).ravel()
    assert y_true_log.shape == y_pred_log.shape, "y_true_log e y_pred_log com formas diferentes"

    # binário no limiar fixo
    y_true_bin = (y_true_log >= thr_log).astype(int)
    scores = y_pred_log  # maior = mais risco
    y_pred_bin = (y_pred_log >= thr_log).astype(int)

    # métricas de ranking e binárias
    has_both = np.unique(y_true_bin).size > 1
    ap = float(average_precision_score(y_true_bin, scores)) if has_both else float("nan")
    rocauc = float(roc_auc_score(y_true_bin, scores)) if has_both else float("nan")
    f2 = float(fbeta_score(y_true_bin, y_pred_bin, beta=beta, zero_division=0))
    prec = float(precision_score(y_true_bin, y_pred_bin, zero_division=0))
    rec = float(recall_score(y_true_bin, y_pred_bin, zero_division=0))
    bacc = float(balanced_accuracy_score(y_true_bin, y_pred_bin))

    cc = _conf_counts(y_true_bin, y_pred_bin)

    # MSE global e em alto-risco (HR = y_true >= thr)
    mse_all = _mse(y_true_log, y_pred_log)
    mask_hr = y_true_bin == 1
    mse_hr = _mse(y_true_log[mask_hr], y_pred_log[mask_hr])

    # "CAC loss" prática (↓ melhor)
    # evita divisão por zero via clip de F2
    f2_clip = max(f2, EPS)
    cac = float(mse_hr) / float(f2_clip) if not math.isnan(mse_hr) else float("inf")

    # threshold ótimo por F2 (provas em validação)
    thr_opt, f2_opt = best_threshold_by_f2(y_true_log, y_pred_log, beta=beta)

    return EsaMetrics(
        name=name, thr_log=float(thr_log), beta=float(beta),
        n=int(len(y_true_log)), pos_ratio=float(y_true_bin.mean()),
        ap=ap, roc_auc=rocauc, f2=f2, precision=prec, recall=rec, balanced_acc=bacc,
        tp=cc["tp"], fp=cc["fp"], tn=cc["tn"], fn=cc["fn"],
        mse_all=mse_all, mse_hr=mse_hr, cac_loss=cac, thr_opt=thr_opt, f2_opt=f2_opt
    )

def build_report(
    name: str,
    y_true_log: np.ndarray,
    y_pred_log: np.ndarray,
    thr_log: float = -6.0,
    beta: float = 2.0,
    out_dir: Optional[Path] = None
) -> pd.DataFrame:
    """
    Devolve um DataFrame (1 linha) com todas as métricas e, se out_dir for passado,
    grava também PR-curve (PNG) e JSON com as métricas.
    """
    mets = compute_esa_metrics(y_true_log, y_pred_log, thr_log=thr_log, beta=beta, name=name)
    df = pd.DataFrame([asdict(mets)])
    if out_dir is not None:
        out_dir = _ensure_dir(Path(out_dir))
        # PR curve
        y_true_bin = (np.asarray(y_true_log) >= thr_log).astype(int)
        _pr_curve(y_true_bin, np.asarray(y_pred_log), f"PR — {name} (thr={thr_log})", out_dir / f"pr_{name}.png")
        # JSON
        with open(out_dir / f"metrics_{name}.json","w") as f:
            json.dump(asdict(mets), f, indent=2)
        # CSV (tabela de uma linha)
        df.to_csv(out_dir / f"metrics_{name}.csv", index=False)
    return df

# ------------------------------ CLI ------------------------------
def _load_csv(path: Path, y_true_col: str, y_pred_col: str, event_col: Optional[str]):
    df = pd.read_csv(path)
    if y_true_col not in df.columns or y_pred_col not in df.columns:
        raise SystemExit(f"Colunas '{y_true_col}' e '{y_pred_col}' são obrigatórias no CSV.")
    y_true = df[y_true_col].to_numpy()
    y_pred = df[y_pred_col].to_numpy()
    events = df[event_col].to_numpy() if (event_col and event_col in df.columns) else None
    return y_true, y_pred, events

def _aggregate_by_event(y_true_log: np.ndarray, y_pred_log: np.ndarray, events: np.ndarray):
    # agrega por evento usando o máximo do log-risk (equivalente ao máximo do risco)
    import pandas as pd
    tmp = pd.DataFrame({"ev": events, "yt": y_true_log, "yp": y_pred_log})
    gb = tmp.groupby("ev", sort=False).agg(yt=("yt","max"), yp=("yp","max"))
    return gb["yt"].to_numpy(), gb["yp"].to_numpy()

def main():
    ap = argparse.ArgumentParser(description="ESA Metrics (CAC-style) — relatório rápido")
    ap.add_argument("--csv", required=True, help="CSV com colunas de verdade e previsão em log-risk.")
    ap.add_argument("--y_true_col", default="y_true_log")
    ap.add_argument("--y_pred_col", default="y_pred_log")
    ap.add_argument("--event_col", default=None, help="Opcional: event_id para métricas ao nível do evento")
    ap.add_argument("--thr", type=float, default=-6.0, help="Limiar do log-risk para classe positiva")
    ap.add_argument("--beta", type=float, default=2.0, help="Beta para F-beta (default F2)")
    ap.add_argument("--out", default="reports/metrics_esa", help="Pasta de saída para relatórios")
    args = ap.parse_args()

    out = _ensure_dir(Path(args.out))
    y_true, y_pred, events = _load_csv(Path(args.csv), args.y_true_col, args.y_pred_col, args.event_col)

    # row-level
    row_df = build_report("row", y_true, y_pred, thr_log=args.thr, beta=args.beta, out_dir=out)

    # event-level (se existir event_id)
    if events is not None:
        yt_e, yp_e = _aggregate_by_event(y_true, y_pred, events)
        ev_df = build_report("event", yt_e, yp_e, thr_log=args.thr, beta=args.beta, out_dir=out)
        all_df = pd.concat([row_df.assign(level="row"), ev_df.assign(level="event")], ignore_index=True)
    else:
        all_df = row_df.assign(level="row")

    # CSV consolidado
    all_df.to_csv(out / "esa_metrics_summary.csv", index=False)
    print("[OK] ESA metrics salvas em:", out.resolve())
    print(all_df.to_string(index=False))

if __name__ == "__main__":
    main()
