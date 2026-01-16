# -*- coding: utf-8 -*-
"""Preprocess v4.0-seq — descrição opcional do módulo"""

from __future__ import annotations

import os, json, argparse, logging, warnings
from typing import Tuple, List, Optional, Dict
import numpy as np
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# ---------- importar o clipper “seguro” (na MESMA pasta) ----------
import sys
sys.path.append(os.path.dirname(__file__))  # garante que a pasta atual está no path
try:
    from safe_clip import ClipByLimitsTransformer
except Exception:
    # fallback: passthrough se o módulo não existir
    class ClipByLimitsTransformer:
        def __init__(self, limits=None, feature_names=None): pass
        def fit(self, X, y=None): return self
        def transform(self, X): return X

# ----------------------------- Logging -----------------------------
def setup_logging(level=logging.INFO):
    logging.basicConfig(level=level, format='[%(asctime)s] %(levelname)s - %(message)s', datefmt='%H:%M:%S')

# ----------------------------- I/O -----------------------------
def save_npz_tensor(path: str, X: np.ndarray, y: np.ndarray, mask: np.ndarray,
                    feature_names: List[str], seq_len: int, event_ids: List[int]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, X=X, y=y, mask=mask,
                        feature_names=np.array(feature_names, dtype=object),
                        seq_len=np.int32(seq_len), event_ids=np.array(event_ids))

def save_ids_csv(path: str, event_ids: List[int], y: np.ndarray, target_col: str):
    df = pd.DataFrame({"event_id": event_ids, target_col: y})
    df.to_csv(path, index=False)

# ----------------------------- Helpers de evento -----------------------------
def event_label_series(
    df: pd.DataFrame,
    event_col: str,
    target_col: str,
    task: str,
    positive_threshold: float,
    strategy: str = "any_high_risk"
) -> pd.Series:
    g = df.sort_values([event_col, "time_to_tca"]).groupby(event_col, sort=True)[target_col]
    if task == "classification":
        uniq = set(pd.unique(df[target_col].dropna()))
        is_binary = uniq.issubset({0, 1})
        if strategy == "last":
            s = g.last()
            return (s >= (1 if is_binary else positive_threshold)).astype(int)
        else:
            s = g.max()
            return (s >= (1 if is_binary else positive_threshold)).astype(int)
    else:
        if strategy in ("max", "any_high_risk"): return g.max()
        elif strategy == "last": return g.last()
        else: return g.max()

def split_by_event(
    df: pd.DataFrame,
    event_col: str,
    test_size: float,
    val_size: float,
    random_state: int,
    stratify_events: bool,
    task: str,
    target_col: str,
    positive_threshold: float,
    event_target_strategy: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if stratify_events:
        ev_label = event_label_series(df, event_col, target_col, task, positive_threshold, event_target_strategy)
        ev_df = pd.DataFrame({event_col: ev_label.index, "y_ev": ev_label.values})
        tr_ev, te_ev = train_test_split(ev_df, test_size=test_size, random_state=random_state, stratify=ev_df["y_ev"])
        tr_ev, val_ev = train_test_split(tr_ev, test_size=val_size, random_state=random_state, stratify=tr_ev["y_ev"])
        tr_ids, val_ids, te_ids = tr_ev[event_col].values, val_ev[event_col].values, te_ev[event_col].values
    else:
        events = df[event_col].unique()
        tr_ids, te_ids = train_test_split(events, test_size=test_size, random_state=random_state)
        tr_ids, val_ids = train_test_split(tr_ids, test_size=val_size, random_state=random_state)
    return (df[df[event_col].isin(tr_ids)].copy(),
            df[df[event_col].isin(val_ids)].copy(),
            df[df[event_col].isin(te_ids)].copy())

# ----------------------------- Limpeza (sem fuga) -----------------------------
def physical_sanity(df: pd.DataFrame) -> None:
    if "miss_distance" in df.columns:
        df.loc[df["miss_distance"] < 0, "miss_distance"] = 0.0
    if "relative_speed" in df.columns:
        df.loc[df["relative_speed"] <= 0, "relative_speed"] = np.nan

def role_median_impute(train_df: pd.DataFrame, df: pd.DataFrame) -> None:
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    train_num = train_df[num_cols].replace([np.inf, -np.inf], np.nan)
    df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)
    for role in ["t_", "c_"]:
        rcols = [c for c in num_cols if c.startswith(role)]
        if rcols:
            med = train_num[rcols].median()
            for c in rcols:
                df[c] = df[c].fillna(med.get(c, train_num[c].median() if c in train_num else df[c].median()))
    rest = [c for c in num_cols if not (c.startswith("t_") or c.startswith("c_"))]
    if rest:
        med_rest = train_num[rest].median()
        for c in rest:
            df[c] = df[c].fillna(med_rest.get(c, train_num[c].median() if c in train_num else df[c].median()))
    for c in df.select_dtypes(include=["object", "category"]).columns:
        df[c] = df[c].fillna("UNKNOWN")

def ffill_by_event(df: pd.DataFrame, event_col: str) -> None:
    if {event_col, "time_to_tca"} <= set(df.columns):
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        df.sort_values([event_col, "time_to_tca"], ascending=[True, False], inplace=True)
        df[num_cols] = df.groupby(event_col, group_keys=False)[num_cols].ffill()
        df.sort_index(inplace=True)

def drop_by_patterns(df: pd.DataFrame, patterns: List[str]) -> List[str]:
    dropped = []
    if not patterns: return dropped
    low = [p.lower() for p in patterns]
    for c in list(df.columns):
        if any(p in c.lower() for p in low):
            df.drop(columns=[c], inplace=True, errors="ignore")
            dropped.append(c)
    return dropped

# ----------------------------- Clip -----------------------------
def load_clip_limits(path: Optional[str]) -> Dict[str, Dict[str, float]]:
    if not path: return {}
    if not os.path.exists(path): raise FileNotFoundError(f"clip_limits_json '{path}' não encontrado")
    with open(path) as f: data = json.load(f)
    return {k: {"low": v.get("low"), "high": v.get("high")} for k, v in data.items()}

# ----------------------------- Preprocessor -----------------------------
def build_preprocessor(train_df: pd.DataFrame,
                       target_col: str,
                       event_col: str,
                       clip_limits: Dict[str, Dict[str, float]],
                       extra_cat: Optional[List[str]] = None) -> Tuple[ColumnTransformer, List[str], List[str]]:
    feat_cols = [c for c in train_df.columns if c not in {target_col, event_col}]
    num_cols = train_df[feat_cols].select_dtypes(include=np.number).columns.tolist()
    cat_cols = sorted(set(feat_cols) - set(num_cols))
    if extra_cat: cat_cols = sorted(set(cat_cols) | set(extra_cat))

    clipper = "passthrough"
    if clip_limits:
        clipper = ClipByLimitsTransformer(limits=clip_limits, feature_names=num_cols)

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clip", clipper),
        ("scaler", StandardScaler()),
    ])

    ohe_kwargs = {"handle_unknown": "ignore"}
    try:
        ohe = OneHotEncoder(sparse_output=False, **ohe_kwargs)
    except TypeError:
        ohe = OneHotEncoder(sparse=False, **ohe_kwargs)

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", ohe),
    ])

    pre = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ])
    return pre, num_cols, cat_cols

# ----------------------------- Sequências -----------------------------
def build_windows_tensor(df: pd.DataFrame,
                         event_col: str,
                         feature_cols: List[str],
                         timestep: int,
                         fill_value: float,
                         sort_by: str = "time_to_tca",
                         order: str = "asc") -> tuple[np.ndarray, np.ndarray, list[int]]:
    X_list, M_list, ids = [], [], []
    asc = (order == "asc")
    for ev_id, g in df.sort_values([event_col, sort_by], ascending=[True, asc]).groupby(event_col, sort=True):
        arr = g[feature_cols].to_numpy(dtype=float)
        L = arr.shape[0]
        if L >= timestep:
            w = arr[-timestep:] if asc else arr[:timestep]
            m = np.ones((timestep,), dtype=np.float32)
        else:
            pad = np.full((timestep - L, arr.shape[1]), fill_value, dtype=float)
            w = np.vstack([pad, arr]) if asc else np.vstack([arr, pad])
            m = np.hstack([np.zeros(timestep - L, dtype=np.float32), np.ones(L, dtype=np.float32)])
        X_list.append(w); M_list.append(m); ids.append(ev_id)
    X = np.stack(X_list, 0); M = np.stack(M_list, 0)
    return X, M, ids

def event_targets_in_order(df: pd.DataFrame,
                           event_col: str,
                           target_col: str,
                           task: str,
                           threshold: float,
                           strategy: str,
                           event_order: Optional[List[int]] = None) -> np.ndarray:
    y_map: Dict[int, float | int] = {}
    for ev, grp in df.sort_values([event_col, "time_to_tca"]).groupby(event_col, sort=True):
        s = grp[target_col]
        if task == "classification":
            uniq = set(pd.unique(s.dropna())); is_binary = uniq.issubset({0,1})
            val = (s.iloc[-1] if strategy=="last" else s.max()) if len(s) else 0
            y_map[ev] = int(val >= (1 if is_binary else threshold))
        else:
            y_map[ev] = float((s.iloc[-1] if strategy=="last" else s.max()) if len(s) else np.nan)
    if event_order is None: event_order = list(y_map.keys())
    return np.asarray([y_map[ev] for ev in event_order])

# ----------------------------- Aux: Tabela 9.1 -----------------------------
def _count_stats_df(df: pd.DataFrame, target_col: str, part_name: str):
    n = int(len(df))
    s = pd.to_numeric(df[target_col], errors="coerce")
    uniq = set(np.unique(s.dropna()))
    pos = int((s == 1).sum()) if uniq.issubset({0,1}) else int((s > 0).sum())
    p = (pos/n*100.0) if n else 0.0
    return {"Partição": part_name, "N linhas": n, "N positivos": pos, "% positivos": round(p, 3)}

# ----------------------------- CLI -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Preprocess v4.0-seq — outputs sequenciais (.npz) + CSVs proc + Tab 9.1")
    ap.add_argument("--input", required=True, help="CSV/Parquet com dados brutos")
    ap.add_argument("--output_train", required=True, help="caminho .npz (será forçada a extensão .npz)")
    ap.add_argument("--output_val",   required=True, help="caminho .npz (será forçada a extensão .npz)")
    ap.add_argument("--output_test",  required=True, help="caminho .npz (será forçada a extensão .npz)")

    # tarefa & rótulos
    ap.add_argument("--target_col", default="risk")
    ap.add_argument("--event_col",  default="event_id")
    ap.add_argument("--task", choices=["regression","classification"], default="classification")
    ap.add_argument("--positive_threshold", type=float, default=-6.0)
    ap.add_argument("--event_target_strategy", choices=["last","max","any_high_risk"], default="any_high_risk")

    # split
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--val_size",  type=float, default=0.1)
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--stratify_events", action="store_true")

    # pré-processamento
    ap.add_argument("--clip_limits_json", default=None, help="reports/eda/clip_limits.json")
    ap.add_argument("--extra_cat", nargs="*", default=None)
    ap.add_argument("--drop_patterns", nargs="*", default=[],
                    help="padrões a remover (ex.: max_risk _at_tca peak post_ future after_tca lastob rcs_estimate tca_flag label target)")
    ap.add_argument("--drop_id_like", action="store_true", help="remove colunas id/uid (event_id nunca entra no preprocessor)")
    ap.add_argument("--save_preprocessor", default=None, help="salva preprocessor (.joblib)")

    # sequências
    ap.add_argument("--seq_config_json", default=None, help="reports/eda/seq/seq_config.json")
    ap.add_argument("--sequence_timestep", type=int, default=0, help="T; se 0, usa recommended_seq_len do EDA")
    ap.add_argument("--sequence_order", choices=["asc","desc"], default="asc", help="ordem temporal por time_to_tca")
    ap.add_argument("--sequence_mask_value", type=float, default=0.0)
    ap.add_argument("--sequence_numeric_only", action="store_true", default=True,
                    help="usar apenas features numéricas (recomendado para RNAs). Default: True")

    # NOVO: CSVs proc + Tabela 9.1
    ap.add_argument("--save_proc_csv_dir", default=None,
                    help="Se definido, salva CSVs 'proc': train_proc.csv, val_proc.csv, test_proc.csv")
    ap.add_argument("--tab9_1_out", default=None,
                    help="(Opcional) CSV da Tabela 9.1 — Partições e prevalência por classe")

    args = ap.parse_args()
    setup_logging()

    # Força extensão .npz
    def force_npz(p: str) -> str:
        base, _ = os.path.splitext(p)
        return base + ".npz"
    out_tr, out_va, out_te = map(force_npz, [args.output_train, args.output_val, args.output_test])

    # Load
    df = pd.read_csv(args.input) if args.input.lower().endswith(".csv") else pd.read_parquet(args.input)
    logging.info("Loaded %s shape=%s", args.input, df.shape)

    # Drop por padrões (potenciais fugas)
    dropped = drop_by_patterns(df, args.drop_patterns)
    if dropped: logging.info("Drop por padrões (%d): %s", len(dropped), dropped)

    # Drop ids se pedido (mantém event_id)
    id_like = {"id", "uid"}
    if args.drop_id_like:
        to_drop_ids = [c for c in df.columns if c.lower() in id_like and c != args.event_col]
        if to_drop_ids:
            df = df.drop(columns=to_drop_ids, errors="ignore")
            logging.info("IDs removidos: %s", to_drop_ids)

    # Limpeza base
    physical_sanity(df)

    # Split por evento
    train_df, val_df, test_df = split_by_event(
        df, args.event_col, args.test_size, args.val_size, args.random_state,
        stratify_events=args.stratify_events, task=args.task,
        target_col=args.target_col, positive_threshold=args.positive_threshold,
        event_target_strategy=args.event_target_strategy
    )
    logging.info("Splits events -> train=%d | val=%d | test=%d",
                 train_df[args.event_col].nunique(),
                 val_df[args.event_col].nunique(),
                 test_df[args.event_col].nunique())

    # Limpeza SEM fuga (fit no train; aplicar a todos)
    for part in (train_df, val_df, test_df): physical_sanity(part)
    role_median_impute(train_df, train_df); ffill_by_event(train_df, args.event_col)
    role_median_impute(train_df, val_df);   ffill_by_event(val_df, args.event_col)
    role_median_impute(train_df, test_df);  ffill_by_event(test_df, args.event_col)

    # >>> Salvar CSVs PROC por partição (row-level, já limpos)
    if args.save_proc_csv_dir:
        os.makedirs(args.save_proc_csv_dir, exist_ok=True)
        train_df.to_csv(os.path.join(args.save_proc_csv_dir, "train_proc.csv"), index=False)
        val_df.to_csv(  os.path.join(args.save_proc_csv_dir, "val_proc.csv"),   index=False)
        test_df.to_csv( os.path.join(args.save_proc_csv_dir, "test_proc.csv"),  index=False)
        logging.info("PROC CSVs gravados em %s", args.save_proc_csv_dir)

    # >>> (Opcional) Tabela 9.1
    if args.tab9_1_out:
        tab = pd.DataFrame([
            _count_stats_df(train_df, args.target_col, "train"),
            _count_stats_df(val_df,   args.target_col, "val"),
            _count_stats_df(test_df,  args.target_col, "test"),
        ], columns=["Partição","N linhas","N positivos","% positivos"])
        os.makedirs(os.path.dirname(args.tab9_1_out), exist_ok=True)
        tab.to_csv(args.tab9_1_out, index=False)
        logging.info("Tabela 9.1 salva em %s", args.tab9_1_out)

    # Seq config do EDA (opcional)
    seq_len_hint, seq_feats_hint = None, None
    if args.seq_config_json and os.path.exists(args.seq_config_json):
        with open(args.seq_config_json) as f:
            cfg = json.load(f)
        seq_len_hint  = int(cfg.get("recommended_seq_len") or 0) or None
        seq_feats_hint = cfg.get("seq_features") or None

    # Definir T
    T = args.sequence_timestep or (seq_len_hint or 0)
    if T <= 0:
        raise ValueError("Defina --sequence_timestep > 0 OU forneça --seq_config_json com 'recommended_seq_len'.")

    # Preprocessor (fit no train)
    clip_limits = load_clip_limits(args.clip_limits_json)
    pre, num_cols, cat_cols = build_preprocessor(train_df, args.target_col, args.event_col, clip_limits, extra_cat=args.extra_cat)

    # X/y row-level (apenas para transformar; labels são ao nível do evento depois)
    def features_only(dfin: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in dfin.columns if c not in {args.target_col, args.event_col}]
        return dfin[cols]

    Xtr_raw = features_only(train_df);  Xva_raw = features_only(val_df);  Xte_raw = features_only(test_df)
    Xtr = pre.fit_transform(Xtr_raw);   Xva = pre.transform(Xva_raw);     Xte = pre.transform(Xte_raw)

    # Guardar preprocessor (opcional)
    if args.save_preprocessor:
        os.makedirs(os.path.dirname(args.save_preprocessor), exist_ok=True)
        joblib.dump(pre, args.save_preprocessor)
        logging.info("Preprocessor salvo em %s", args.save_preprocessor)

    # Nomes finais de features após o preprocessor
    cat_feature_names = []
    try:
        ohe = pre.named_transformers_["cat"].named_steps.get("ohe", None)
        if ohe is not None and len(cat_cols):
            cat_feature_names = list(ohe.get_feature_names_out(cat_cols))
    except Exception:
        pass
    final_feature_names = num_cols + cat_feature_names  # ordem consistente

    # Escolha das features para a sequência
    if args.sequence_numeric_only:
        seq_feats = [c for c in final_feature_names if c in num_cols]
    else:
        seq_feats = list(final_feature_names)

    # (opcional) respeitar seq_features do EDA
    if seq_feats_hint:
        chosen = []
        for name in final_feature_names:
            for base in seq_feats_hint:
                if name == base or name.startswith(base + "_") or name.startswith(base + "="):
                    chosen.append(name); break
        if chosen:
            seq_feats = chosen

    # Constrói DataFrames com event_id & time_to_tca alinhados à transformação
    def attach_ev(df_ev: pd.DataFrame, X_trans: np.ndarray, feat_names: List[str]) -> pd.DataFrame:
        aux = (df_ev[[args.event_col, "time_to_tca"]]
               .reset_index(drop=True)
               .rename(columns={"time_to_tca": "_t2tca"}))  # evita duplicação
        Xdf = pd.DataFrame(X_trans, columns=final_feature_names).reset_index(drop=True)
        if "_t2tca" in Xdf.columns:  # por segurança
            Xdf = Xdf.drop(columns=["_t2tca"])
        return pd.concat([aux, Xdf], axis=1)

    tr_seq_in = attach_ev(train_df, Xtr, final_feature_names)
    va_seq_in = attach_ev(val_df,   Xva, final_feature_names)
    te_seq_in = attach_ev(test_df,  Xte, final_feature_names)

    # Constrói janelas e máscara
    X_tr, M_tr, ids_tr = build_windows_tensor(tr_seq_in, args.event_col, seq_feats,
                                              timestep=T, fill_value=args.sequence_mask_value,
                                              sort_by="_t2tca", order=args.sequence_order)
    X_va, M_va, ids_va = build_windows_tensor(va_seq_in, args.event_col, seq_feats,
                                              timestep=T, fill_value=args.sequence_mask_value,
                                              sort_by="_t2tca", order=args.sequence_order)
    X_te, M_te, ids_te = build_windows_tensor(te_seq_in, args.event_col, seq_feats,
                                              timestep=T, fill_value=args.sequence_mask_value,
                                              sort_by="_t2tca", order=args.sequence_order)

    # Labels ao nível do evento (alinhados por ids)
    y_tr = event_targets_in_order(train_df, args.event_col, args.target_col, args.task,
                                  args.positive_threshold, args.event_target_strategy, ids_tr)
    y_va = event_targets_in_order(val_df,   args.event_col, args.target_col, args.task,
                                  args.positive_threshold, args.event_target_strategy, ids_va)
    y_te = event_targets_in_order(test_df,  args.event_col, args.target_col, args.task,
                                  args.positive_threshold, args.event_target_strategy, ids_te)

    # Salvar npz + CSVs auxiliares (ids,y)
    save_npz_tensor(out_tr, X_tr, y_tr, M_tr, seq_feats, T, ids_tr)
    save_npz_tensor(out_va, X_va, y_va, M_va, seq_feats, T, ids_va)
    save_npz_tensor(out_te, X_te, y_te, M_te, seq_feats, T, ids_te)
    save_ids_csv(out_tr.replace(".npz", "_ids.csv"), ids_tr, y_tr, args.target_col)
    save_ids_csv(out_va.replace(".npz", "_ids.csv"), ids_va, y_va, args.target_col)
    save_ids_csv(out_te.replace(".npz", "_ids.csv"), ids_te, y_te, args.target_col)

    logging.info("Gravado:\n  %s\n  %s\n  %s", out_tr, out_va, out_te)
    logging.info("Shapes: Xtr=%s | Xva=%s | Xte=%s  (mask idem)", X_tr.shape, X_va.shape, X_te.shape)
    logging.info("Features usadas (%d): %s", len(seq_feats), seq_feats[:10] + (["..."] if len(seq_feats)>10 else []))

if __name__ == "__main__":
    main()
