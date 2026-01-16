
import argparse, os, pandas as pd, numpy as np

def safe_read_csv(p):
    if p is None: return None
    if not os.path.exists(p): 
        print(f"[WARN] not found: {p}")
        return None
    try:
        return pd.read_csv(p)
    except Exception as e:
        print(f"[WARN] failed to read {p}: {e}")
        return None

def pick_col(df, candidates, default=None):
    if df is None: return None
    cols_lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in cols_lower: return cols_lower[c]
    return default

def standardize(df, arch=None, split=None, f2=None, pr=None, rec=None, prec=None, split_name=None):
    keep = {}
    if arch is None: arch = pick_col(df, ["architecture","arch","model","name"])
    if split is None: split = pick_col(df, ["partition","particao","split","fold","set","subset"])
    if f2   is None: f2   = pick_col(df, ["f2","f2_score"])
    if pr   is None: pr   = pick_col(df, ["pr_auc","ap","aupr","average_precision"])
    if rec  is None: rec  = pick_col(df, ["recall","recall_score","tpr"])
    if prec is None: prec = pick_col(df, ["precision","precision_score","ppv"])

    if arch is None or f2 is None or pr is None or rec is None or prec is None:
        return pd.DataFrame(columns=["Arquitetura","Partição","F2","PR-AUC","Recall","Precisão"])

    tmp = df.copy()
    tmp = tmp.rename(columns={arch:"Arquitetura",
                              f2:"F2", pr:"PR-AUC", rec:"Recall", prec:"Precisão"})
    # Partição
    if split is not None:
        tmp = tmp.rename(columns={split:"Partição"})
    else:
        tmp["Partição"] = split_name or "val"
    # garantir numéricos
    for c in ["F2","PR-AUC","Recall","Precisão"]:
        tmp[c] = pd.to_numeric(tmp[c], errors="coerce")
    # mantêm só colunas relevantes
    tmp = tmp[["Arquitetura","Partição","F2","PR-AUC","Recall","Precisão"]]
    return tmp

def compute_delta(none_df, bal_df, want_split):
    a = none_df[none_df["Partição"].str.lower()==want_split.lower()].copy()
    b = bal_df[ bal_df["Partição"].str.lower()==want_split.lower()].copy()
    if a.empty or b.empty:
        return pd.DataFrame(columns=["Arquitetura","Partição","ΔF2","ΔPR-AUC","ΔRecall","ΔPrecisão"])
    # merge by architecture
    m = pd.merge(a, b, on=["Arquitetura","Partição"], suffixes=("_none","_bal"))
    if m.empty:
        # try merge by architecture only, then set split
        m = pd.merge(a.drop(columns=["Partição"]), b.drop(columns=["Partição"]), on=["Arquitetura"], suffixes=("_none","_bal"))
        if m.empty:
            return pd.DataFrame(columns=["Arquitetura","Partição","ΔF2","ΔPR-AUC","ΔRecall","ΔPrecisão"])
        m["Partição"] = want_split
    # compute deltas
    out = pd.DataFrame({
        "Arquitetura": m["Arquitetura"],
        "Partição": m["Partição"],
        "ΔF2": m["F2_bal"] - m["F2_none"],
        "ΔPR-AUC": m["PR-AUC_bal"] - m["PR-AUC_none"],
        "ΔRecall": m["Recall_bal"] - m["Recall_none"],
        "ΔPrecisão": m["Precisão_bal"] - m["Precisão_none"],
    })
    return out.sort_values(["Partição","Arquitetura"]).reset_index(drop=True)

def main():
    ap = argparse.ArgumentParser(description="Tabela 9.5 — Variação com o balanceamento por arquitetura")
    ap.add_argument("--none_val_csv", required=False)  # baseline sem balanceamento (val)
    ap.add_argument("--none_test_csv", required=False) # baseline sem balanceamento (test)
    ap.add_argument("--bal_val_csv",  required=False)  # com balanceamento (val) — class weights, focal ou ROS
    ap.add_argument("--bal_test_csv", required=False)  # com balanceamento (test)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    val_none = standardize(safe_read_csv(args.none_val_csv), split_name="val")
    tst_none = standardize(safe_read_csv(args.none_test_csv), split_name="test")
    val_bal  = standardize(safe_read_csv(args.bal_val_csv),  split_name="val")
    tst_bal  = standardize(safe_read_csv(args.bal_test_csv), split_name="test")

    # concatenate by split
    none_all = pd.concat([val_none, tst_none], ignore_index=True)
    bal_all  = pd.concat([val_bal,  tst_bal ], ignore_index=True)

    # compute deltas per split
    parts = []
    for sp in ["val","test"]:
        parts.append(compute_delta(none_all, bal_all, sp))
    tab = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    # format numeric to 3 decimals when saving
    if not tab.empty:
        for c in ["ΔF2","ΔPR-AUC","ΔRecall","ΔPrecisão"]:
            tab[c] = tab[c].round(3)

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    tab.to_csv(args.out_csv, index=False)
    print(tab.to_string(index=False) if not tab.empty else "Tabela 9.5 vazia — verifica os CSVs/colunas.")

if __name__ == "__main__":
    main()
