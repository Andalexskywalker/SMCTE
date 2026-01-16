
import argparse, os, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

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

def event_label_series(df: pd.DataFrame, event_col: str, target_col: str) -> pd.Series:
    g = df.sort_values([event_col, "time_to_tca"]).groupby(event_col, sort=True)[target_col]
    # binário → last; contínuo → max>0 vira positivo para estratificação simples
    uniq = set(pd.unique(df[target_col].dropna()))
    is_binary = uniq.issubset({0, 1})
    if is_binary:
        return g.last().astype(int)
    return (g.max() > 0).astype(int)

def count_stats(df: pd.DataFrame, target_col: str, part_name: str):
    n = int(len(df))
    s = pd.to_numeric(df[target_col], errors="coerce")
    uniq = set(np.unique(s.dropna()))
    pos = int((s == 1).sum()) if uniq.issubset({0,1}) else int((s > 0).sum())
    p = (pos/n*100.0) if n else 0.0
    return {"Partição": part_name, "N linhas": n, "N positivos": pos, "% positivos": round(p, 3)}

def autodetect(colnames, *cands, default=None):
    low = {c.lower(): c for c in colnames}
    for c in cands:
        if c in low: return low[c]
    return default

def main():
    ap = argparse.ArgumentParser(description="Faz proc a partir de train/test já existentes e cria Tabela 9.1")
    ap.add_argument("--train", required=True, help="CSV do train pré-existente")
    ap.add_argument("--test",  required=True, help="CSV do test pré-existente")
    ap.add_argument("--out_dir", required=True, help="Pasta onde salvar train_proc/val_proc/test_proc")
    ap.add_argument("--tab9_1_out", required=True, help="Caminho CSV da Tabela 9.1")
    ap.add_argument("--target_col", default=None, help="Nome do target (se vazio, auto)")
    ap.add_argument("--event_col", default=None, help="Nome do event_id (se vazio, auto)")
    ap.add_argument("--val_size", type=float, default=0.1, help="Fração do train que vira validação")
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--stratify_events", action="store_true", help="Estratificar por label ao nível do evento")
    args = ap.parse_args()

    # Ler
    tr = pd.read_csv(args.train)
    te = pd.read_csv(args.test)

    # Autodetect target/event se necessário
    tcol = args.target_col or autodetect(tr.columns, "risk","label","target","y","class","is_positive", default=None)
    ecol = args.event_col  or autodetect(tr.columns, "event_id","event","id_event","eventid", default=None)
    if tcol is None:
        raise SystemExit("❌ Não encontrei coluna target (tenta --target_col).")
    if ecol is None:
        raise SystemExit("❌ Não encontrei coluna de evento (tenta --event_col).")

    # Sanidade
    for d in (tr, te): physical_sanity(d)

    # Splitting: train→ train_part + val_part (por evento, opcionalmente estratificado)
    if args.stratify_events and ecol in tr.columns:
        ev_label = event_label_series(tr, ecol, tcol)
        ev_df = pd.DataFrame({ecol: ev_label.index, "y_ev": ev_label.values})
        tr_ev, va_ev = train_test_split(ev_df, test_size=args.val_size, random_state=args.random_state, stratify=ev_df["y_ev"])
        tr_ids, va_ids = tr_ev[ecol].values, va_ev[ecol].values
        tr_part = tr[tr[ecol].isin(tr_ids)].copy()
        va_part = tr[tr[ecol].isin(va_ids)].copy()
    else:
        tr_part, va_part = train_test_split(tr, test_size=args.val_size, random_state=args.random_state)

    # Imputação baseada só no train_part
    for d in (tr_part, va_part, te):
        physical_sanity(d)
    role_median_impute(tr_part, tr_part); ffill_by_event(tr_part, ecol)
    role_median_impute(tr_part, va_part); ffill_by_event(va_part, ecol)
    role_median_impute(tr_part, te);      ffill_by_event(te, ecol)

    # Salvar
    ensure_dir(args.out_dir)
    p_tr = os.path.join(args.out_dir, "train_proc.csv")
    p_va = os.path.join(args.out_dir, "val_proc.csv")
    p_te = os.path.join(args.out_dir, "test_proc.csv")
    tr_part.to_csv(p_tr, index=False); va_part.to_csv(p_va, index=False); te.to_csv(p_te, index=False)

    # Tabela 9.1
    ensure_dir(os.path.dirname(args.tab9_1_out))
    tab = pd.DataFrame([
        count_stats(tr_part, tcol, "train"),
        count_stats(va_part, tcol, "val"),
        count_stats(te,      tcol, "test"),
    ], columns=["Partição","N linhas","N positivos","% positivos"])
    tab.to_csv(args.tab9_1_out, index=False)

    # Print amigável
    print("👉 Gravado:")
    print(" -", p_tr)
    print(" -", p_va)
    print(" -", p_te)
    print("\nTabela 9.1 — Partições e prevalência por classe")
    print(tab.to_string(index=False))

if __name__ == "__main__":
    main()
