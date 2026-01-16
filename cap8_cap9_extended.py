
import os, re, json, argparse, pathlib, pandas as pd, numpy as np
import shutil

def find_first(paths):
    for p in paths:
        if p and os.path.exists(p): return p
    return None

def search_glob(root, pattern):
    root = pathlib.Path(root)
    hits = list(root.rglob(pattern))
    return str(hits[0]) if hits else None

def safe_read_csv(path):
    if not path or not os.path.exists(path): return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None

def fmt3(x):
    try: return f"{float(x):.3f}"
    except: return ""

def write_table(df, path, cols=None):
    if df is None or df.empty:
        return False
    if cols:
        df = df[[c for c in cols if c in df.columns]].copy()

    # formatar APENAS colunas numéricas (ou numéricas após coerção)
    TEXT_COLS = {
        "Arquitetura", "Partição",
        "Recall (todos/top-30)", "Precisão (todos/top-30)",
        "Observações (papel na operação)"
    }
    for c in df.columns:
        if c in TEXT_COLS:
            continue
        coerced = pd.to_numeric(df[c], errors="coerce")
        # só formata se houver pelo menos um número válido na coluna
        if coerced.notna().any():
            df[c] = coerced.map(lambda x: f"{x:.3f}" if pd.notna(x) else "")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    return True


def build_tab_9_4(val_all, val_top, test_all=None, test_top=None, out_csv="cap9_tab_9_4_com_vs_sem_selecao.csv"):
    def merge_pair(df_all, df_top, particao):
        if df_all is None or df_top is None or df_all.empty or df_top.empty:
            return None
        a = df_all.copy(); b = df_top.copy()
        a["Partição"] = particao; b["Partição"] = particao
        keep = ["model","pr_auc","f2","recall","precision","Partição"]
        a = a[keep].rename(columns={"model":"Arquitetura","pr_auc":"PR-AUC","f2":"F2","recall":"Recall","precision":"Precisão"})
        b = b[keep].rename(columns={"model":"Arquitetura","pr_auc":"PR-AUC (top-30)","f2":"F2 (top-30)","recall":"Recall (top-30)","precision":"Precisão (top-30)"})
        m = pd.merge(a, b, on=["Arquitetura","Partição"], how="inner")
        if m.empty: return None
        m["ΔF2"] = m["F2 (top-30)"] - m["F2"]
        m["ΔPR-AUC"] = m["PR-AUC (top-30)"] - m["PR-AUC"]
        m["Recall (todos/top-30)"] = m["Recall"].apply(lambda x: f"{float(x):.3f}") + " / " + m["Recall (top-30)"].apply(lambda x: f"{float(x):.3f}")
        m["Precisão (todos/top-30)"] = m["Precisão"].apply(lambda x: f"{float(x):.3f}") + " / " + m["Precisão (top-30)"].apply(lambda x: f"{float(x):.3f}")
        return m[["Arquitetura","Partição","F2","F2 (top-30)","ΔF2","PR-AUC","PR-AUC (top-30)","ΔPR-AUC","Recall (todos/top-30)","Precisão (todos/top-30)"]]

    chunks = []
    if val_all is not None and not val_all.empty and val_top is not None and not val_top.empty:
        chunks.append(merge_pair(val_all, val_top, "val"))
    if test_all is not None and not test_all.empty and test_top is not None and not test_top.empty:
        chunks.append(merge_pair(test_all, test_top, "test"))
    chunks = [c for c in chunks if c is not None and not c.empty]
    if not chunks:
        return None
    return pd.concat(chunks, axis=0, ignore_index=True)

def build_tab_9_5(val_all, val_ros, test_all=None, test_ros=None, out_csv="cap9_tab_9_5_delta_ros.csv"):
    def deltas(df_base, df_ros, particao):
        if df_base is None or df_ros is None or df_base.empty or df_ros.empty:
            return None
        a = df_base[["model","pr_auc","f2","recall","precision"]].copy()
        b = df_ros [["model","pr_auc","f2","recall","precision"]].copy()
        m = pd.merge(a, b, on="model", suffixes=("_base","_ros"))
        if m.empty: return None
        m["Arquitetura"] = m["model"]; m["Partição"] = particao
        m["ΔF2"] = m["f2_ros"] - m["f2_base"]
        m["ΔPR-AUC"] = m["pr_auc_ros"] - m["pr_auc_base"]
        m["ΔRecall"] = m["recall_ros"] - m["recall_base"]
        m["ΔPrecisão"] = m["precision_ros"] - m["precision_base"]
        return m[["Arquitetura","Partição","ΔF2","ΔPR-AUC","ΔRecall","ΔPrecisão"]]

    rows = []
    if val_all is not None and not val_all.empty and val_ros is not None and not val_ros.empty:
        rows.append(deltas(val_all, val_ros, "val"))
    if test_all is not None and not test_all.empty and test_ros is not None and not test_ros.empty:
        rows.append(deltas(test_all, test_ros, "test"))
    rows = [r for r in rows if r is not None and not r.empty]
    if not rows:
        return None
    return pd.concat(rows, axis=0, ignore_index=True)


def build_tab_9_7(cac_csv, out_csv="cap9_tab_9_7_cac_finalistas.csv"):
    df = safe_read_csv(cac_csv)
    if df is None or df.empty:
        return None

    def pick(*cands):
        cols_lower = {c.lower(): c for c in df.columns}
        for c in cands:
            if c in cols_lower: return cols_lower[c]
        return None

    col_model = pick("model","architecture","arch","name")
    col_part  = pick("particao","partition","split","fold","part")
    col_mse   = pick("weighted_mse_highrisk","weighted_mse_high_risk","w_mse_highrisk",
                     "mse_highrisk","mse_high_risk")
    col_f2    = pick("f2","f2_score")
    col_cac   = pick("cac","cost_of_avoidance","avoidance_cost","cost")

    mapping = {}
    if col_model: mapping[col_model] = "Arquitetura"
    if col_part:  mapping[col_part]  = "Partição"
    if col_mse:   mapping[col_mse]   = "Weighted MSE @ alto risco"
    if col_f2:    mapping[col_f2]    = "F2"
    if col_cac:   mapping[col_cac]   = "CAC (↓)"

    if not mapping:
        return None

    t = df[list(mapping.keys())].rename(columns=mapping)

    # formatar só as métricas numéricas (write_table já trata, mas por segurança)
    for c in ["Weighted MSE @ alto risco","F2","CAC (↓)"]:
        if c in t.columns:
            t[c] = pd.to_numeric(t[c], errors="coerce")

    return t


def build_tab_9_8(baseline_val_csv, top_json, out_csv="cap9_tab_9_8_resumo_finalistas.csv"):
    df = safe_read_csv(baseline_val_csv)
    if df is None or df.empty: return None
    # carregar top3/topN
    finalists = None
    if top_json and os.path.exists(top_json):
        with open(top_json) as f:
            j = json.load(f)
        # aceitar lista simples ["gru","tcn",...] ou lista de dicts [{"model":"gru"},...]
        if isinstance(j, list):
            if j and isinstance(j[0], dict) and "model" in j[0]:
                finalists = [d["model"] for d in j]
            else:
                finalists = j
    # filtrar
    if finalists:
        df = df[df["model"].isin(finalists)].copy()
    # selecionar colunas
    cols = {"model":"Arquitetura", "pr_auc":"PR-AUC (val)", "f2":"F2 (val)", "recall":"Recall", "precision":"Precisão"}
    keep = [c for c in cols.keys() if c in df.columns]
    t = df[keep].rename(columns=cols)
    for c in ["PR-AUC (val)","F2 (val)","Recall","Precisão"]:
        if c in t.columns: t[c] = t[c].apply(fmt3)
    # Observações vazias para preencher no Word
    t["Observações (papel na operação)"] = ""
    # ordenar por PR-AUC ou F2 se existirem
    sort_by = "PR-AUC (val)" if "PR-AUC (val)" in t.columns else ("F2 (val)" if "F2 (val)" in t.columns else None)
    if sort_by:
        t = t.sort_values(sort_by, ascending=False)
    return t

def top_k_with_rank(csv_path, value_col=None, k=15, rename_cols=("Canal","Score","Rank")):
    df = safe_read_csv(csv_path)
    if df is None or df.empty: 
        return None

    # 1) Descobrir a coluna de "feature"
    cand_feat = ["feature","canal","channel","name","variable","var","feat","col"]
    feat_col = None
    cols_lower = {c.lower(): c for c in df.columns}
    for c in cand_feat:
        if c in cols_lower:
            feat_col = cols_lower[c]; break
    if feat_col is None:
        feat_col = df.columns[0]  # fallback: 1ª coluna

    # 2) Descobrir a coluna da métrica (auto)
    # Se value_col vier e existir (case-insensitive), usamos; senão tentamos candidatos comuns
    metric_col = None
    if value_col:
        lc = value_col.lower()
        if lc in cols_lower: 
            metric_col = cols_lower[lc]
    if metric_col is None:
        # superset de nomes usuais (AUC PR/ROC, AP, MI, perm-importance, delta, score, etc.)
        cand_metric = [
            "auc","pr_auc","ap","aupr","average_precision",
            "roc_auc","roc-auc","auc_roc",
            "mi","mutual_info","mutual_information",
            "perm_importance","permutation_importance","delta_metric","delta",
            "importance","score","metric"
        ]
        for c in cand_metric:
            if c in cols_lower:
                metric_col = cols_lower[c]; break
    if metric_col is None:
        # último recurso: tenta a 2ª coluna
        if len(df.columns) > 1:
            metric_col = df.columns[1]
        else:
            return None

    # 3) Selecionar, ordenar (desc) e ranquear
    t = df[[feat_col, metric_col]].dropna().copy()
    # tentar converter para float
    t[metric_col] = pd.to_numeric(t[metric_col], errors="coerce")
    t = t.dropna(subset=[metric_col])
    t = t.sort_values(metric_col, ascending=False).head(k)
    t["Rank"] = np.arange(1, len(t)+1)
    t = t.rename(columns={feat_col: rename_cols[0], metric_col: rename_cols[1]})
    # formatar a métrica
    t[rename_cols[1]] = t[rename_cols[1]].apply(lambda x: f"{float(x):.3f}")
    return t


def copy_if_exists(src, dst):
    if src and os.path.exists(src):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        return True
    return False

# ======= Proc splits (train/val/test) =======
def physical_sanity(df):
    if "miss_distance" in df.columns:
        df.loc[df["miss_distance"] < 0, "miss_distance"] = 0.0
    if "relative_speed" in df.columns:
        df.loc[df["relative_speed"] <= 0, "relative_speed"] = np.nan

def role_median_impute(train_df, df):
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

def ffill_by_event(df, event_col):
    if {event_col, "time_to_tca"} <= set(df.columns):
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        df.sort_values([event_col, "time_to_tca"], ascending=[True, False], inplace=True)
        df[num_cols] = df.groupby(event_col, group_keys=False)[num_cols].ffill()
        df.sort_index(inplace=True)

def split_by_event(df, event_col, test_size=0.2, val_size=0.1, random_state=42, stratify_events=True, target_col="risk", positive_threshold=-6.0):
    from sklearn.model_selection import train_test_split
    # label por evento para estratificar
    def event_label_series(df):
        g = df.sort_values([event_col, "time_to_tca"]).groupby(event_col, sort=True)[target_col]
        uniq = set(pd.unique(df[target_col].dropna()))
        is_binary = uniq.issubset({0, 1})
        s = g.max()
        return (s >= (1 if is_binary else positive_threshold)).astype(int)
    if stratify_events:
        ev_label = event_label_series(df)
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

def export_proc_splits(raw_path, out_dir, event_col="event_id", target_col="risk", positive_threshold=-6.0, stratify_events=True, test_size=0.2, val_size=0.1, random_state=42):
    import pandas as pd, os
    df = pd.read_csv(raw_path) if raw_path.endswith(".csv") else pd.read_parquet(raw_path)
    physical_sanity(df)
    train_df, val_df, test_df = split_by_event(df, event_col, test_size, val_size, random_state, stratify_events, target_col, positive_threshold)
    for part in (train_df, val_df, test_df): physical_sanity(part)
    role_median_impute(train_df, train_df); ffill_by_event(train_df, event_col)
    role_median_impute(train_df, val_df);   ffill_by_event(val_df, event_col)
    role_median_impute(train_df, test_df);  ffill_by_event(test_df, event_col)
    os.makedirs(out_dir, exist_ok=True)
    train_df.to_csv(os.path.join(out_dir, "train_proc.csv"), index=False)
    val_df.to_csv(os.path.join(out_dir, "val_proc.csv"), index=False)
    test_df.to_csv(os.path.join(out_dir, "test_proc.csv"), index=False)
    # tabela 8.1 (se quiseres): prevalência por partição
    def pos_count(d):
        if target_col in d.columns and pd.api.types.is_numeric_dtype(d[target_col]):
            return int((d[target_col] >= positive_threshold).sum())
        return int((d.get("y", pd.Series([0]*len(d)))==1).sum())
    rows = []
    for name, d in (("Treino", train_df), ("Validação", val_df), ("Teste", test_df)):
        n = len(d); pos = pos_count(d); pct = round(100*pos/n, 2) if n else 0.0
        rows.append({"Partição": name, "N linhas": n, "N positivos": pos, "% positivos": pct})
    tab81 = pd.DataFrame(rows, columns=["Partição","N linhas","N positivos","% positivos"])
    tab81.to_csv(os.path.join(out_dir, "cap8_tab_8_1_particoes_prevalencia.csv"), index=False)
    return os.path.join(out_dir, "train_proc.csv"), os.path.join(out_dir, "val_proc.csv"), os.path.join(out_dir, "test_proc.csv")

def main():
    ap = argparse.ArgumentParser(description="Gera Tabelas/Figuras dos Cap. 8 e 9 a partir dos artefactos dos modelos.")
    ap.add_argument("--root", default=".", help="raiz onde procurar relatórios (por ex.: projeto/)")
    ap.add_argument("--outdir", default="cap8_cap9_out", help="pasta de saída")
    # caminhos explícitos (opcionais)
    ap.add_argument("--all_val_csv", default=None, help="reports/*/baseline_results.csv (val ALL)")
    ap.add_argument("--top_val_csv", default=None, help="runs top-30 baseline_results.csv (val top-30)")
    ap.add_argument("--ros_val_csv", default=None, help="runs ROS baseline_results.csv (val ROS)")
    ap.add_argument("--all_test_csv", default=None, help="reports/*/test_results.csv (test ALL)")
    ap.add_argument("--top_test_csv", default=None, help="runs top-30 test_results.csv (test top-30)")
    ap.add_argument("--ros_test_csv", default=None, help="runs ROS test_results.csv (test ROS)")
    ap.add_argument("--cac_csv", default=None, help="reports/models_seq/cac_results.csv")
    ap.add_argument("--finalists_json", default=None, help="models_seq/top3_models.json")
    ap.add_argument("--baseline_val_for_finalists", default=None, help="reports/seq_models_quick/baseline_results.csv")
    ap.add_argument("--univar_auc_csv", default=None, help="cenas_satelites/univariate_auc.csv")
    ap.add_argument("--mutual_info_csv", default=None, help="cenas_satelites/mutual_info_all.csv")
    ap.add_argument("--perm_import_csv", default=None, help="cenas_satelites/perm_importance_hgb.csv")
    ap.add_argument("--prcurve_best_none_val", default=None, help="reports/models_seq/artifacts/prcurve_best_seq_none_val.png")
    ap.add_argument("--pr_best_after_hpo_val", default=None, help="reports/models_seq/artifacts/pr_best_seq_after_hpo_val.png")
    # proc splits
    ap.add_argument("--raw_dataset", default=None, help="CSV/Parquet bruto para gerar train/val/test_proc.csv e Tab 8.1")
    ap.add_argument("--proc_outdir", default=None, help="pasta para guardar train/val/test_proc.csv")

    args = ap.parse_args()
    outdir = pathlib.Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # ===== descobrir ficheiros se não foram passados =====
    val_all = safe_read_csv(find_first([args.all_val_csv,
                                        search_glob(args.root, "reports/models_seq/baseline_results.csv"),
                                        search_glob(args.root, "reports/seq_models_quick/baseline_results.csv")]))
    val_top = safe_read_csv(find_first([args.top_val_csv,
                                        search_glob(args.root, "runs/**/top*/*/baseline_results.csv"),
                                        search_glob(args.root, "**/top*/baseline_results.csv")]))
    val_ros = safe_read_csv(find_first([args.ros_val_csv,
                                        search_glob(args.root, "runs/**/ros*/**/baseline_results.csv"),
                                        search_glob(args.root, "**/ros*/baseline_results.csv")]))

    test_all = safe_read_csv(find_first([args.all_test_csv,
                                         search_glob(args.root, "reports/models_seq/test_results.csv"),
                                         search_glob(args.root, "reports/seq_models_quick/test_results.csv")]))
    test_top = safe_read_csv(find_first([args.top_test_csv,
                                         search_glob(args.root, "runs/**/top*/*/test_results.csv"),
                                         search_glob(args.root, "**/top*/test_results.csv")]))
    test_ros = safe_read_csv(find_first([args.ros_test_csv,
                                         search_glob(args.root, "runs/**/ros*/**/test_results.csv"),
                                         search_glob(args.root, "**/ros*/test_results.csv")]))

    # ===== Tabela 9.4 =====
    tab94 = build_tab_9_4(val_all, val_top, test_all, test_top)
    if tab94 is not None and not tab94.empty:
        write_table(tab94, str(outdir/"cap9_tab_9_4_com_vs_sem_selecao.csv"),
                    cols=["Arquitetura","Partição","F2","F2 (top-30)","ΔF2","PR-AUC","PR-AUC (top-30)","ΔPR-AUC","Recall (todos/top-30)","Precisão (todos/top-30)"])

    # ===== Tabela 9.5 =====
    tab95 = build_tab_9_5(val_all, val_ros, test_all, test_ros)
    if tab95 is not None and not tab95.empty:
        write_table(tab95, str(outdir/"cap9_tab_9_5_delta_ros.csv"),
                    cols=["Arquitetura","Partição","ΔF2","ΔPR-AUC","ΔRecall","ΔPrecisão"])

    # ===== Figuras 9.4 / 9.5 (copiar se existirem) =====
    pr_none = find_first([args.prcurve_best_none_val,
                          search_glob(args.root, "reports/models_seq/artifacts/prcurve_best_seq_none_val.png")])
    pr_hpo  = find_first([args.pr_best_after_hpo_val,
                          search_glob(args.root, "reports/models_seq/artifacts/pr_best_seq_after_hpo_val.png")])
    if pr_none:
        copy_if_exists(pr_none, str(outdir/"FIG9_4_prcurve_best_seq_none_val.png"))
    if pr_hpo:
        copy_if_exists(pr_hpo, str(outdir/"FIG9_5_pr_best_seq_after_hpo_val.png"))

    # ===== Tabela 9.7 CAC =====
    cac_csv = find_first([args.cac_csv, search_glob(args.root, "reports/models_seq/cac_results.csv")])
    tab97 = build_tab_9_7(cac_csv)
    if tab97 is not None and not tab97.empty:
        write_table(tab97, str(outdir/"cap9_tab_9_7_cac_finalistas.csv"),
                    cols=["Arquitetura","Partição","Weighted MSE @ alto risco","F2","CAC (↓)"])

    # ===== Tabela 9.8 finalistas =====
    baseline_val_for_finalists = find_first([args.baseline_val_for_finalists,
                                             search_glob(args.root, "reports/seq_models_quick/baseline_results.csv"),
                                             search_glob(args.root, "reports/models_seq/baseline_results.csv")])
    finalists_json = find_first([args.finalists_json,
                                 search_glob(args.root, "models_seq/top3_models.json"),
                                 search_glob(args.root, "models_seq/finalists.json")])
    tab98 = build_tab_9_8(baseline_val_for_finalists, finalists_json)
    if tab98 is not None and not tab98.empty:
        write_table(tab98, str(outdir/"cap9_tab_9_8_resumo_finalistas.csv"))

    # ===== Cap. 8: Tabelas 8.3 / 8.4 / 8.5 =====
    univar = find_first([args.univar_auc_csv, search_glob(args.root, "cenas_satelites/univariate_auc.csv")])
    mi     = find_first([args.mutual_info_csv, search_glob(args.root, "cenas_satelites/mutual_info_all.csv")])
    perm   = find_first([args.perm_import_csv, search_glob(args.root, "cenas_satelites/perm_importance_hgb.csv")])

    tab83 = top_k_with_rank(univar, value_col=None, k=15, rename_cols=("Canal","AUC","Rank"))
    if tab83 is not None:
        write_table(tab83, str(outdir/"cap8_tab_8_3_top15_auc.csv"))
    tab84 = top_k_with_rank(mi, value_col=None, k=15, rename_cols=("Canal","MI","Rank"))
    if tab84 is not None:
        write_table(tab84, str(outdir/"cap8_tab_8_4_top15_mi.csv"))
    tab85 = top_k_with_rank(perm, value_col=None, k=15, rename_cols=("Canal","Δ-métrica","Rank"))
    if tab85 is not None:
        write_table(tab85, str(outdir/"cap8_tab_8_5_top15_perm.csv"))

    # ===== Proc splits (train/val/test) =====
    if args.raw_dataset and args.proc_outdir:
        train_csv, val_csv, test_csv = export_proc_splits(args.raw_dataset, args.proc_outdir)
        print("OK: train/val/test_proc gravados em", args.proc_outdir)

    # Relatório final de ficheiros criados
    created = list(pathlib.Path(outdir).glob("*"))
    if created:
        print("[CRIADOS] Artefactos Cap.8/9:")
        for p in created:
            print(" -", p)
    else:
        print("[INFO] Nada criado. Verifica se os CSVs fonte existem ou passa caminhos explicitamente.")

if __name__ == "__main__":
    main()
