import argparse, os, pandas as pd, numpy as np

def find_target_col(df, user_col=None):
    if user_col and user_col in df.columns: return user_col
    cand = ["risk","y","label","target","class","is_positive","pos","y_event"]
    cols_lower = {c.lower(): c for c in df.columns}
    for c in cand:
        if c in cols_lower: return cols_lower[c]
    # fallback: tentar 2ª coluna se binária
    if len(df.columns) > 1:
        c = df.columns[1]
        vals = pd.to_numeric(df[c], errors="coerce").dropna().unique()
        if set(np.unique(vals)).issubset({0,1}): return c
    raise SystemExit(f"❌ Não encontrei coluna de target. Colunas: {list(df.columns)}")

def count_stats(path, part_name, target_col=None):
    if not os.path.exists(path): return None
    df = pd.read_csv(path)
    tcol = find_target_col(df, target_col)
    s = pd.to_numeric(df[tcol], errors="coerce")
    # regra positiva robusta: se binário → ==1; caso contrário → > 0
    uniq = set(np.unique(s.dropna()))
    if uniq.issubset({0,1}): pos = int((s==1).sum())
    else:                     pos = int((s>0).sum())
    n = int(len(df)); p = (pos/n*100.0) if n else 0.0
    return {"Partição": part_name, "N linhas": n, "N positivos": pos, "% positivos": round(p, 3)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val",   required=True)
    ap.add_argument("--test",  required=True)
    ap.add_argument("--out",   required=True)
    ap.add_argument("--target", default=None, help="(opcional) nome exacto da coluna target")
    args = ap.parse_args()

    rows = []
    for part, path in [("train", args.train), ("val", args.val), ("test", args.test)]:
        st = count_stats(path, part, args.target)
        if st: rows.append(st)

    if not rows: raise SystemExit("❌ Nenhum CSV encontrado.")
    df = pd.DataFrame(rows, columns=["Partição","N linhas","N positivos","% positivos"])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    # print bonitinho para copy-paste
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()
