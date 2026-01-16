import argparse, os, numpy as np, pandas as pd

def ensure_dir(png_path: str):
    os.makedirs(os.path.dirname(png_path), exist_ok=True)

def from_probs_csv(probs_csv: str, out_png: str, title: str | None = None):
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, average_precision_score
    df = pd.read_csv(probs_csv)

    def pick(*cands):
        cols_lower = {c.lower(): c for c in df.columns}
        for c in cands:
            if c in df.columns: return c
            if c.lower() in cols_lower: return cols_lower[c.lower()]
        return None

    y_col = pick("y_true", "y")
    s_col = pick("y_prob", "score", "prob", "p1")
    if y_col is None or s_col is None:
        raise SystemExit("CSV precisa colunas: y_true,y_prob (ou y,score/prob).")
    y_true = df[y_col].to_numpy().astype(int)
    y_prob = df[s_col].to_numpy().astype(float)

    p, r, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    ensure_dir(out_png)
    plt.figure(figsize=(6,5))
    plt.plot(r, p)
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(title or f"Curva PR — AP={ap:.3f}")
    plt.grid(True, ls=":", alpha=0.3)
    plt.tight_layout(); plt.savefig(out_png, dpi=220); plt.close()
    print(f"[OK] PR-curve salva em {out_png} (AP={ap:.3f})")

def load_npz(npz_path: str):
    data = np.load(npz_path, allow_pickle=True)
    X = data["X"]; y = data["y"]
    M = data["mask"] if "mask" in data.files else None
    return X, y, M

def _align_mask_shape(model, M: np.ndarray | None):
    """Adapta a máscara ao segundo input do modelo: (N,T) -> (N,T) ou (N,T,1) ou (N,T,C)."""
    if M is None:
        return None
    try:
        inp = model.inputs[1]
        rank = len(inp.shape)
        if rank == 2:                    # (None, T)
            return M.astype("float32")
        if rank == 3:                    # (None, T, C)
            last = inp.shape[-1]
            if last is None or int(last) == 1:
                return M[..., None].astype("float32")
            return np.repeat(M[..., None], int(last), axis=-1).astype("float32")
        return M.astype("float32")
    except Exception:
        return M.astype("float32")

def from_keras_model(model_path: str, val_npz: str, out_png: str, title: str | None = None):
    import tensorflow as tf
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, average_precision_score

    X, y_true, M = load_npz(val_npz)
    model = tf.keras.models.load_model(model_path, compile=False)
    nin = len(model.inputs)

    if nin == 1:
        y_prob = model.predict(X, verbose=0).reshape(-1)
    else:
        # construir máscara se não existir no npz
        if M is None:
            # 1 onde a janela tem dados (não-zero) e 0 onde é padding
            M = (np.abs(X).sum(axis=-1) > 0).astype("float32")
        M_aligned = _align_mask_shape(model, M)
        inputs = [X, M_aligned]
        # se o modelo tiver >2 inputs, preenche extras com zeros compatíveis
        for k in range(2, nin):
            shp = [d if d is not None else (X.shape[0] if i==0 else 1) for i, d in enumerate(model.inputs[k].shape)]
            dummy = np.zeros(shp, dtype="float32")
            inputs.append(dummy)
        y_prob = model.predict(inputs, verbose=0).reshape(-1)

    p, r, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    ensure_dir(out_png)
    plt.figure(figsize=(6,5))
    plt.plot(r, p)
    plt.xlabel("Recall"); plt.ylabel("Precision")
    name = os.path.splitext(os.path.basename(model_path))[0]
    plt.title(title or f"{name} — PR (val) AP={ap:.3f}")
    plt.grid(True, ls=":", alpha=0.3)
    plt.tight_layout(); plt.savefig(out_png, dpi=220); plt.close()
    print(f"[OK] PR-curve salva em {out_png} (AP={ap:.3f})")

def from_baseline_and_dir(baseline_csv: str, models_dir: str, val_npz: str, out_png: str, title: str | None = None):
    df = pd.read_csv(baseline_csv)
    if "pr_auc" not in df.columns or "model" not in df.columns:
        raise SystemExit("baseline CSV precisa colunas: model, pr_auc.")
    row = df.sort_values("pr_auc", ascending=False).iloc[0]
    model_name = str(row["model"])
    candidates = [os.path.join(models_dir, model_name + ext) for ext in (".keras",".h5",".hdf5",".model")]
    model_path = next((c for c in candidates if os.path.exists(c)), None)
    if model_path is None:
        raise SystemExit(f"Modelo '{model_name}' não encontrado em {models_dir} (tente --model_path).")
    print(f"[INFO] Best by PR-AUC: {model_name} → {model_path}")
    from_keras_model(model_path, val_npz, out_png, title=title or f"{model_name} — PR (val)")

def main():
    print("[mk_pr_curve] mask-support = ON")
    ap = argparse.ArgumentParser(description="Gerar Curvas PR (val) para o relatório.")
    ap.add_argument("--probs_csv", default=None, help="CSV com colunas y_true,y_prob (ou y,score)")
    ap.add_argument("--model_path", default=None, help="Caminho para o modelo Keras/H5")
    ap.add_argument("--val_npz", default=None, help="NPZ com X,y (+ opcional mask) para validação")
    ap.add_argument("--baseline_csv", default=None, help="baseline_results.csv (colunas model,pr_auc)")
    ap.add_argument("--models_dir", default=None, help="Pasta com ficheiros .keras/.h5")
    ap.add_argument("--out", required=True, help="PNG de saída")
    ap.add_argument("--title", default=None, help="Título opcional")
    args = ap.parse_args()

    if args.probs_csv:
        from_probs_csv(args.probs_csv, args.out, title=args.title); return
    if args.model_path and args.val_npz:
        from_keras_model(args.model_path, args.val_npz, args.out, title=args.title); return
    if args.baseline_csv and args.models_dir and args.val_npz:
        from_baseline_and_dir(args.baseline_csv, args.models_dir, args.val_npz, args.out, title=args.title); return

    raise SystemExit("Fornece: --probs_csv OU (--model_path & --val_npz) OU (--baseline_csv & --models_dir & --val_npz)")

if __name__ == "__main__":
    main()
