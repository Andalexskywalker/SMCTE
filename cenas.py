# seq_quick_benchmark.py — 6 modelos sequenciais + ensemble Top-3 + export dos 3 melhores
# Entradas: .npz com X(N,T,F), y(N,), mask(N,T), feature_names, seq_len, event_ids
# Saídas: modelos em models_seq/, relatórios e figuras em reports/seq_models_quick[/<RUN_TAG>]

import os, json, warnings, argparse, random
from pathlib import Path
import numpy as np
import pandas as pd

# matplotlib sem GUI
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers as L

from sklearn.metrics import (
    average_precision_score, roc_auc_score, precision_recall_curve,
    precision_score, recall_score, fbeta_score, balanced_accuracy_score,
    confusion_matrix
)

# ------------------------- Utils -------------------------
def set_seeds(seed=42):
    random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True); return p

def load_npz(path):
    z = np.load(path, allow_pickle=True)
    X, y, M = z["X"], z["y"], z["mask"]
    feat_names = z["feature_names"].tolist()
    T = int(z["seq_len"])
    return X, y.astype(int), M.astype(np.float32), feat_names, T

def class_weights(y):
    # balanceamento simples por frequência
    pos = max(1e-9, float((y == 1).mean()))
    return {0: 1.0, 1: max(1.0, (1.0 - pos) / pos)}

def best_threshold_fbeta(y_true, y_prob, beta=2.0):
    p, r, thr = precision_recall_curve(y_true, y_prob)
    f = (1+beta**2) * (p*r) / (beta**2*p + r + 1e-12)
    idx = int(np.nanargmax(f))
    thr_opt = float(thr[max(idx-1, 0)]) if len(thr) else 0.5  # thr len = len(p)-1
    return thr_opt, float(f[idx])

def metrics_at(y_true, y_prob, thr):
    y_pred = (y_prob >= thr).astype(int)
    has_both = len(np.unique(y_true)) > 1
    return dict(
        pr_auc = float(average_precision_score(y_true, y_prob)) if has_both else float("nan"),
        roc_auc= float(roc_auc_score(y_true, y_prob)) if has_both else float("nan"),
        f2     = float(fbeta_score(y_true, y_pred, beta=2, zero_division=0)),
        recall = float(recall_score(y_true, y_pred, zero_division=0)),
        precision = float(precision_score(y_true, y_pred, zero_division=0)),
        balanced_acc = float(balanced_accuracy_score(y_true, y_pred))
    )

def save_pr_curve(y_true, y_prob, title, out_png):
    p, r, _ = precision_recall_curve(y_true, y_prob)
    plt.figure(figsize=(5,4))
    plt.plot(r, p)
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(title)
    plt.grid(True, ls=":")
    plt.tight_layout(); plt.savefig(out_png, dpi=160); plt.close()

def save_confusion(y_true, y_pred, title, out_png):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4.2,3.8))
    plt.imshow(cm)
    for (i,j),v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center")
    plt.xticks([0,1], ["0","1"]); plt.yticks([0,1], ["0","1"])
    plt.xlabel("Pred"); plt.ylabel("True"); plt.title(title)
    plt.tight_layout(); plt.savefig(out_png, dpi=160); plt.close()

def save_history_hist(hist, out_png, title="Training"):
    h = hist.history
    plt.figure(figsize=(5.2,3.8))
    if "loss" in h: plt.plot(h["loss"], label="loss")
    if "val_loss" in h: plt.plot(h["val_loss"], label="val_loss")
    # se não houver métricas adicionais, mostramos só loss
    plt.legend(); plt.title(title); plt.xlabel("epoch")
    plt.grid(alpha=0.3, ls=":")
    plt.tight_layout(); plt.savefig(out_png, dpi=160); plt.close()

# ------------------------- Model zoo (6) -------------------------
def build_model(name: str, T: int, F: int, dropout=0.3):
    inp = L.Input(shape=(T, F), name="seq")
    mask_in = L.Input(shape=(T,), name="mask")  # 1=real, 0=pad

    # Evita Lambda (compatível com Keras 3)
    mexp = L.Reshape((T, 1), name="mask_reshaped")(mask_in)
    x = L.Multiply(name="apply_mask")([inp, mexp])

    if name == "lstm":
        x = L.LSTM(64, return_sequences=True)(x)
        x = L.LSTM(32)(x)
    elif name == "gru":
        x = L.GRU(64, return_sequences=True)(x)
        x = L.GRU(32)(x)
    elif name == "bilstm":
        x = L.Bidirectional(L.LSTM(64, return_sequences=True))(x)
        x = L.Bidirectional(L.LSTM(32))(x)
    elif name == "bigru":
        x = L.Bidirectional(L.GRU(64, return_sequences=True))(x)
        x = L.Bidirectional(L.GRU(32))(x)
    elif name == "cnn1d":
        x = L.Conv1D(64, 5, padding="same")(x); x = L.ReLU()(x)
        x = L.Conv1D(64, 3, padding="same")(x); x = L.ReLU()(x)
        x = L.GlobalMaxPooling1D()(x)
    elif name == "tcn":
        def tcn_block(xi, f, k, d):
            y = L.Conv1D(f, k, padding="causal", dilation_rate=d)(xi)
            y = L.BatchNormalization()(y); y = L.ReLU()(y); y = L.Dropout(dropout)(y)
            y = L.Conv1D(f, k, padding="causal", dilation_rate=d)(y)
            y = L.BatchNormalization()(y); y = L.ReLU()(y)
            res = xi if xi.shape[-1] == f else L.Conv1D(f, 1, padding="same")(xi)
            return L.Add()([res, y])
        x = tcn_block(x, 64, 3, 1)
        x = tcn_block(x, 64, 3, 2)
        x = tcn_block(x, 64, 3, 4)
        x = tcn_block(x, 64, 3, 8)
        x = L.GlobalAveragePooling1D()(x)
    else:
        raise ValueError(f"Modelo desconhecido: {name}")

    # Se ainda for 3D (batch, T, F), pool para vetor
    if len(x.shape) == 3:
        x = L.GlobalAveragePooling1D()(x)

    x = L.Dropout(dropout)(x)
    out = L.Dense(1, activation="sigmoid")(x)
    model = keras.Model([inp, mask_in], out, name=name)
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="binary_crossentropy")
    return model

# ------------------------- Treino + avaliação -------------------------
def train_and_eval(name, Xtr, ytr, Mtr, Xva, yva, Mva, out_dir: Path, seed=42, epochs=80, batch=256):
    set_seeds(seed)
    T, F = Xtr.shape[1], Xtr.shape[2]
    model = build_model(name, T, F)

    cw = class_weights(ytr)
    cb = [
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3, verbose=0),
        keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True, verbose=0),
    ]
    hist = model.fit(
        [Xtr, Mtr], ytr,
        validation_data=([Xva, Mva], yva),
        epochs=epochs, batch_size=batch, verbose=0,
        class_weight=cw, callbacks=cb
    )

    # Probabilidades e threshold ótimo por F2 na validação
    p_va = model.predict([Xva, Mva], verbose=0).ravel()
    thr_opt, _ = best_threshold_fbeta(yva, p_va, beta=2.0)
    yhat_va = (p_va >= thr_opt).astype(int)
    mets = metrics_at(yva, p_va, thr_opt)

    # Artefactos
    ensure_dir(out_dir)
    save_history_hist(hist, out_dir / f"train_curve_{name}.png", title=f"Treino — {name}")
    save_pr_curve(yva, p_va, f"PR — {name}", out_dir / f"pr_{name}_val.png")
    save_confusion(yva, yhat_va, f"CM — {name} (thr={thr_opt:.3f})", out_dir / f"cm_{name}_val.png")

    info = dict(model=name, thr_opt=thr_opt, **mets)
    with open(out_dir / f"val_metrics_{name}.json","w") as f:
        json.dump(info, f, indent=2)
    model_path = out_dir / f"{name}.keras"
    model.save(model_path)
    return info, str(model_path)

# ------------------------- Main -------------------------
def main():
    ap = argparse.ArgumentParser(description="Quick benchmark seq (6 modelos) + ensemble Top-3 + export")
    ap.add_argument("--train_npz", required=True)
    ap.add_argument("--val_npz", required=True)
    ap.add_argument("--test_npz", default=None)
    ap.add_argument("--out_base", default="reports/seq_models_quick")
    ap.add_argument("--models_dir", default="models_seq")
    ap.add_argument("--run_tag", default="", help="Subpasta para separar execuções (ex.: top30)")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--which", nargs="*", default=["lstm","gru","bilstm","bigru","cnn1d","tcn"])
    args = ap.parse_args()

    set_seeds(args.seed)

    # paths
    OUT_BASE = ensure_dir(Path(args.out_base) / (args.run_tag if args.run_tag else ""))
    ART = ensure_dir(OUT_BASE / "artifacts")
    MODELS = ensure_dir(Path(args.models_dir) / (args.run_tag if args.run_tag else ""))

    # dados
    Xtr, ytr, Mtr, feats, T = load_npz(args.train_npz)
    Xva, yva, Mva, feats2, T2 = load_npz(args.val_npz)
    assert feats == feats2 and T == T2, "train e val incompatíveis"
    has_test = bool(args.test_npz and os.path.exists(args.test_npz))
    if has_test:
        Xte, yte, Mte, feats3, T3 = load_npz(args.test_npz)
        assert feats == feats3 and T == T3, "val e test incompatíveis"

    # treino + avaliação por modelo
    results = []
    saved = {}
    for name in args.which:
        print(f"[{name}] training…")
        info, mp = train_and_eval(name, Xtr, ytr, Mtr, Xva, yva, Mva, ART, seed=args.seed,
                                  epochs=args.epochs, batch=args.batch)
        info["model_path_val"] = mp
        results.append(info); saved[name] = mp

    # ordenar por PR-AUC (val), depois F2, e salvar CSV
    res_df = sorted(results, key=lambda d: (d["pr_auc"], d["f2"]), reverse=True)
    pd.DataFrame(res_df).to_csv(OUT_BASE/"baseline_results.csv", index=False)

    # barras comparativas (ainda sem ensemble; vamos re-gerar depois)
    def _bar(metric, out_png, rows):
        xs = [r["model"] for r in rows][::-1]
        ys = [r[metric] for r in rows][::-1]
        plt.figure(figsize=(7, max(3, 0.35*len(xs))))
        plt.barh(xs, ys)
        plt.xlabel(metric.upper()); plt.title(f"{metric.upper()} por modelo (val)")
        plt.grid(axis="x", alpha=0.2, ls=":")
        plt.tight_layout(); plt.savefig(out_png, dpi=200); plt.close()
    _bar("pr_auc", OUT_BASE/"baseline_pr_auc.png", res_df)
    _bar("f2", OUT_BASE/"baseline_f2.png", res_df)

    # avaliar em TEST com o threshold de validação e gerar PR/CM (provas)
    if has_test:
        print("[test] avaliação nos 6 modelos…")
        rows_test = []
        for r in res_df:
            name = r["model"]; thr = r["thr_opt"]
            model = keras.models.load_model(saved[name])
            pte = model.predict([Xte, Mte], verbose=0).ravel()
            ypred = (pte >= thr).astype(int)
            mets_t = metrics_at(yte, pte, thr)
            rows_test.append(dict(model=name, **mets_t))
            save_pr_curve(yte, pte, f"PR — {name} (test)", ART/f"pr_{name}_test.png")
            save_confusion(yte, ypred, f"CM — {name} (test)", ART/f"cm_{name}_test.png")
        pd.DataFrame(rows_test).to_csv(OUT_BASE/"test_results.csv", index=False)

    # selecionar Top-3 por PR-AUC (val) e exportar
    top3 = res_df[:3]
    exported = []
    for rank, r in enumerate(top3, start=1):
        name = r["model"]
        src = saved[name]
        dst = MODELS / f"top{rank}_{name}.keras"
        mdl = keras.models.load_model(src)
        mdl.save(dst)
        exported.append(dict(rank=rank, model=name, path=str(dst.as_posix()),
                             thr_opt=r["thr_opt"], pr_auc=r["pr_auc"], f2=r["f2"]))

    with open(MODELS/"top3_models.json","w") as f:
        json.dump(dict(selected_by="PR-AUC (val)", models=exported,
                       artifacts=str(ART.as_posix()),
                       baseline_csv=str((OUT_BASE/'baseline_results.csv').as_posix())), f, indent=2)

    # ===== Ensemble Top-3 (média das probabilidades) =====
    top_names = [r["model"] for r in res_df[:3]]
    print("\n[ensemble] modelos:", top_names)

    # --- VAL ---
    probs_va = np.zeros(len(yva), dtype=np.float32)
    for n in top_names:
        mdl = keras.models.load_model(saved[n])
        probs_va += mdl.predict([Xva, Mva], verbose=0).ravel().astype(np.float32)
    probs_va /= len(top_names)
    thr_e, _ = best_threshold_fbeta(yva, probs_va, beta=2.0)
    ens_val = dict(model="ensemble_top3", **metrics_at(yva, probs_va, thr_e))
    save_pr_curve(yva, probs_va, "PR — ensemble_top3 (val)", ART/"pr_ensemble_val.png")
    save_confusion(yva, (probs_va>=thr_e).astype(int), f"CM — ensemble_top3 (thr={thr_e:.3f})", ART/"cm_ensemble_val.png")

    # atualiza baseline_results.csv incluindo o ensemble
    base_csv = OUT_BASE/"baseline_results.csv"
    base_df = pd.read_csv(base_csv) if base_csv.exists() else pd.DataFrame(res_df)
    base_df = pd.concat([base_df, pd.DataFrame([ens_val])], ignore_index=True)
    base_df.to_csv(base_csv, index=False)

    # --- TEST (se existir) ---
    if has_test:
        probs_te = np.zeros(len(yte), dtype=np.float32)
        for n in top_names:
            mdl = keras.models.load_model(saved[n])
            probs_te += mdl.predict([Xte, Mte], verbose=0).ravel().astype(np.float32)
        probs_te /= len(top_names)
        ens_test = dict(model="ensemble_top3", **metrics_at(yte, probs_te, thr_e))
        save_pr_curve(yte, probs_te, "PR — ensemble_top3 (test)", ART/"pr_ensemble_test.png")
        save_confusion(yte, (probs_te>=thr_e).astype(int), f"CM — ensemble_top3 (thr={thr_e:.3f})", ART/"cm_ensemble_test.png")
        test_csv = OUT_BASE/"test_results.csv"
        test_df = pd.read_csv(test_csv) if test_csv.exists() else pd.DataFrame()
        test_df = pd.concat([test_df, pd.DataFrame([ens_test])], ignore_index=True)
        test_df.to_csv(test_csv, index=False)

    # (re)gerar barras já com o ensemble incluído
    base_df = pd.read_csv(base_csv)
    for metric, outf in [("pr_auc", OUT_BASE/"baseline_pr_auc.png"),
                         ("f2", OUT_BASE/"baseline_f2.png")]:
        xs = base_df["model"].astype(str).tolist()[::-1]
        ys = base_df[metric].astype(float).tolist()[::-1]
        plt.figure(figsize=(7, max(3, 0.35*len(xs))))
        plt.barh(xs, ys)
        plt.xlabel(metric.upper()); plt.title(f"{metric.upper()} por modelo (val)")
        plt.grid(axis="x", alpha=0.2, ls=":")
        plt.tight_layout(); plt.savefig(outf, dpi=200); plt.close()

    # --- prints finais ---
    print("\nTop-3 (val, por PR-AUC):")
    for e in exported:
        print(f"  #{e['rank']} {e['model']}  PR-AUC={e['pr_auc']:.4f}  F2={e['f2']:.4f} → {e['path']}")
    print("\n[OK] Artefactos principais:")
    print(" -", OUT_BASE/"baseline_results.csv")
    if has_test: print(" -", OUT_BASE/"test_results.csv")
    print(" -", OUT_BASE/"baseline_pr_auc.png")
    print(" -", OUT_BASE/"baseline_f2.png")
    print(" -", ART/"train_curve_<model>.png, pr_<model>_val.png, cm_<model>_val.png")
    if has_test: print(" -", ART/"pr_<model>_test.png, cm_<model>_test.png")
    print(" -", MODELS/"top3_models.json")
    print(" - Ensemble figs:", ART/"pr_ensemble_val.png", ART/"cm_ensemble_val.png",
          ("|" + str(ART/'pr_ensemble_test.png') + " " + str(ART/'cm_ensemble_test.png')) if has_test else "")

if __name__ == "__main__":
    main()
