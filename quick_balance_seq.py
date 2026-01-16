
"""
quick_balance_seq.py — treina/avalia modelos sequenciais (GRU/LSTM/BiGRU/BiLSTM/CNN1D/TCN)
com/sem balanceamento (ROS / class_weight / focal) e salva métricas em CSV.

Entradas: .npz com X (N,T,F), y (N,), mask (N,T). (o mesmo formato que tens)
Saídas: CSV com linhas por arquitetura e partição (val/test): F2, PR-AUC, Recall, Precisão.
"""

import os, argparse, json, math, random
import numpy as np
import pandas as pd

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow.keras import layers as L, Model, optimizers as opt, callbacks as cb, backend as K

from sklearn.metrics import average_precision_score, precision_score, recall_score, fbeta_score

# ---------- Utils ----------

def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)

def load_npz(path):
    d = np.load(path, allow_pickle=True)
    X = d["X"].astype("float32")
    y = d["y"].astype("float32")
    M = d["mask"].astype("float32")
    feat_names = d.get("feature_names", None)
    return X, y, M, feat_names

def oversample_ros(X, y, M):
    yb = (y > 0.5).astype(int)
    idx_pos = np.where(yb == 1)[0]
    idx_neg = np.where(yb == 0)[0]
    if len(idx_pos) == 0 or len(idx_neg) == 0:
        return X, y, M
    n_target = max(len(idx_pos), len(idx_neg))
    # sample with replacement to match n_target for both classes
    pos_sampled = np.random.choice(idx_pos, size=n_target, replace=True)
    neg_sampled = np.random.choice(idx_neg, size=n_target, replace=True)
    idx_new = np.concatenate([pos_sampled, neg_sampled])
    np.random.shuffle(idx_new)
    return X[idx_new], y[idx_new], M[idx_new]

def compute_class_weight(y):
    yb = (y > 0.5).astype(int)
    n = len(yb)
    n0 = max(1, int((yb == 0).sum()))
    n1 = max(1, int((yb == 1).sum()))
    w0 = n / (2.0 * n0)
    w1 = n / (2.0 * n1)
    return {0: float(w0), 1: float(w1)}

def focal_bce(gamma=2.0, alpha=0.25):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-6, 1-1e-6)
        bce = -(y_true * tf.math.log(y_pred) + (1-y_true) * tf.math.log(1-y_pred))
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1-y_pred)
        w = alpha * tf.pow(1-pt, gamma)
        return tf.reduce_mean(w * bce)
    return loss

# ---------- Models ----------

def build_backbone(arch, units=128, layers=2, dropout=0.3, kernel_size=3):
    def fn(x):
        if arch == "gru":
            for i in range(layers-1):
                x = L.GRU(units, return_sequences=True)(x)
                x = L.Dropout(dropout)(x)
            x = L.GRU(units, return_sequences=False)(x)
        elif arch == "lstm":
            for i in range(layers-1):
                x = L.LSTM(units, return_sequences=True)(x)
                x = L.Dropout(dropout)(x)
            x = L.LSTM(units, return_sequences=False)(x)
        elif arch == "bigru":
            for i in range(layers-1):
                x = L.Bidirectional(L.GRU(units, return_sequences=True))(x)
                x = L.Dropout(dropout)(x)
            x = L.Bidirectional(L.GRU(units, return_sequences=False))(x)
        elif arch == "bilstm":
            for i in range(layers-1):
                x = L.Bidirectional(L.LSTM(units, return_sequences=True))(x)
                x = L.Dropout(dropout)(x)
            x = L.Bidirectional(L.LSTM(units, return_sequences=False))(x)
        elif arch == "cnn1d":
            for i in range(layers):
                x = L.Conv1D(units, kernel_size=kernel_size, padding="same", activation="relu")(x)
                x = L.Dropout(dropout)(x)
            x = L.GlobalAveragePooling1D()(x)
        elif arch == "tcn":
            # TCN simplificada: blocos dilatados + residual
            for i in range(layers):
                dil = 2**i
                y = L.Conv1D(units, kernel_size=kernel_size, padding="same", dilation_rate=dil, activation="relu")(x)
                y = L.Dropout(dropout)(y)
                y = L.Conv1D(units, kernel_size=1, padding="same")(y)
                x = L.Add()([x, y])
                x = L.Activation("relu")(x)
            x = L.GlobalAveragePooling1D()(x)
        else:
            raise ValueError(f"Arquitetura desconhecida: {arch}")
        return x
    return fn

def build_masked_model(input_shape, arch="bilstm", units=128, layers_n=2, dropout=0.3, kernel_size=3, lr=1e-3):
    T, F = input_shape
    inp_seq = L.Input(shape=(T, F), name="seq")
    inp_mask = L.Input(shape=(T,), name="mask")
    m = L.Lambda(lambda z: tf.expand_dims(z, -1))(inp_mask)
    x = L.Multiply()([inp_seq, m])  # aplica máscara (0 nas posições de padding)

    x = build_backbone(arch, units=units, layers=layers_n, dropout=dropout, kernel_size=kernel_size)(x)
    out = L.Dense(1, activation="sigmoid")(x)
    model = Model([inp_seq, inp_mask], out, name=arch)
    model.compile(optimizer=opt.Adam(lr), loss="binary_crossentropy")
    return model

# ---------- Train/Eval ----------

def eval_metrics(y_true, y_prob, thresh=0.5):
    y_true = (y_true > 0.5).astype(int)
    ap = float(average_precision_score(y_true, y_prob))
    y_pred = (y_prob >= thresh).astype(int)
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    f2 = float(fbeta_score(y_true, y_pred, beta=2, zero_division=0))
    return ap, f2, rec, prec

def run_one(arch, Xtr, ytr, Mtr, Xva, yva, Mva, balance, epochs, batch, lr, units, layers_n, dropout, kernel):
    # balanceamento
    class_weight = None
    if balance == "ros":
        Xtr, ytr, Mtr = oversample_ros(Xtr, ytr, Mtr)
    elif balance == "class_weight":
        class_weight = compute_class_weight(ytr)

    model = build_masked_model((Xtr.shape[1], Xtr.shape[2]), arch=arch, units=units, layers_n=layers_n, dropout=dropout, kernel_size=kernel, lr=lr)
    if balance == "focal":
        model.compile(optimizer=opt.Adam(lr), loss=focal_bce(gamma=2.0, alpha=0.25))

    es = cb.EarlyStopping(monitor="val_loss", patience=max(2, epochs//3), restore_best_weights=True, verbose=0)
    hist = model.fit([Xtr, Mtr], ytr, validation_data=([Xva, Mva], yva),
                     epochs=epochs, batch_size=batch, callbacks=[es],
                     class_weight=class_weight, verbose=0)
    # probs (val/test serão avaliados fora)
    yv_prob = model.predict([Xva, Mva], verbose=0).reshape(-1)
    return model, yv_prob

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_npz", required=True)
    ap.add_argument("--val_npz", required=True)
    ap.add_argument("--test_npz", required=True)
    ap.add_argument("--archs", nargs="+", default=["gru","bigru","bilstm","cnn1d","tcn"])
    ap.add_argument("--balance", choices=["none","ros","class_weight","focal"], default="class_weight")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--units", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--kernel_size", type=int, default=3)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--models_outdir", default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)

    Xtr, ytr, Mtr, _ = load_npz(args.train_npz)
    Xva, yva, Mva, _ = load_npz(args.val_npz)
    Xte, yte, Mte, _ = load_npz(args.test_npz)

    results = []
    for arch in args.archs:
        print(f"[RUN] {arch} | balance={args.balance}")
        model, yv_prob = run_one(arch, Xtr, ytr, Mtr, Xva, yva, Mva,
                                 balance=args.balance, epochs=args.epochs, batch=args.batch_size,
                                 lr=args.lr, units=args.units, layers_n=args.layers, dropout=args.dropout, kernel=args.kernel_size)
        # val metrics
        ap_val, f2_val, rec_val, prec_val = eval_metrics(yva, yv_prob, 0.5)
        results.append({"Arquitetura": arch, "Partição": "val", "PR-AUC": ap_val, "F2": f2_val, "Recall": rec_val, "Precisão": prec_val, "balance": args.balance})
        # test
        yte_prob = model.predict([Xte, Mte], verbose=0).reshape(-1)
        ap_te, f2_te, rec_te, prec_te = eval_metrics(yte, yte_prob, 0.5)
        results.append({"Arquitetura": arch, "Partição": "test", "PR-AUC": ap_te, "F2": f2_te, "Recall": rec_te, "Precisão": prec_te, "balance": args.balance})

        # save model (optional)
        if args.models_outdir:
            os.makedirs(args.models_outdir, exist_ok=True)
            out_path = os.path.join(args.models_outdir, f"{arch}_{args.balance}.keras")
            model.save(out_path)

    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print("\n=== Resultados ===")
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()
