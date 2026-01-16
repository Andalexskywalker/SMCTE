import os
import sys
import json
import numpy as np
import pandas as pd
import streamlit as st

# ---------- Bootstrap p/ o unpickle do preprocessor encontrar 'safe_clip' ----------
APP_DIR = os.path.dirname(__file__)
CANDIDATES = [APP_DIR, os.path.join(APP_DIR, "utils")]
for p in CANDIDATES:
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)
try:
    import safe_clip  # noqa: F401
except ModuleNotFoundError:
    try:
        import utils.safe_clip as _sc
        sys.modules["safe_clip"] = _sc
    except Exception:
        pass
# -----------------------------------------------------------------------------------

# Matplotlib sem GUI
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# TF/Keras
from tensorflow import keras

# ---------------------- Config ----------------------
st.set_page_config(page_title="SMCTE — Seq Model Selector", page_icon="🛰️", layout="wide")

TOP3_INDEX = "models_seq/top3_models.json"
PREPROCESSOR_PATH = "models/pre.joblib"
DEFAULT_RAW_CSV = "csv/test_data.csv"

RISK_THRESHOLD_CONT = -6.0
DEFAULT_THRESHOLD = 0.30

# ---------------------- Helpers ----------------------
@st.cache_resource(show_spinner=False)
def load_preprocessor():
    import joblib
    if not os.path.exists(PREPROCESSOR_PATH):
        raise FileNotFoundError(f"Preprocessor não encontrado: {PREPROCESSOR_PATH}")
    pre = joblib.load(PREPROCESSOR_PATH)
    try:
        raw_in = list(pre.feature_names_in_)
    except Exception:
        raw_in = None
    try:
        out_names = list(pre.get_feature_names_out())
    except Exception:
        out_names = None
    return pre, raw_in, out_names

@st.cache_resource(show_spinner=False)
def load_top3_models():
    if not os.path.exists(TOP3_INDEX):
        raise FileNotFoundError(f"Índice dos top-3 não encontrado: {TOP3_INDEX}")
    meta = json.load(open(TOP3_INDEX, "r", encoding="utf-8"))
    models_meta = meta.get("models", [])
    loaded = {}
    for item in models_meta:
        path = item["path"]; name = item["model"]
        if not os.path.exists(path):
            st.warning(f"Modelo '{name}' não encontrado em {path}.")
            continue
        m = keras.models.load_model(path)
        shp = m.inputs[0].shape  # (None, T, F)
        T = int(shp[1]); F = int(shp[2])
        loaded[name] = (m, (T, F))
    if not loaded:
        raise RuntimeError("Nenhum dos top-3 modelos foi carregado.")
    return loaded, meta

@st.cache_resource(show_spinner=False)
def load_seq_feature_names():
    """
    Lê os nomes das features (ordem correta) usadas no treino sequencial,
    a partir de um dos .npz criados pelo preprocessing v4 (contém 'feature_names').
    """
    candidates = ["data/val_seq.npz", "data/train_seq.npz", "data/test_seq.npz"]
    for p in candidates:
        if os.path.exists(p):
            z = np.load(p, allow_pickle=True)
            feats = z["feature_names"].tolist()
            seq_len = int(z["seq_len"])
            return feats, seq_len, p
    raise FileNotFoundError(
        "Não encontrei nenhum dos .npz com meta de treino (data/val_seq.npz, data/train_seq.npz, data/test_seq.npz). "
        "Gera-os com o preprocessing sequencial v4."
    )

def binarize_risk(series, thr=RISK_THRESHOLD_CONT):
    s = pd.Series(series)
    vals = set(pd.unique(s.dropna()))
    if vals.issubset({0, 1}):
        return s.astype(int)
    return (s >= thr).astype(int)

def plot_pr_curve(y_true, y_proba, title="Precision-Recall"):
    from sklearn.metrics import precision_recall_curve, average_precision_score
    p, r, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(r, p); ax.grid(True, ls=":")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title(f"{title} (AP={ap:.3f})")
    st.pyplot(fig, use_container_width=False)

def plot_confusion(y_true, y_pred, title="Confusion Matrix"):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4.8, 3.8))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xlabel("Pred"); ax.set_ylabel("True"); ax.set_title(title)
    st.pyplot(fig, use_container_width=False)

def compute_metrics(y_true, y_proba, y_pred):
    from sklearn.metrics import (
        average_precision_score, roc_auc_score, fbeta_score,
        precision_score, recall_score, balanced_accuracy_score
    )
    has_both = len(np.unique(y_true)) > 1
    pr_auc = average_precision_score(y_true, y_proba) if has_both else np.nan
    roc = roc_auc_score(y_true, y_proba) if has_both else np.nan
    f2 = fbeta_score(y_true, y_pred, beta=2, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    bacc = balanced_accuracy_score(y_true, y_pred)
    return {"PR-AUC": pr_auc, "ROC-AUC": roc, "F2": f2, "Recall": rec, "Precision": prec, "Balanced Acc": bacc}

def build_seq_windows(df_proc: pd.DataFrame, event_col: str, time_col: str,
                      feat_cols: list[str], T: int):
    X_list, M_list, ev_ids = [], [], []
    for ev_id, g in df_proc.sort_values([event_col, time_col], ascending=[True, True]).groupby(event_col, sort=True):
        arr = g[feat_cols].to_numpy(dtype=float)
        L = arr.shape[0]
        if L >= T:
            sel = arr[-T:]; mask = np.ones(T, dtype=np.float32)
        else:
            pad = np.zeros((T - L, len(feat_cols)), dtype=float)
            sel = np.vstack([pad, arr])
            mask = np.concatenate([np.zeros(T - L, dtype=np.float32), np.ones(L, dtype=np.float32)])
        X_list.append(sel); M_list.append(mask); ev_ids.append(ev_id)
    X = np.stack(X_list, axis=0).astype(np.float32) if X_list else np.zeros((0, T, len(feat_cols)), dtype=np.float32)
    M = np.stack(M_list, axis=0).astype(np.float32) if M_list else np.zeros((0, T), dtype=np.float32)
    return X, M, ev_ids

# ---------------------- UI ----------------------
st.title("SMCTE — Previsão Sequencial (Top-3)")

# Artefactos
try:
    pre, RAW_FEATURES, PROC_FEATURES = load_preprocessor()
except Exception as e:
    st.error(f"Falha a carregar preprocessor: {e}")
    st.stop()

try:
    TOP_MODELS, TOP_META = load_top3_models()
except Exception as e:
    st.error(f"Falha a carregar modelos sequenciais: {e}")
    st.stop()

try:
    SEQ_FEATURES, SEQ_T_HINT, SEQ_SRC = load_seq_feature_names()
except Exception as e:
    st.error(str(e))
    st.stop()

with st.sidebar:
    st.info("ℹ️ **Sobre o Projeto**\n\n**Problema:** O lixo espacial ameaça satélites. Métodos tradicionais geram muitos falsos alarmes.\n\n**Solução:** Este sistema usa Deep Learning (RNNs) para prever colisões com maior precisão.")
    
    st.header("Configuração")
    model_names = sorted(TOP_MODELS.keys())
    model_names_disp = model_names + ["ensemble_top3"]
    choice = st.selectbox("Modelo sequencial", model_names_disp, index=0)
    threshold = st.slider("Decision threshold", 0.0, 1.0, DEFAULT_THRESHOLD, 0.01)
    st.caption("Métricas reportadas **ao nível do evento** (any_high_risk).")

# ---------- Pré-carregamento do CSV raw ----------
df = None
preloaded = False
if os.path.exists(DEFAULT_RAW_CSV):
    try:
        df = pd.read_csv(DEFAULT_RAW_CSV, low_memory=False)
        preloaded = True
        st.success(f"CSV pré-carregado: **{DEFAULT_RAW_CSV}**  —  {df.shape[0]} linhas × {df.shape[1]} colunas")
    except Exception as e:
        st.warning(f"Falha ao pré-carregar {DEFAULT_RAW_CSV}: {e}")

# Uploader opcional
uploaded = st.file_uploader(
    "Carrega outro CSV **raw** (opcional) com colunas: `event_id`, `time_to_tca`, features… (e opcional `risk`)",
    type=["csv"], disabled=False
)
if uploaded is not None:
    try:
        uploaded.seek(0)
        df_up = pd.read_csv(uploaded, sep=None, engine="python", low_memory=False)
    except Exception:
        uploaded.seek(0)
        df_up = pd.read_csv(uploaded, encoding="utf-8", sep=",", low_memory=False)
    df = pd.DataFrame(df_up)
    preloaded = False
    st.info(f"CSV carregado: **{getattr(uploaded,'name','uploaded.csv')}** — {df.shape[0]} × {df.shape[1]}")

if df is None:
    st.error("Não foi possível obter um CSV. Garante `csv/test_data.csv` ou usa o uploader.")
    st.stop()

# Checks mínimos
needed = {"event_id", "time_to_tca"}
missing = sorted(list(needed - set(df.columns)))
if missing:
    st.error("Faltam colunas obrigatórias:\n\n- " + "\n- ".join(missing))
    st.stop()

# Ground-truth (map por event_id)
y_true_map = None
if "risk" in df.columns:
    try:
        tmp = df[["event_id", "risk"]].copy()
        tmp["y_bin"] = binarize_risk(tmp["risk"], thr=RISK_THRESHOLD_CONT)
        y_true_map = tmp.groupby("event_id", sort=True)["y_bin"].max().to_dict()
    except Exception:
        y_true_map = None

# Preprocess (raw → proc)
if RAW_FEATURES is None:
    st.error("O `pre.joblib` não expõe `feature_names_in_`. Regera o preprocessor (sklearn ≥ 1.1).")
    st.stop()

miss_raw = sorted(list(set(RAW_FEATURES) - set(df.columns)))
if miss_raw:
    st.error("Faltam colunas RAW para o preprocessor:\n\n- " + "\n- ".join(miss_raw))
    st.stop()

X_raw = df[RAW_FEATURES].copy()
try:
    X_proc = pre.transform(X_raw)
except Exception as e:
    st.error(f"Erro ao aplicar preprocessor: {e}")
    st.stop()

# Nomes de saída do preprocessor
if PROC_FEATURES is None:
    # caso extremo: sem nomes, criamos genéricos
    PROC_FEATURES = [f"f{i}" for i in range(X_proc.shape[1])]

# DataFrame com output do preprocessor
proc_df = pd.DataFrame(X_proc, columns=PROC_FEATURES)
proc_df.insert(0, "time_to_tca", pd.to_numeric(df["time_to_tca"], errors="coerce").values)
proc_df.insert(0, "event_id", df["event_id"].values)

# === Alinhar nomes: mapear "num__feature" -> "feature" ===
def base_name(col: str) -> str:
    # remove prefixos tipo 'num__' ou 'cat__'
    if "__" in col:
        col = col.split("__", 1)[1]
    return col

# constrói mapa base->nome_real (primeira ocorrência; suficiente p/ numéricas)
base_to_real = {}
for c in proc_df.columns:
    if c in ("event_id", "time_to_tca"):
        continue
    b = base_name(c)
    base_to_real.setdefault(b, c)

# construir a lista de colunas reais na MESMA ORDEM do treino sequencial
missing_seq_feats = [f for f in SEQ_FEATURES if f not in base_to_real]
if missing_seq_feats:
    st.error(
        "As seguintes features de treino sequencial não existem após o preprocessor atual:\n\n- "
        + "\n- ".join(missing_seq_feats)
        + "\n\nGarante que estás a usar o MESMO `pre.joblib` do treino sequencial."
    )
    st.stop()

FEAT_FOR_MODEL_REAL = [base_to_real[f] for f in SEQ_FEATURES]
proc_df_model = proc_df[["event_id", "time_to_tca"] + FEAT_FOR_MODEL_REAL]

# Inferência — funções
def infer_with_model(model_obj, shape_TF, df_proc, feat_cols):
    T_expected, F_expected = shape_TF
    if len(feat_cols) != F_expected:
        raise ValueError(f"Features ({len(feat_cols)}) != F esperado ({F_expected}). "
                         "Garante que o preprocessor é o mesmo do treino.")
    Xseq, Mseq, ev_ids = build_seq_windows(df_proc, "event_id", "time_to_tca", feat_cols, T_expected)
    if Xseq.shape[0] == 0:
        return np.array([]), np.array([]), []
    probs = model_obj.predict([Xseq, Mseq], verbose=0).ravel().astype(float)
    return probs, Mseq, ev_ids

def infer(choice_name: str, df_proc, feat_cols):
    if choice_name == "ensemble_top3":
        probs_sum = None; any_ev_ids = None; any_shape = None
        for nm, (mdl, shapeTF) in TOP_MODELS.items():
            p, _, ev_ids = infer_with_model(mdl, shapeTF, df_proc, feat_cols)
            if probs_sum is None:
                probs_sum = p; any_ev_ids = ev_ids; any_shape = shapeTF
            else:
                probs_sum = probs_sum + p
        probs = probs_sum / max(1, len(TOP_MODELS))
        return "Ensemble (Top-3)", probs, any_ev_ids, any_shape
    else:
        mdl, shapeTF = TOP_MODELS[choice_name]
        probs, _, ev_ids = infer_with_model(mdl, shapeTF, df_proc, feat_cols)
        return choice_name, probs, ev_ids, shapeTF

# ---------- Chamada da inferência ----------
model_label = ""
y_proba_ev = np.array([])
ev_ids = []
shapeTF = (0, 0)

try:
    model_label, y_proba_ev, ev_ids, shapeTF = infer(choice, proc_df_model, FEAT_FOR_MODEL_REAL)
except Exception as e:
    st.error(f"Falha na inferência sequencial: {e}")
    st.stop()

if y_proba_ev.size == 0 or len(ev_ids) == 0:
    st.warning("Sem eventos após o pré-processamento (Xseq vazio). Verifica se `event_id`/`time_to_tca` estão corretos.")
    st.stop()

# Resultados por evento
y_pred_ev = (y_proba_ev >= threshold).astype(int)
res = pd.DataFrame({"event_id": ev_ids, "proba": y_proba_ev, "pred": y_pred_ev})

st.subheader("Previsão por evento")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Eventos", len(res))
c2.metric(f"Positivos (thr={threshold:.2f})", int(res["pred"].sum()))
if len(res):
    c3.metric("Média de prob.", f"{res['proba'].mean():.3f}")
    T_expected, F_expected = shapeTF
    c4.metric("Janela (T×F)", f"{T_expected} × {F_expected}")

st.dataframe(res.head(200), use_container_width=True)
st.download_button("Descarregar previsões por evento (CSV)",
                   res.to_csv(index=False).encode("utf-8"),
                   file_name=f"pred_seq_{model_label.replace(' ','_').lower()}.csv",
                   mime="text/csv")

st.divider()
st.subheader("Avaliação (se o CSV tiver `risk`) — nível de evento")

# Ground-truth alinhado (se tiver)
if y_true_map is None:
    st.info("Sem ground-truth de evento — se incluíres `risk`, avaliamos por any_high_risk por evento.")
else:
    try:
        y_true_event = np.array([y_true_map.get(eid, np.nan) for eid in ev_ids], dtype=float)
        ok = ~np.isnan(y_true_event)
        if ok.sum() == 0:
            st.info("Não foi possível alinhar ground-truth com os event_id previstos.")
        else:
            y_true_event = y_true_event[ok].astype(int)
            y_proba_eval = y_proba_ev[ok]
            thr = st.slider("Threshold para métricas", 0.0, 1.0, threshold, 0.01, key="thr_metrics")
            y_pred_eval = (y_proba_eval >= thr).astype(int)
            mets = compute_metrics(y_true_event, y_proba_eval, y_pred_eval)
            st.table(pd.DataFrame([mets]).T.rename(columns={0: "value"}))

            c1, c2 = st.columns(2)
            with c1: plot_pr_curve(y_true_event, y_proba_eval, title=f"PR — {model_label} (evento)")
            with c2: plot_confusion(y_true_event, y_pred_eval, title=f"CM — {model_label} (evento)")
    except Exception as e:
        st.info(f"Ground-truth presente mas não foi possível alinhar: {e}")

src = DEFAULT_RAW_CSV if preloaded else "(upload do utilizador)"
st.caption(
    f"Fonte de dados: **{src}** — features de treino lidas de **{SEQ_SRC}**. "
    "Os modelos são redes sequenciais (`.keras`) treinadas em janelas fixas (T×F) com máscara para *padding*. "
    "O preprocessor (`pre.joblib`) é aplicado por linha; depois usamos **exatamente as features do treino sequencial** "
    "na mesma ordem guardada no `.npz` (mapeando automaticamente nomes tipo `num__feature` → `feature`)."
)
