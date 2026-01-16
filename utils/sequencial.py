import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler

# Parâmetros de I/O
CLEAN_CSV    = "csv/train_data_clean.csv"
FEATURES_CSV = "csv/train_data_features.csv"
OUTPUT_CSV   = "csv/train_seq_data.csv"

# Colunas-chave
EVENT_COL    = "event_id"
TIME_COL     = "time_to_tca"
TARGET_COL   = "risk"

# Janela temporal
TIMESTEPS    = 5
FILL_VALUE   = 0.0

# 1) Carrega CDMs limpos e estatísticas estáticas
df_cdm  = pd.read_csv(CLEAN_CSV)
df_feat = pd.read_csv(FEATURES_CSV)

# 2) Faz merge, mantendo 'risk' do CDM e colocando o static em 'risk_static'
df = df_cdm.merge(
    df_feat,
    on=EVENT_COL,
    how="left",
    suffixes=("", "_static")
)

# 3) Tratamento de strings residuais
df = df.replace({
    "False": 0,
    "True":  1,
    "UNKNOWN": 0
})

# 4) One‐hot encode de qualquer coluna object (e.g. c_object_type)
cat_cols = df.select_dtypes(include="object").columns.tolist()
df = pd.get_dummies(df, columns=cat_cols, dummy_na=False)

# 5) Lista de todas as features (exclui index, tempo e target)
all_feats = [
    c for c in df.columns
    if c not in (EVENT_COL, TIME_COL, TARGET_COL)
]

# 6) Normaliza apenas as features numéricas
scaler = StandardScaler()
df[all_feats] = scaler.fit_transform(df[all_feats].to_numpy())

# 7) Função para construir as janelas temporais por evento
def build_seq_windows(df, event_col, feat_cols, time_col, timesteps, fill_value):
    Xw, yw, eids = [], [], []
    # para cada evento, ordena por tempo e extrai a matriz
    for eid, grp in df.sort_values([event_col, time_col]).groupby(event_col):
        arr = grp[feat_cols].to_numpy()
        L   = arr.shape[0]
        if L >= timesteps:
            win = arr[-timesteps:]
        else:
            pad = np.full((timesteps - L, arr.shape[1]), fill_value, dtype=float)
            win = np.vstack([pad, arr])
        Xw.append(win)
        # target = último risk do CDM
        yw.append(grp[TARGET_COL].iloc[-1])
        eids.append(eid)
    return np.stack(Xw, axis=0), np.array(yw), eids

# 8) Cria X_seq, y_seq e lista de event_ids
X_seq, y_seq, eids = build_seq_windows(
    df, EVENT_COL, all_feats, TIME_COL, TIMESTEPS, FILL_VALUE
)

# 9) Achata para (n_events, TIMESTEPS * n_feats) e cria colunas “feat_t-4 … feat_t-0”
flat = X_seq.reshape(X_seq.shape[0], -1)
cols = [f"{feat}_t{-i-1}" for i in range(TIMESTEPS) for feat in all_feats]

df_seq = pd.DataFrame(flat, columns=cols)
df_seq[EVENT_COL]  = eids
df_seq[TARGET_COL] = y_seq

# 10) Salva
df_seq.to_csv(OUTPUT_CSV, index=False)
print(f"-> {OUTPUT_CSV} criado com shape {df_seq.shape}")
