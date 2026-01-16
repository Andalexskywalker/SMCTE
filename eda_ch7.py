# fig_missing_top30.py
# Lê reports/eda/missing_top30.csv e gera uma figura "tipo-tabela"
# com barras horizontais (contagens) e rótulos com percentagens.
# Acrescenta um quadro lateral com a política de imputação.

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

IN  = Path("reports/eda/missing_top30.csv")   # gerado no EDA
OUT = Path("reports/eda/missing_top30_fig.png")

# --- carregar e garantir colunas necessárias ---
df = pd.read_csv(IN)
# compatibilidade com versões antigas (onde só havia n_missing)
if "feature" not in df.columns:
    df = df.rename(columns={df.columns[0]: "feature"})
if "n_missing" not in df.columns:
    raise SystemExit("missing_top30.csv precisa da coluna 'n_missing'.")
if "pct_missing" not in df.columns:
    # % de linhas em falta por coluna
    # tentativa: inferir N a partir do máximo possível (não vem no CSV)
    # se o EDA tiver também dtypes_count.csv não ajuda; então pede ao user
    # => fallback: normaliza pela maior contagem (não é ideal). Melhor: passa N via env var.
    print("[aviso] 'pct_missing' não existe; vou inferir com base no maior n_missing.")
    N = int(df["n_missing"].max())  # fallback conservador
    df["pct_missing"] = 100.0 * df["n_missing"] / max(N, 1)

# ordenar por n_missing desc e ficar só com 30
top = (df[df["n_missing"] > 0]
         .sort_values("n_missing", ascending=False)
         .head(30)
         .copy())

if top.empty:
    fig, ax = plt.subplots(figsize=(8,3))
    ax.axis("off")
    ax.text(0.5, 0.5, "Sem colunas com valores em falta no treino.", ha="center", va="center")
    fig.savefig(OUT, dpi=220, bbox_inches="tight")
    print("Figura criada:", OUT)
    raise SystemExit(0)

# --- figura: barras + quadro de imputação ---
mpl.rcParams["axes.unicode_minus"] = False
h = max(3.0, 0.35*len(top) + 1.4)
fig = plt.figure(figsize=(12, h))
gs = fig.add_gridspec(nrows=1, ncols=5, width_ratios=[4,0.2,1.2,0.1,0.1])

ax = fig.add_subplot(gs[0, :3])

y = list(range(len(top)))[::-1]
ax.barh(y, top["n_missing"][::-1].values, color="#4C78A8")
ax.set_yticks(y)
ax.set_yticklabels(top["feature"][::-1].values)
ax.set_xlabel("Contagem de ausências no treino")
ax.set_title("Top-30 colunas com valores em falta (treino)")

# rótulos com % por coluna (à direita da barra)
for yi, (cnt, pct) in enumerate(zip(top["n_missing"][::-1].values, top["pct_missing"][::-1].values)):
    ax.text(cnt, yi, f"  {pct:.1f}%", va="center", ha="left", fontsize=9)

ax.grid(axis="x", alpha=0.2, ls=":")

# pequeno eixo superior com escala normalizada (opcional)
ax_top = ax.twiny()
ax_top.set_xlim(ax.get_xlim())
ax_top.set_xlabel("escala comum")

# quadro lateral com a política de imputação
ax_pol = fig.add_subplot(gs[0, 3:])
ax_pol.axis("off")
policy = (
    "Política de imputação (treino → aplicada nas restantes partições):\n"
    "• Variáveis contínuas: mediana.\n"
    "• Indicadores binários: valor neutro (0/False).\n"
    "• Sequências estritamente temporais: forward-fill (quando aplicável).\n"
    "• Estatísticas ajustadas só no treino e serializadas no pipeline."
)
ax_pol.text(0.0, 0.98, policy, va="top", ha="left", fontsize=10, wrap=True)

fig.tight_layout()
fig.savefig(OUT, dpi=220, bbox_inches="tight")
print("Figura criada ", OUT)
