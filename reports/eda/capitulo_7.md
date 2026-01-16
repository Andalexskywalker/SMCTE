# 7. Limpeza, Pré-Processamento e Análise Exploratória (EDA)

Este capítulo descreve o tratamento dos dados, os mecanismos de prevenção de fuga de informação, as decisões de balanceamento e a análise exploratória (EDA) que suportam as escolhas metodológicas adotadas nos modelos. O objetivo é assegurar um *pipeline* reprodutível, sem *leakage*, que preserve a estrutura temporal por evento e forneça entradas estáveis aos algoritmos.

## 7.1. Caracterização do *dataset* e alvo

O *dataset* contém séries temporais de encontros orbitais identificados por `event_id`, com observações ordenadas por `time_to_tca`. A variável contínua `risk` origina o alvo binário de “alto risco” com limiar **-6.0**:
\[
y_i=\mathbb{1}{\{\text{risk}_i \ge -6.0\}}
\]
Para análise, considerámos rotulagem **linha-a-linha** e **ao nível do evento** (*any_high_risk*).

**Tabelas/Figuras**  
• Tabela 7.1 — Tipos de dados: `reports/eda/dtypes_count.csv`  
• Tabela 7.2 — Top-30 colunas com valores em falta: `reports/eda/missing_top30.csv`  
• Figura 7.1 — Distribuição de `risk` + *scatter* vs `time_to_tca`: `reports/eda/risk_overview.png`  
• Figura 7.2 — Proporção de classes (row-level): `reports/eda/balance_row.png`  
• Figura 7.3 — Proporção de classes (event-level): `reports/eda/balance_event.png`

**Resumo**  
• Linhas: **162634** | Colunas: **101** (numéricas: **100**, categóricas: **1**).  
• Classe positiva (linha): **6.3%** | (evento): **15.0%**.  
• Outliers presentes → justifica *winsorization* (ver §7.4).

## 7.2. Prevenção de *leakage* e divisão por evento

Todas as imputações, escalas e codificações são ajustadas **apenas no treino** e aplicadas a validação/teste. A divisão é **event-wise** por `event_id` (opcionalmente estratificada no rótulo ao nível do evento). Removemos variáveis com risco de fuga (e.g., `max_risk`, `_at_tca`, `peak`, `post_`, `c_rcs_estimate`).

## 7.3. *Missing values* e sanidade física

1) **Sanidade física**: `miss_distance<0→0`; `relative_speed≤0→NaN`.  
2) **Imputação por papel** (fit no treino; apply em val/test): medianas para `t_`/`c_` e restantes; categóricas por *most frequent/UNKNOWN*.  
3) **Forward-fill por evento**: ordenado por `time_to_tca` (desc), preservando causalidade.

## 7.4. *Clipping* (winsorization) e robustez

Aplicámos *clipping* por percentis **[0.5; 99.5]** do treino; os limites estão em **`reports/eda/clip_limits.json`**, usados no *pipeline* antes da normalização para reduzir a influência de *outliers*.

## 7.5. Normalização e codificação (*ColumnTransformer*)

O *transformer* (ajustado no treino e serializado em `models/pre.joblib`) aplica:
- Numéricas: `SimpleImputer(median)` → `StandardScaler()`.  
- Categóricas: `SimpleImputer(most_frequent)` → `OneHotEncoder(ignore)`.

**Tabela 7.3 — Resumo do *pipeline* de pré-processamento**  
| Etapa | Descrição | Ajuste | Aplicação |
|---|---|---|---|
| Sanidade física | Correções físicas simples (e.g., miss_distance≥0; relative_speed>0→NaN) | — | Train/Val/Test |
| Split por evento | Separação por `event_id`, estratificável por rótulo de evento | — | — |
| Imputação por papel | Medianas do treino (`t_`, `c_`, resto); categóricas `most_frequent` | **Treino** | Val/Test |
| Forward-fill | Por `event_id` em ordem a `time_to_tca` (desc) | — | Train/Val/Test |
| Clipping | Winsorização por percentis (0.5–99.5) do treino | — | Train/Val/Test |
| ColumnTransformer | Num: Imputer→Scaler; Cat: Imputer→OHE | **Treino** | Val/Test |


## 7.6. Desbalanceamento e *oversampling*

Avaliámos modelos **com e sem** *RandomOverSampler (ROS)* **apenas no treino**; validação/teste mantêm a distribuição real. Esta decisão foi validada nas experiências (Cap. 11) com ganhos em **PR-AUC** e **recall** na configuração vencedora.

## 7.7. EDA dirigida por predição: ranking e distribuição por classe

• **Mutual Information (MI)** — Top-10: azimuth, c_j2k_inc, relative_velocity_n, relative_velocity_t, c_j2k_sma, event_id, t_j2k_inc, c_h_per, c_h_apo, t_j2k_sma  
• **AUC/PR-AUC univariada** — Top-10 PR-AUC: time_to_tca, t_rcs_estimate, t_j2k_sma, t_j2k_inc, t_h_apo, t_h_per, c_h_per, t_span, c_j2k_sma, c_h_apo  
• **Violinos (Top-K por MI)** — selecione 6–8 para o relatório:
  - reports/eda/violin/violin_azimuth.png\n  - reports/eda/violin/violin_c_j2k_inc.png\n  - reports/eda/violin/violin_relative_velocity_n.png\n  - reports/eda/violin/violin_relative_velocity_t.png\n  - reports/eda/violin/violin_c_j2k_sma.png\n  - reports/eda/violin/violin_event_id.png\n  - reports/eda/violin/violin_t_j2k_inc.png\n  - reports/eda/violin/violin_c_h_per.png\n
• **Redundância** — Heatmap (Top-K): `reports/eda/corr/heatmap_top.png`  
• **Séries temporais (exemplos)** — `reports/eda/risk_trajectories_examples.png`

**Conclusões do EDA para o pipeline.**  
(i) As Top-K mostram separação por classe, mas **FS rígida** (e.g., K=30) pode degradar modelos tabulares potentes → *baseline* com **todas** as *features* e comparação com `K=30`.  
(ii) *Outliers* e escalas heterogéneas justificam *clipping* + normalização.  
(iii) O desbalanceamento sustenta o uso de **ROS** no treino.

## 7.8. Configuração final (treino e app)

- Limiar do alvo: **-6.0** (alto risco).  
- Split: **event-wise** (estratificável por rótulo de evento).  
- Sanidade física + imputação por papel + forward-fill por evento.  
- *Clipping* por percentis → **`reports/eda/clip_limits.json`**.  
- `ColumnTransformer` (imputer→scaler / imputer→OHE) → **`models/pre.joblib`**.  
- Balanceamento (classificação): **ROS** apenas no treino.

