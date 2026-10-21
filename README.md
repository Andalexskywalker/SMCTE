# Sistema de Monitorização e Colisão entre Tráfego Espacial (SMCTE)

**Autores:** André Ramos & Vitor Esteves  
**Contexto:** Projeto Final de Curso (Licenciatura em Engenharia Informática, IPCB)

![Status do Projeto](https://img.shields.io/badge/Status-Concluído-success)
![Python](https://img.shields.io/badge/Python-3.9+-blue)
![Deep Learning](https://img.shields.io/badge/AI-Deep%20Learning-orange)

## 📡 Sobre o Projeto

O **SMCTE** é uma solução avançada de **Conjunction Assessment** (Avaliação de Conjunções) desenvolvida para prever e monitorizar o risco de colisão entre satélites e detritos espaciais. 

Numa era onde o "Space Debris" ameaça a sustentabilidade das órbitas terrestres, este projeto propõe uma abordagem baseada em **Deep Learning** (Redes Neuronais Recorrentes e Seq2Seq) para complementar ou substituir os modelos físicos tradicionais, que são computacionalmente pesados.

### 🎯 Objetivos Principais
1.  **Monitorização em Tempo Real:** Dashboard interativo para análise de eventos de conjunção.
2.  **Previsão de Risco:** Estimativa da probabilidade de colisão (Risk Prediction) usando modelos sequenciais.
3.  **Análise Temporal:** Estudo da evolução do risco até ao momento de maior aproximação (TCA - Time of Closest Approach).

## 🛠️ Stack Tecnológica

Este projeto combina Engenharia de Software com Data Science avançada:

*   **Frontend/UI:** [Streamlit](https://streamlit.io/) (Visualização de Dados Interativa)
*   **Deep Learning:** TensorFlow / Keras (Modelos LSTM, GRU, Seq2Seq)
*   **Processamento de Dados:** Pandas, NumPy, Scikit-Learn
*   **Visualização:** Matplotlib, Seaborn

## 🚀 Funcionalidades da Aplicação

A aplicação web (`app.py`) permite aos operadores:
- **Carregar Ficheiros de Conjunção:** Suporte para ficheiros CSV com dados de telemétria orbital (CDM - Conjunction Data Messages).
- **Infeiriência AI:** Executar os modelos de Deep Learning pré-treinados para prever o risco futuro.
- **Visualização Gráfica:**
    - Matrizes de Confusão e Curvas de Precisão-Recall.
    - Evolução dos parâmetros orbitais (Miss Distance, Probabilidade de Colisão).
    - Alertas automáticos para eventos de "Alto Risco".

## 📦 Instalação e Execução

Para rodar este projeto localmente, siga os passos abaixo:

1.  **Clonar o repositório:**
    ```bash
    git clone https://github.com/seu-utilizador/smcte.git
    cd smcte
    ```

2.  **Criar um ambiente virtual (recomendado):**
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # Linux/Mac
    source venv/bin/activate
    ```

3.  **Instalar dependências:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Executar a aplicação:**
    ```bash
    streamlit run app.py
    ```

## 📊 Estrutura do Projeto

*   `app.py`: Ponto de entrada da aplicação Web.
*   `models/`: Contém os modelos `.keras` treinados e o pre-processador `pre.joblib`.
*   `SMCTE/`: Notebooks de Análise Exploratória (EDA) e Feature Engineering.
*   `csv/`: Dados de exemplo para teste.

---
*Este projeto foi desenvolvido no âmbito académico para demonstrar a aplicação de IA em problemas críticos de Engenharia Aeroespacial.*
