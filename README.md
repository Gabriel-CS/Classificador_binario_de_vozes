# 🎙️ Classificador Binário de Vozes (Male vs. Female)

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-orange)

## 📖 Sobre o Projeto

Este projeto foi desenvolvido como trabalho avaliativo da disciplina de **Tópicos de Matemática Aplicada**. O objetivo principal foi colocar em prática os métodos matemáticos e estatísticos apresentados em aula para resolver um problema real de classificação.

O sistema consiste em um **Classificador Binário de Vozes**, capaz de distinguir entre fala masculina e feminina. Para isso, utilizamos a linguagem **Python** para implementar um pipeline completo de processamento de dados, desde a extração de features de áudio até a comparação de diferentes algoritmos de aprendizado de máquina.

## 🚀 Pipeline e Metodologia

O desenvolvimento do projeto seguiu um fluxo estruturado de processamento de dados e modelagem:

1.  **Pré-processamento de Áudio:**
    *   Coleta de dados brutos de áudio (homens e mulheres).
    *   Extração de escalas de áudio e conversão para dados estruturados em **CSV**.
    *   Extração e limpeza de características (features).

2.  **Matemática e Estatística Aplicada:**
    *   **PCA (Principal Component Analysis):** Estimativa e redução de dimensionalidade para otimizar os dados.
    *   **K-means & Método da Silhueta:** Avaliação de agrupamentos não supervisionados para entender a separabilidade das classes.
    *   **Triangulação de Delaunay:** Construção de vizinhanças geométricas para análise estrutural dos dados.

3.  **Classificação Supervisionada:**
    *   Treinamento e comparação de desempenho entre três algoritmos:
        *   🌲 **Random Forest**
        *   🔍 **KNN (K-Nearest Neighbors)**
        *   📐 **SVM (Support Vector Machine)**

## 🛠️ Tecnologias Utilizadas

*   **Linguagem:** Python
*   **Manipulação de Dados:** Pandas, NumPy
*   **Processamento de Áudio:** [Librosa]
*   **Machine Learning:** Scikit-Learn
*   **Visualização:** Matplotlib, Seaborn
*   **Ambiente:** Jupyter Notebook / VS Code

## 📊 Conjunto de Dados

Os dados utilizados neste projeto são provenientes de um banco de dados público disponível no **Kaggle**, contendo amostras de áudio de vozes masculinas e femininas.

*   **Fonte:** [Inserir Link do Dataset no Kaggle se houver]
*   **Formato:** Arquivos de áudio (.wav/.mp3) convertidos para CSV.

## 📦 Como Rodar o Projeto

Siga os passos abaixo para executar o código em sua máquina local:

1.  **Clone o repositório:**
    ```bash
    git clone https://github.com/SEU_USUARIO/NOME_DO_REPOSITORIO.git
    ```

2.  **Navegue até a pasta do projeto:**
    ```bash
    cd NOME_DO_REPOSITORIO
    ```

3.  **Instale as dependências:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Certifique-se de criar um arquivo requirements.txt com as bibliotecas usadas)*

4.  **Execute o notebook ou script principal:**
    ```bash
    # Exemplo para Jupyter
    jupyter notebook analisis_vozes.ipynb

    # Ou script Python
    python main.py
    ```

## 📈 Resultados

*(Opcional: Adicione aqui um breve resumo ou gráfico de qual classificador teve melhor acurácia)*

| Classificador | Acurácia |
| :--- | :--- |
| **Random Forest** | [Inserir %] |
| **SVM** | [Inserir %] |
| **KNN** | [Inserir %] |

## 👥 Autores

*   **[Seu Nome]** - [Seu GitHub](https://github.com/Gabriel-CS)
*   **[Nome do Colega]** - 


---
*Projeto acadêmico desenvolvido para a disciplina de Tópicos de Matemática Aplicada.*
