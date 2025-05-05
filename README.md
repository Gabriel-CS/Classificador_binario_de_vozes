# Classificador binario de vozes

Este repositório contém o código e a documentação de um projeto acadêmico de classificação binária de vozes, desenvolvido como trabalho avaliativo na universidade.

📋 Sumário
• Visão Geral

• Tecnologias e Dependências

• Estrutura do Projeto

• Pré-processamento dos Dados

• Métodos e Algoritmos

• Como Executar

• Resultados

• Boas Práticas e Dicas de Melhoria de Código

• Contribuições
• Licença

Visão Geral

O objetivo deste projeto é construir um classificador binário que distingu entre dois tipos de vozes (por exemplo, fala masculina vs. feminina). O pipeline inclui:

1. Extração e pré-processamento de características de áudio.
2. Extimativa e Redução de dimensionalidade com PCA.
3. Avaliação de agrupamentos usando K-means e método de silhueta.
4. Construção de vizinhança baseada em Triangulação de Delaunay.
5. Treinamento e comparação de classificadores: Random Forest, KNN e SVM.

Tecnologias e Dependências

Python 3.9+

bibliotecas principais:

• numpy, pandas

• scikit-learn

• scipy

• matplotlib / seaborn

• librosa (para manipulação de áudio)
