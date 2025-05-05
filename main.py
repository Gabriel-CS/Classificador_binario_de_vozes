import pandas as pd
import numpy as np
from scipy.spatial import Delaunay
from scipy.spatial.distance import mahalanobis
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import plotly.express as px

# models
from utils.SVM import main_svm
from utils.Knn import main_knn
from utils.Randon_forest import main_Randon_forest

def carregar_dados(caminho_csv):
    """
    Carrega os dados do arquivo CSV.
    """
    try:
        df = pd.read_csv(caminho_csv)
        print("✅ Dados carregados com sucesso!")
        return df
    except Exception as e:
        print(f"❌ Erro ao carregar o arquivo: {e}")
        return None

def construir_grafo(df, tipo_distancia='mahalanobis'):
    """
    Constrói o grafo de vizinhança utilizando a triangulação de Delaunay.
    Permite escolher entre distância Euclidiana e Mahalanobis.
    """
    pontos = df[['PC1', 'PC2']].values
    triang = Delaunay(pontos)
    
    if tipo_distancia == 'mahalanobis':
        cov = np.cov(pontos, rowvar=False)
        inv_covmat = np.linalg.inv(cov)
    
    arestas_set = set()
    for simplex in triang.simplices:
        for i in range(len(simplex)):
            for j in range(i+1, len(simplex)):
                aresta = tuple(sorted((simplex[i], simplex[j])))
                arestas_set.add(aresta)
    
    arestas = []
    for (i, j) in arestas_set:
        p_i = pontos[i]
        p_j = pontos[j]
        if tipo_distancia == 'mahalanobis':
            d = mahalanobis(p_i, p_j, inv_covmat)
        elif tipo_distancia == 'euclidiana':
            d = np.linalg.norm(p_i - p_j)
        else:
            raise ValueError("Tipo de distância inválido. Escolha 'mahalanobis' ou 'euclidiana'.")
        arestas.append((i, j, d))
    
    print(f"🔢 Total de nós: {len(pontos)}")
    print(f"🔗 Total de arestas: {len(arestas)}")
    
    return pontos, arestas

def plotar_grafo(pontos, arestas, df, tipo_distancia):
    """
    Plota o grafo de vizinhança utilizando Plotly.
    """
    fig = go.Figure()
    
    for (i, j, peso) in arestas:
        x_coords = [pontos[i][0], pontos[j][0], None]
        y_coords = [pontos[i][1], pontos[j][1], None]
        fig.add_trace(go.Scatter(
            x=x_coords, 
            y=y_coords,
            mode='lines',
            line=dict(width=1, color='gray'),
            hoverinfo='none'
        ))
    
    if 'genero' in df.columns:
        generos = df['genero']
        cores = {'male': 'blue', 'female': 'red'}
        cor_nos = [cores.get(g, 'green') for g in generos]
    else:
        cor_nos = 'blue'
    
    textos = []
    for idx, linha in df.iterrows():
        if 'genero' in df.columns:
            textos.append(f"Índice: {idx}<br>Gênero: {linha['genero']}")
        else:
            textos.append(f"Índice: {idx}")
    
    fig.add_trace(go.Scatter(
        x=pontos[:, 0],
        y=pontos[:, 1],
        mode='markers',
        marker=dict(
            size=8,
            color=cor_nos,
            line=dict(width=1, color='black')
        ),
        text=textos,
        hoverinfo='text'
    ))
    
    fig.update_layout(
        title=f"Grafo de Vizinhança ({tipo_distancia.capitalize()})",
        xaxis_title="PC1",
        yaxis_title="PC2",
        showlegend=False,
        hovermode='closest'
    )
    
    fig.show()

def grafo_vizinhanca():
    caminho_csv = 'Data\processed\dados_full_reduzidos.csv'  # Substitua pelo caminho correto
    tipo_distancia = 'mahalanobis'  # Altere para 'euclidiana' se desejar
    
    df = carregar_dados(caminho_csv)
    if df is None:
        return

    pontos, arestas = construir_grafo(df, tipo_distancia)
    plotar_grafo(pontos, arestas, df, tipo_distancia)

def metodo_silhueta(df, max_k=20):
    """
    Aplica o método da silhueta para determinar o número ideal de clusters.
    """
    X = df[['PC1', 'PC2']].values
    silhuetas = []

    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X)
        score = silhouette_score(X, labels)
        silhuetas.append(score)
        print(f"🔍 k = {k}, Silhouette Score = {score:.4f}")

    # Determina o k com o maior Silhouette Score
    k_otimo = np.argmax(silhuetas) + 2  # +2 porque o range começa em 2
    print(f"\n✅ Número ideal de clusters (k) com base no método da silhueta: {k_otimo}")

    # Plotando o gráfico da silhueta
    plt.figure(figsize=(8, 5))
    plt.plot(range(2, max_k + 1), silhuetas, 'bo-')
    plt.xlabel('Número de Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Método da Silhueta para Determinar k Ótimo')
    plt.grid(True)
    plt.show()

def silhueta():
    caminho_csv = 'Data\processed\dados_full_reduzidos.csv'  # Substitua pelo caminho correto
    df = carregar_dados(caminho_csv)
    if df is None:
        return

    metodo_silhueta(df, max_k=20)

def aplicar_kmeans(df, n_clusters=2):
    """
    Aplica o algoritmo K-Means para identificar clusters nos dados.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['cluster'] = kmeans.fit_predict(df[['PC1', 'PC2']])
    print(f"🔍 K-Means aplicado com {n_clusters} clusters.")
    return df, kmeans

def plotar_clusters_com_delaunay(df, kmeans):
    """
    Plota os clusters identificados e seus centróides,
    com triangulação de Delaunay sobre os pontos.
    """
    fig = px.scatter(df, x='PC1', y='PC2', color='cluster',
                     title='Clusters com K-Means + Triangulação de Delaunay',
                     labels={'PC1': 'Componente Principal 1', 'PC2': 'Componente Principal 2'},
                     color_continuous_scale=px.colors.qualitative.Set1)

    # Adiciona os centróides ao gráfico
    centroids = kmeans.cluster_centers_
    fig.add_trace(go.Scatter(x=centroids[:, 0], y=centroids[:, 1], mode='markers',
                             marker=dict(size=12, color='black', symbol='x'),
                             name='Centróides'))

    # Triangulação de Delaunay
    points = df[['PC1', 'PC2']].values
    tri = Delaunay(points)

    # Adiciona as linhas da triangulação
    for simplex in tri.simplices:
        pts = points[simplex]
        fig.add_trace(go.Scatter(x=np.append(pts[:, 0], pts[0, 0]),
                                 y=np.append(pts[:, 1], pts[0, 1]),
                                 mode='lines',
                                 line=dict(color='gray', width=0.5),
                                 showlegend=False))

    fig.show()

def agrupamento():
    caminho_csv = 'Data/processed/dados_full_reduzidos.csv'  # Substitua pelo caminho correto
    n_clusters = 2  # Número de clusters desejado

    df = carregar_dados(caminho_csv)
    if df is None:
        return

    df, kmeans = aplicar_kmeans(df, n_clusters)
    plotar_clusters_com_delaunay(df, kmeans)

if '__main__' == __name__:
    # # Gerado uma estrutura de vizinhanças com grafo conexo
    # grafo_vizinhanca()
    
    # # Aplicação do metodo de silhueta para encontrar o melhor numero de clusters
    # silhueta()

    # # Aplicação do metodo K-means em conjunto com diagrama de Voronoi para gerar um agrupamento 
    # agrupamento()

    main_svm() # Classificador binário com SVM

    # main_knn() # Classificador binário com Knn

    # main_Randon_forest() # Classificador binário com RandomForest
