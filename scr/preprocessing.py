import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# script normalizador dos dados
def carregar_csv(caminho_csv):
    """
    Carrega o arquivo CSV contendo os dados completos.
    """
    try:
        df = pd.read_csv(caminho_csv)
        print("Dados carregados com sucesso!")
        return df
    except Exception as e:
        print(f"Erro ao carregar o arquivo: {e}")
        return None

def normalizar_features(df, colunas_feature):
    """
    Normaliza as features utilizando o StandardScaler (Z-Score),
    transformando os dados para média 0 e desvio padrão 1.
    """
    scaler = StandardScaler()
    df[colunas_feature] = scaler.fit_transform(df[colunas_feature])
    return df, scaler

def main_normalize():
    # Defina o caminho para o arquivo CSV completo
    caminho_csv = 'Data\processed\processed_audio_features.csv'  # altere para o diretório correto
    
    # Carrega os dados
    df = carregar_csv(caminho_csv)
    if df is None:
        return
    
    # Identifica as colunas que serão normalizadas
    # Supondo que 'genero' seja a coluna de rótulo e,
    # se existir, 'arquivo' não será normalizada.
    colunas_excluir = ['genero', 'arquivo'] if 'arquivo' in df.columns else ['genero']
    colunas_feature = [col for col in df.columns if col not in colunas_excluir]
    
    # Normaliza as colunas de features
    df_normalizado, scaler = normalizar_features(df.copy(), colunas_feature)
    
    # Salva o DataFrame normalizado em um novo arquivo CSV
    novo_csv = os.path.join(os.path.dirname(caminho_csv), 'dados_full_normalized.csv')
    df_normalizado.to_csv(novo_csv, index=False)
    print(f"\nArquivo normalizado salvo em: {novo_csv}")

def carregar_csv(caminho_csv):
    """
    Carrega o arquivo CSV contendo os dados normalizados.
    """
    try:
        df = pd.read_csv(caminho_csv)
        print("Dados carregados com sucesso!")
        return df
    except Exception as e:
        print(f"Erro ao carregar o arquivo: {e}")
        return None

def aplicar_pca(df, colunas_feature, n_components=0.95):
    """
    Aplica o PCA para redução de dimensionalidade.
    n_components pode ser um inteiro (número de componentes) ou um float indicando a porcentagem de variância a ser mantida.
    Retorna o DataFrame com as componentes principais e o objeto PCA.
    """
    pca = PCA(n_components=n_components)
    componentes_principais = pca.fit_transform(df[colunas_feature])
    
    # Cria um DataFrame com as componentes principais
    colunas_pca = [f'PC{i+1}' for i in range(componentes_principais.shape[1])]
    df_pca = pd.DataFrame(componentes_principais, columns=colunas_pca)
    
    # Se existir a coluna rótulo ('genero'), adiciona-a ao DataFrame resultante
    if 'genero' in df.columns:
        df_pca['genero'] = df['genero'].values

    print("Explicação da variância para cada componente principal:")
    for i, perc in enumerate(pca.explained_variance_ratio_):
        print(f"PC{i+1}: {perc:.4f}")
    
    print(f"Variância acumulada: {pca.explained_variance_ratio_.cumsum()[-1]:.4f}")
    return df_pca, pca

def main_pca():
    # Defina o caminho para o CSV com os dados normalizados
    caminho_csv = 'Data\processed\dados_full_normalized.csv'  # altere para o caminho correto
    
    # Carrega os dados
    df = carregar_csv(caminho_csv)
    if df is None:
        return
    
    # Definindo as colunas de features para o PCA.
    # Excluímos a coluna 'genero' (e 'arquivo' se existir) para aplicar o PCA somente nas variáveis numéricas.
    colunas_excluir = ['genero', 'arquivo'] if 'arquivo' in df.columns else ['genero']
    colunas_feature = [col for col in df.columns if col not in colunas_excluir]
    
    print("Aplicando o PCA...")
    # n_components pode ser ajustado. Aqui, usamos 0.95 para manter 95% da variância total.
    df_pca, pca = aplicar_pca(df, colunas_feature, n_components=0.95)
    
    # Salva o DataFrame com as componentes principais em um novo CSV
    pasta_csv = os.path.dirname(caminho_csv)
    novo_csv = os.path.join(pasta_csv, 'dados_full_normalized_pca.csv')
    df_pca.to_csv(novo_csv, index=False)
    print(f"\nArquivo com dados reduzidos salvo em: {novo_csv}")
