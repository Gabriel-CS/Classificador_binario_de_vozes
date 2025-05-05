import os
import librosa
import numpy as np
import pandas as pd
from glob import glob

# Configurações
PASTA_PRINCIPAL = 'Data'
TAXA_AMOSTRAGEM = 16000
DURACAO_SEGMENTO = 1.0
LIMIAR_SILENCIO = 20

def preprocessamento(caminho_audio):
    """Aplica todas as etapas de pré-processamento"""
    # Carregar áudio
    y, sr = librosa.load(caminho_audio, sr=TAXA_AMOSTRAGEM)
    
    # 1. Remoção de ruído com filtro de pré-ênfase
    y = librosa.effects.preemphasis(y)  # [[6]]
    
    # 2. Normalização
    y = librosa.util.normalize(y)  # [[5]]
    
    # 3. Trimming (remoção de silêncio nas extremidades)
    y_trimmed, _ = librosa.effects.trim(y, top_db=LIMIAR_SILENCIO)  # [[2]]
    
    # 4. Segmentação em janelas de 1 segundo
    segmentos = []
    tamanho_amostra = int(TAXA_AMOSTRAGEM * DURACAO_SEGMENTO)
    
    for i in range(0, len(y_trimmed), tamanho_amostra):
        segmento = y_trimmed[i:i+tamanho_amostra]
        if len(segmento) == tamanho_amostra:
            segmentos.append(segmento)
    
    return segmentos

def extrair_features(segmento):
    """Extrai features expandidas com novas características"""
    # MFCCs com estatísticas detalhadas [[3]][[4]]
    mfccs = librosa.feature.mfcc(y=segmento, sr=TAXA_AMOSTRAGEM, n_mfcc=13)
    
    # Novas características de MFCC
    mfcc_features = []
    for i in range(mfccs.shape[0]):
        mfcc_features.extend([
            np.mean(mfccs[i, :]),    # Média
            np.std(mfccs[i, :]),     # Desvio padrão [[6]]
            np.min(mfccs[i, :]),     # Mínimo
            np.max(mfccs[i, :])      # Máximo
        ])
    
    # Características de graves e agudos [[2]]
    bass_mean = np.mean(mfccs[:4, :])   # Primeiros 4 MFCCs (graves)
    treble_mean = np.mean(mfccs[4:, :]) # Demais MFCCs (agudos)

    # Outras features mantidas
    zcr = librosa.feature.zero_crossing_rate(segmento).mean()
    spectral_centroid = librosa.feature.spectral_centroid(y=segmento, sr=TAXA_AMOSTRAGEM).mean()
    pitch = np.nanmedian(librosa.core.piptrack(y=segmento, sr=TAXA_AMOSTRAGEM)[0])
    
    return np.hstack([mfcc_features, bass_mean, treble_mean, zcr, spectral_centroid, pitch])

# Processamento completo
dados = []
rotulos = []

for genero in ['male', 'female']:
    caminhos = glob(os.path.join(PASTA_PRINCIPAL, genero, '*.wav'))
    
    for caminho in caminhos:
        print(f"Processando: {caminho}")
        segmentos = preprocessamento(caminho)
        
        for seg in segmentos:
            features = extrair_features(seg)
            dados.append(features)
            rotulos.append(genero)

# Criação do DataFrame com novas colunas [[1]][[4]]
colunas = []
for i in range(1, 14):
    colunas.extend([f'MFCC{i}_mean', f'MFCC{i}_std', f'MFCC{i}_min', f'MFCC{i}_max'])
colunas.extend(['BassMean', 'TrebleMean', 'ZCR', 'SpectralCentroid', 'Pitch'])

df = pd.DataFrame(dados, columns=colunas)
df['label'] = rotulos

# Salvar CSV
df.to_csv('processed_audio_features.csv', index=False)
print("Processamento concluído! Arquivo CSV gerado.")
