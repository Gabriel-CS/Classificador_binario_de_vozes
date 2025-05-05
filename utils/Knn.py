import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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

def pre_processamento(df):
    """
    Separa as features e os rótulos.
    Considera que a coluna 'genero' contém os rótulos (male, female)
    e as demais colunas (exceto 'arquivo', se existir) são as features.
    Converte os rótulos para números utilizando LabelEncoder.
    """
    colunas_excluir = ['genero', 'arquivo'] if 'arquivo' in df.columns else ['genero']
    colunas_features = [col for col in df.columns if col not in colunas_excluir]
    
    X = df[colunas_features].values
    y = df['genero'].values
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    return X, y_encoded, le

def dividir_dados(X, y, test_size=0.3, random_state=42):
    """
    Divide os dados em conjuntos de treinamento e teste com estratificação.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"🔢 Conjunto de treinamento: {len(X_train)} amostras")
    print(f"🔢 Conjunto de teste: {len(X_test)} amostras")
    return X_train, X_test, y_train, y_test

def treinar_knn(X_train, y_train, k=3):
    """
    Treina o classificador KNN com k = 3.
    """
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(X_train, y_train)
    print("🤖 Modelo KNN treinado com sucesso!")
    return knn_model

def avaliar_modelo(model, X_test, y_test, label_encoder):
    """
    Avalia o desempenho do modelo utilizando acurácia, relatório de classificação
    e exibe a matriz de confusão interativa com Plotly.
    Retorna as predições realizadas para os dados de teste.
    """
    y_pred = model.predict(X_test)
    
    # Cálculo da acurácia
    acc = accuracy_score(y_test, y_pred)
    print(f"Acurácia: {acc * 100:.2f}%")
    
    # Relatório de classificação
    print("\nRelatório de Classificação:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    
    # Matriz de confusão interativa com Plotly Express
    cm = confusion_matrix(y_test, y_pred)
    fig_cm = px.imshow(cm,
                       labels=dict(x="Rótulo Predito", y="Rótulo Real", color="Contagem"),
                       x=label_encoder.classes_,
                       y=label_encoder.classes_,
                       text_auto=True,
                       color_continuous_scale='Blues')
    fig_cm.update_layout(title="Matriz de Confusão",
                         xaxis_title="Rótulo Predito",
                         yaxis_title="Rótulo Real")
    fig_cm.show()
    
    return y_pred

def plot_classificacao_resultados(X_test, y_test, y_pred, label_encoder):
    """
    Plota a classificação do modelo APENAS com os dados de teste:
    - Corretos em azul/vermelho (cores reais)
    - Incorretos em amarelo
    - Quantidade de pontos na legenda
    """
    # Identifica classificações incorretas
    incorrect_mask = (y_pred != y_test)
    X_incorrect = X_test[incorrect_mask]
    y_incorrect_true = y_test[incorrect_mask]
    y_incorrect_pred = y_pred[incorrect_mask]

    # Calcula quantidades
    correct_count = (~incorrect_mask).sum()
    incorrect_count = incorrect_mask.sum()

    # Cria figura
    fig = make_subplots(rows=1, cols=1,
                        subplot_titles=["Resultado da Classificação nos Dados de Teste"])

    # Dados de teste corretos
    correct_mask = ~incorrect_mask
    X_correct = X_test[correct_mask]
    y_correct_real = y_test[correct_mask]
    cores_treino = {"male": "blue", "female": "red"}
    cores_corretos = [cores_treino[lbl] for lbl in label_encoder.inverse_transform(y_correct_real)]
    
    fig.add_trace(
        go.Scatter(
            x=X_correct[:, 0],
            y=X_correct[:, 1],
            mode="markers",
            marker=dict(size=10, color=cores_corretos, symbol="diamond"),
            name=f"Teste Correto ({correct_count})",
            hovertext=[f"Real: {lbl}<br>Pred: {lbl}" 
                      for lbl in label_encoder.inverse_transform(y_correct_real)]
        )
    )

    # Dados de teste incorretos
    fig.add_trace(
        go.Scatter(
            x=X_incorrect[:, 0],
            y=X_incorrect[:, 1],
            mode="markers",
            marker=dict(size=10, color="yellow", symbol="x", line=dict(width=1)),
            name=f"Teste Incorreto ({incorrect_count})",
            hovertext=[f"Real: {label_encoder.inverse_transform([t])[0]}<br>Pred: {label_encoder.inverse_transform([p])[0]}" 
                      for t, p in zip(y_incorrect_true, y_incorrect_pred)]
        )
    )

    fig.update_layout(
        title="Visualização da Classificação (Apenas Teste)",
        xaxis_title="PC1", yaxis_title="PC2",
        legend=dict(
            orientation="h", 
            yanchor="bottom", 
            y=1.02, 
            xanchor="right", 
            x=1,
            bgcolor="rgba(255,255,255,0.8)"
        ),
        hoverlabel=dict(bgcolor="white", font_size=12)
    )
    fig.show()

def plot_teste_pontos(X_test):
    """
    Plota apenas os pontos de teste em cinza.
    """
    fig = make_subplots(rows=1, cols=1,
                        subplot_titles=["Dados de Teste (Não Classificados)"])
    
    fig.add_trace(
        go.Scatter(
            x=X_test[:, 0],
            y=X_test[:, 1],
            mode="markers",
            marker=dict(size=8, color="gray", symbol="diamond", line=dict(width=1, color="black")),
            text=["Teste: Não Classificado" for _ in range(X_test.shape[0])],
            name="Teste"
        )
    )
    
    fig.update_layout(title="Visualização dos Dados de Teste",
                      xaxis_title="PC1", yaxis_title="PC2")
    fig.show()

def validar_modelo_cross_validation(X_train, y_train, k = 3, cv = 5):
    """
    Realiza validação cruzada e retorna modelo treinado com todos os dados de treino.
    """
    print(f"\n🔁 Avaliando KNN com Validação Cruzada no Treino ({cv} folds)...")
    model = KNeighborsClassifier(n_neighbors=k)
    
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
    df_scores = pd.DataFrame({'Fold': range(1, cv+1), 'Acurácia': scores})
    
    # Gráfico de barras da validação cruzada
    fig = px.bar(df_scores, x='Fold', y='Acurácia', text_auto='.2f',
                 title='📊 Acurácia por Fold na Validação Cruzada',
                 labels={'Acurácia': 'Acurácia (%)'})
    fig.update_layout(yaxis_tickformat=".0%", yaxis_range=[0,1])
    fig.show()
    
    print(f" Média: {scores.mean()*100:.2f}% | Desvio: {scores.std()*100:.2f}%")
    
    # Treinamento final com todos os dados de treino
    model.fit(X_train, y_train)
    return model


def main_knn():
    caminho_csv = 'Data/processed/dados_full_reduzidos.csv'  # Altere para o caminho correto do seu arquivo
    df = carregar_dados(caminho_csv)
    if df is None:
        return
    
    # Pré-processamento: separa features e rótulos e codifica os rótulos
    X, y, le = pre_processamento(df)

    # Divisão dos dados em treino e teste
    X_train, X_test, y_train, y_test = dividir_dados(X, y, test_size=0.3)

    # Validação cruzada e treinamento do modelo final
    modelo_final = validar_modelo_cross_validation(X_train, y_train, k=5, cv=5)

    # Avaliação no conjunto de teste
    y_pred = avaliar_modelo(modelo_final, X_test, y_test, le)
    
    # Visualização dos resultados
    plot_teste_pontos(X_test)
    plot_classificacao_resultados(X_test, y_test, y_pred, le)
