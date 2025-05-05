import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def carregar_dados(caminho_csv):
    """
    Carrega os dados de um arquivo CSV.
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
    Separa features e rótulos, aplicando codificação nos rótulos.
    """
    colunas_excluir = ['genero', 'arquivo'] if 'arquivo' in df.columns else ['genero']
    colunas_features = [col for col in df.columns if col not in colunas_excluir]
    
    X = df[colunas_features].values
    y = df['genero'].values
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    return X, y_encoded, le

def dividir_dados(X, y, test_size = 0.3, random_state = 42):
    """
    Divide os dados em treino e teste com estratificação.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"🔢 Treino: {len(X_train)} amostras | Teste: {len(X_test)} amostras")
    return X_train, X_test, y_train, y_test

def treinar_rf(X_train, y_train, n_estimators = 100):
    """
    Treina um modelo Random Forest.
    """
    rf_model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    rf_model.fit(X_train, y_train)
    print("🤖 Modelo Random Forest treinado com sucesso!")
    return rf_model

def avaliar_modelo(model, X_test, y_test,label_encoder):
    """
    Avalia o modelo e exibe métricas.
    """
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"🎯 Acurácia: {acc * 100:.2f}%")
    
    print("\n📊 Relatório de Classificação:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    
    cm = confusion_matrix(y_test, y_pred)
    fig_cm = px.imshow(
        cm,
        labels=dict(x="Predito", y="Real", color="Contagem"),
        x=label_encoder.classes_,
        y=label_encoder.classes_,
        text_auto=True,
        color_continuous_scale='Blues'
    )
    fig_cm.update_layout(title="📌 Matriz de Confusão", xaxis_title="Predito", yaxis_title="Real")
    fig_cm.show()

    return y_pred

def plot_teste_pontos(X_test):
    """
    Plota apenas os pontos de teste em cinza.
    """
    fig = make_subplots(rows=1, cols=1,
                        subplot_titles=["📍 Dados de Teste (Não Classificados)"])
    
    fig.add_trace(
        go.Scatter(
            x=X_test[:, 0], y=X_test[:, 1],
            mode="markers",
            marker=dict(size=8, color="gray", symbol="diamond", line=dict(width=1)),
            text=["Teste: Não Classificado" for _ in range(X_test.shape[0])],
            name="Teste"
        )
    )
    
    fig.update_layout(title="🎨 Visualização de Dados de Teste", 
                      xaxis_title="PC1", yaxis_title="PC2")
    fig.show()

def plot_classificacao_resultados(X_test, y_test, y_pred, label_encoder):
    """
    Plota classificação do modelo com destaque para erros.
    """
    incorrect_mask = (y_pred != y_test)
    correct_count = (~incorrect_mask).sum()
    incorrect_count = incorrect_mask.sum()

    fig = make_subplots(rows=1, cols=1,
                        subplot_titles=["🔍 Resultado da Classificação"])
    
    # Pontos corretos
    correct_mask = ~incorrect_mask
    X_correct = X_test[correct_mask]
    y_correct = y_test[correct_mask]
    cores = {"male": "blue", "female": "red"}
    cores_corretos = [cores[lbl] for lbl in label_encoder.inverse_transform(y_correct)]
    
    fig.add_trace(
        go.Scatter(
            x=X_correct[:, 0], y=X_correct[:, 1],
            mode="markers",
            marker=dict(size=10, color=cores_corretos, symbol="diamond"),
            name=f"Corretos ({correct_count})",
            hovertext=[f"Real: {lbl}<br>Pred: {lbl}" 
                      for lbl in label_encoder.inverse_transform(y_correct)]
        )
    )

    # Pontos incorretos
    X_incorrect = X_test[incorrect_mask]
    y_incorrect_true = y_test[incorrect_mask]
    y_incorrect_pred = y_pred[incorrect_mask]
    
    fig.add_trace(
        go.Scatter(
            x=X_incorrect[:, 0], y=X_incorrect[:, 1],
            mode="markers",
            marker=dict(size=10, color="yellow", symbol="x", line=dict(width=1)),
            name=f"Incorretos ({incorrect_count})",
            hovertext=[f"Real: {label_encoder.inverse_transform([t])[0]}<br>Pred: {label_encoder.inverse_transform([p])[0]}" 
                      for t, p in zip(y_incorrect_true, y_incorrect_pred)]
        )
    )

    fig.update_layout(
        title="✅ Classificação Final (Apenas Teste)",
        xaxis_title="PC1", yaxis_title="PC2",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.show()

def validar_modelo_cross_validation(X_train, y_train, n_estimators = 100, cv = 5):
    """
    Realiza validação cruzada e retorna modelo treinado com todos os dados de treino.
    """
    print(f"\n🔁 Validação Cruzada ({cv}-folds) no Treino:")
    rf_model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    
    scores = cross_val_score(rf_model, X_train, y_train, cv=cv, scoring='accuracy')
    df_scores = pd.DataFrame({'Fold': range(1, cv+1), 'Acurácia': scores})
    
    # Gráfico de barras da validação cruzada
    fig = px.bar(df_scores, x='Fold', y='Acurácia', text_auto='.2f',
                 title='📊 Acurácia por Fold na Validação Cruzada',
                 labels={'Acurácia': 'Acurácia (%)'})
    fig.update_layout(yaxis_tickformat=".0%", yaxis_range=[0,1])
    fig.show()
    
    print(f" Média: {scores.mean()*100:.2f}% | Desvio: {scores.std()*100:.2f}%")
    
    # Treinamento final com todos os dados de treino
    rf_model.fit(X_train, y_train)
    return rf_model

def main_Randon_forest():
    caminho_csv = 'Data/processed/dados_full_reduzidos.csv'
    df = carregar_dados(caminho_csv)
    if df is None:
        return

    # Pré-processamento
    X, y, le = pre_processamento(df)
    
    # Divisão dos dados
    X_train, X_test, y_train, y_test = dividir_dados(X, y, test_size=0.3)
    
    # Validação cruzada e treinamento final
    modelo_final = validar_modelo_cross_validation(X_train, y_train, n_estimators=100, cv=5)
    
    # Avaliação no conjunto de teste
    print("\n Avaliação no Conjunto de Teste:")
    y_pred = avaliar_modelo(modelo_final, X_test, y_test, le)
    
    # Visualizações
    plot_teste_pontos(X_test)
    plot_classificacao_resultados(X_test, y_test, y_pred, le)
