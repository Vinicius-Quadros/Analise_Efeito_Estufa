# Funções para visualização
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def analisar_importancia_features(modelo, X_train):
    """
    Analisa a importância das features usadas no modelo.
    """
    # Obtendo o modelo XGBoost do pipeline
    model_xgb = modelo.named_steps['model']
    
    # Obtendo importância das features
    importancia = model_xgb.feature_importances_
    
    # Criando DataFrame com importâncias
    df_importancia = pd.DataFrame({
        'Feature': range(len(importancia)),  # Índices temporários
        'Importância': importancia
    })
    
    # Ordenando por importância em ordem decrescente
    df_importancia = df_importancia.sort_values('Importância', ascending=False)
    
    # Exibindo as 15 features mais importantes
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importância', y='Feature', data=df_importancia.head(15))
    plt.title('15 Features Mais Importantes')
    plt.tight_layout()
    plt.savefig('importancia_features.png')
    
    print("\n15 Features mais importantes:")
    print(df_importancia.head(15))
    
    return df_importancia

def comparar_com_baseline(mse_modelo, mse_baseline):
    """
    Compara o desempenho do modelo com o baseline.
    """
    melhoria_percentual = ((mse_baseline - mse_modelo) / mse_baseline) * 100
    
    print("\nComparação com baseline:")
    print(f"MSE Modelo: {mse_modelo:.4f}")
    print(f"MSE Baseline: {mse_baseline:.4f}")
    print(f"Melhoria: {melhoria_percentual:.2f}%")
    
    # Visualização da comparação
    plt.figure(figsize=(8, 6))
    barras = plt.bar(['Baseline', 'XGBoost'], [mse_baseline, mse_modelo])
    barras[0].set_color('gray')
    barras[1].set_color('green')
    plt.ylabel('MSE (Erro Quadrático Médio)')
    plt.title('Comparação de Erro: Baseline vs XGBoost')
    plt.savefig('comparacao_baseline.png')
    
    return melhoria_percentual