"""
Utilitários para visualização de dados e resultados.

Este módulo contém funções para criar gráficos, visualizar importância de features,
comparar modelos, e outras operações relacionadas à visualização.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
from typing import Dict, Tuple, List, Optional, Any

from utils.io import ensure_directory_exists


def save_figure(fig: plt.Figure, filename: str, directory: str = 'graficos', 
               dpi: int = 300, close_fig: bool = True) -> str:
    """
    Salva uma figura matplotlib em um arquivo.
    
    Args:
        fig: Objeto Figure do matplotlib
        filename: Nome do arquivo
        directory: Diretório onde o arquivo será salvo
        dpi: Resolução da imagem
        close_fig: Se True, fecha a figura após salvar
        
    Returns:
        Caminho completo onde a figura foi salva
    """
    ensure_directory_exists(directory)
    filepath = os.path.join(directory, filename)
    filepath = os.path.normpath(filepath)
    
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
    print(f"Gráfico salvo em '{filepath}'")
    
    if close_fig:
        plt.close(fig)
    
    return filepath


def plot_feature_importance(model: Any, X_train: pd.DataFrame, 
                           n_features: int = 15, 
                           filename: str = 'importancia_features.png') -> pd.DataFrame:
    """
    Plota e salva um gráfico de importância de features.
    
    Args:
        model: Modelo treinado (deve ser um pipeline com XGBoost)
        X_train: Dados de treino
        n_features: Número de features para mostrar
        filename: Nome do arquivo para salvar o gráfico
        
    Returns:
        DataFrame com importâncias das features
    """
    # Obter o modelo XGBoost do pipeline
    model_xgb = model.named_steps['model']
    
    # Obter importâncias das features
    importances = model_xgb.feature_importances_
    
    # Criar DataFrame com importâncias
    feature_df = pd.DataFrame({
        'Feature': range(len(importances)),
        'Importância': importances
    })
    
    # Ordenar por importância em ordem decrescente
    feature_df = feature_df.sort_values('Importância', ascending=False)
    
    # Criar figura para as n_features mais importantes
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importância', y='Feature', data=feature_df.head(n_features))
    plt.title(f'Top {n_features} Features Mais Importantes')
    plt.tight_layout()
    
    # Salvar figura
    save_figure(plt.gcf(), filename)
    
    # Exibir top features
    print(f"\n{n_features} Features mais importantes:")
    print(feature_df.head(n_features))
    
    return feature_df


def plot_model_comparison(mse_model: float, mse_baseline: float, 
                         filename: str = 'comparacao_baseline.png') -> float:
    """
    Plota e salva um gráfico comparando o modelo com o baseline.
    
    Args:
        mse_model: MSE do modelo
        mse_baseline: MSE do baseline
        filename: Nome do arquivo para salvar o gráfico
        
    Returns:
        Melhoria percentual do modelo em relação ao baseline
    """
    # Calcular melhoria percentual
    improvement = ((mse_baseline - mse_model) / mse_baseline) * 100
    
    # Criar figura
    plt.figure(figsize=(8, 6))
    bars = plt.bar(['Baseline', 'XGBoost'], [mse_baseline, mse_model])
    bars[0].set_color('gray')
    bars[1].set_color('green')
    
    # Adicionar valores nas barras
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.2f}',
                 ha='center', va='bottom')
    
    plt.ylabel('MSE (Erro Quadrático Médio)')
    plt.title('Comparação de Erro: Baseline vs XGBoost')
    
    # Adicionar texto com a melhoria
    plt.figtext(0.5, 0.01, f'Melhoria: {improvement:.2f}%', 
               ha='center', fontsize=12)
    
    # Salvar figura
    save_figure(plt.gcf(), filename)
    
    # Exibir resultados
    print("\n" + "="*50)
    print("COMPARAÇÃO COM BASELINE:")
    print("="*50)
    print(f"MSE Modelo: {mse_model:.4f}")
    print(f"MSE Baseline: {mse_baseline:.4f}")
    print(f"Melhoria percentual: {improvement:.2f}%")
    print("="*50 + "\n")
    
    return improvement


def plot_predictions_vs_actual(y_true: np.ndarray, y_pred: np.ndarray, 
                              title: str = 'Previsões vs Valores Reais',
                              filename: str = 'previsoes_vs_reais.png') -> None:
    """
    Plota e salva um gráfico de valores previstos vs valores reais.
    
    Args:
        y_true: Valores reais
        y_pred: Valores previstos
        title: Título do gráfico
        filename: Nome do arquivo para salvar o gráfico
    """
    plt.figure(figsize=(10, 6))
    
    # Criar scatter plot
    scatter = plt.scatter(y_true, y_pred, alpha=0.5, c='blue', edgecolors='k')
    
    # Linha de referência (y=x)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Referência (y=x)')
    
    # Calcular métricas para mostrar no gráfico
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Adicionar métricas ao gráfico
    plt.annotate(f'MSE: {mse:.4f}\nR²: {r2:.4f}', 
                xy=(0.05, 0.95), xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    plt.xlabel('Valores Reais')
    plt.ylabel('Previsões')
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Salvar figura
    save_figure(plt.gcf(), filename)
    
    # Imprimir no terminal também
    print("\n" + "="*50)
    print("AVALIAÇÃO DE PREVISÕES:")
    print("="*50)
    print(f"MSE entre previsões e valores reais: {mse:.4f}")
    print(f"R² entre previsões e valores reais: {r2:.4f}")
    print(f"Gráfico salvo como '{filename}' na pasta de gráficos")
    print("="*50)


def plot_distribution(data: np.ndarray, column_name: str, 
                     title: str = None,
                     filename: str = 'distribuicao.png') -> None:
    """
    Plota e salva um histograma da distribuição de uma variável.
    
    Args:
        data: Dados para o histograma
        column_name: Nome da coluna/variável
        title: Título do gráfico (opcional)
        filename: Nome do arquivo para salvar o gráfico
    """
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=30)
    
    if title is None:
        title = f'Distribuição de {column_name}'
        
    plt.title(title)
    plt.xlabel(column_name)
    plt.ylabel('Frequência')
    
    # Salvar figura
    save_figure(plt.gcf(), filename)


def plot_time_series(time_index: np.ndarray, actual_values: np.ndarray, 
                    predicted_values: Optional[np.ndarray] = None,
                    title: str = 'Série Temporal',
                    x_label: str = 'Tempo',
                    y_label: str = 'Valor',
                    filename: str = 'serie_temporal.png') -> None:
    """
    Plota e salva um gráfico de série temporal.
    
    Args:
        time_index: Índice temporal (eixo x)
        actual_values: Valores reais
        predicted_values: Valores previstos (opcional)
        title: Título do gráfico
        x_label: Rótulo do eixo x
        y_label: Rótulo do eixo y
        filename: Nome do arquivo para salvar o gráfico
    """
    plt.figure(figsize=(12, 6))
    
    # Plotar valores reais
    plt.plot(time_index, actual_values, 'b-', label='Real')
    
    # Plotar valores previstos, se fornecidos
    if predicted_values is not None:
        plt.plot(time_index, predicted_values, 'r--', label='Previsto')
        plt.legend()
    
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)
    
    # Salvar figura
    save_figure(plt.gcf(), filename)


def plot_correlation_matrix(df: pd.DataFrame, 
                           title: str = 'Matriz de Correlação',
                           filename: str = 'matriz_correlacao.png') -> None:
    """
    Plota e salva uma matriz de correlação.
    
    Args:
        df: DataFrame com as variáveis numéricas
        title: Título do gráfico
        filename: Nome do arquivo para salvar o gráfico
    """
    # Selecionar apenas colunas numéricas
    numeric_df = df.select_dtypes(include=['number'])
    
    # Calcular matriz de correlação
    corr_matrix = numeric_df.corr()
    
    # Criar figura
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title(title)
    plt.tight_layout()
    
    # Salvar figura
    save_figure(plt.gcf(), filename)