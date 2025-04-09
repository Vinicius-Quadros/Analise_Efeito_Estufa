"""
Módulo para criação, treinamento e avaliação de modelos.

Este módulo contém funções para criar pipelines de modelo,
treinar, avaliar e fazer previsões com modelos de machine learning.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from typing import Tuple, List, Dict, Optional, Any, Union

from config import XGBOOST_PARAMS, PARAM_GRID, TARGET_COLUMN
from core.preprocessing import criar_preprocessador


def criar_baseline(X_train: pd.DataFrame, X_test: pd.DataFrame, 
                  y_train: pd.Series, y_test: pd.Series) -> Tuple[float, float]:
    """
    Cria um modelo baseline simples para comparação.
    
    Args:
        X_train: Features de treino
        X_test: Features de teste
        y_train: Target de treino
        y_test: Target de teste
        
    Returns:
        Tupla com (mse_baseline, rmse_baseline)
    """
    # Baseline simples: previsão pela média
    media_emissao = y_train.mean()
    y_pred_baseline = np.ones(len(y_test)) * media_emissao

    # Calculando erro do baseline
    mse_baseline = mean_squared_error(y_test, y_pred_baseline)
    rmse_baseline = np.sqrt(mse_baseline)

    print("\n" + "="*50)
    print("DESEMPENHO DO MODELO BASELINE:")
    print("="*50)
    print(f"MSE Baseline: {mse_baseline:.4f}")
    print(f"RMSE Baseline: {rmse_baseline:.4f}")
    print("="*50 + "\n")

    return mse_baseline, rmse_baseline


def criar_pipeline_xgboost(colunas_numericas: List[str], 
                           colunas_categoricas: List[str],
                           params: Dict = None) -> Pipeline:
    """
    Cria um pipeline completo com pré-processamento e modelo XGBoost.
    
    Args:
        colunas_numericas: Lista de nomes de colunas numéricas
        colunas_categoricas: Lista de nomes de colunas categóricas
        params: Parâmetros para o XGBoost (opcional)
        
    Returns:
        Pipeline sklearn configurado
    """
    # Obter preprocessador
    preprocessor = criar_preprocessador(colunas_numericas, colunas_categoricas)
    
    # Parâmetros do modelo
    model_params = XGBOOST_PARAMS.copy() if params is None else params
    
    # Criar pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', xgb.XGBRegressor(**model_params))
    ])
    
    return pipeline


def treinar_e_avaliar_modelo(pipeline: Pipeline, 
                            X_train: pd.DataFrame, X_test: pd.DataFrame,
                            y_train: pd.Series, y_test: pd.Series) -> Tuple:
    """
    Treina o modelo e avalia seu desempenho.
    
    Args:
        pipeline: Pipeline com preprocessador e modelo
        X_train: Features de treino
        X_test: Features de teste
        y_train: Target de treino
        y_test: Target de teste
        
    Returns:
        Tupla com (modelo_treinado, y_pred, mse, rmse, r2)
    """
    print("\n" + "="*50)
    print("TREINANDO E AVALIANDO O MODELO XGBOOST")
    print("="*50)
    
    try:
        # Treinar o modelo
        pipeline.fit(X_train, y_train)
        
        # Fazer previsões
        y_pred = pipeline.predict(X_test)
        
        # Calcular métricas
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        # Exibir resultados
        print("MÉTRICAS DE DESEMPENHO DO MODELO:")
        print(f"Erro Quadrático Médio (MSE): {mse:.4f}")
        print(f"Raiz do Erro Quadrático Médio (RMSE): {rmse:.4f}")
        print(f"Coeficiente de Determinação (R²): {r2:.4f}")
        print("="*50 + "\n")
        
        return pipeline, y_pred, mse, rmse, r2
        
    except Exception as e:
        print(f"ERRO durante o treinamento do modelo: {e}")
        
        # Debug adicional
        print(f"Shapes: X_train: {X_train.shape}, y_train: {y_train.shape}")
        print(f"Tipos: X_train dtype: {X_train.dtypes.iloc[0]}, y_train dtype: {y_train.dtype}")
        
        # Verificar valores problemáticos
        print("Verificando valores problemáticos em y_train...")
        problematicos = []
        for i, valor in enumerate(y_train):
            if np.isnan(valor) or np.isinf(valor):
                problematicos.append((i, valor))
            if len(problematicos) >= 5:
                break
        
        if problematicos:
            print(f"Valores problemáticos encontrados: {problematicos}")
        else:
            print("Nenhum valor problemático (NaN/Inf) encontrado nas primeiras posições.")
        
        return None, None, float('inf'), float('inf'), float('-inf')


def otimizar_hiperparametros(pipeline: Pipeline, 
                             X_train: pd.DataFrame, y_train: pd.Series,
                             param_grid: Dict = None,
                             cv: int = 3,
                             verbose: int = 1) -> Pipeline:
    """
    Realiza otimização de hiperparâmetros usando GridSearchCV.
    
    Args:
        pipeline: Pipeline com preprocessador e modelo
        X_train: Features de treino
        y_train: Target de treino
        param_grid: Grid de parâmetros para busca (opcional)
        cv: Número de folds para validação cruzada
        verbose: Nível de verbosidade (0=silencioso, >0=detalhado)
        
    Returns:
        Pipeline com os melhores parâmetros encontrados
    """
    print("\nIniciando otimização de hiperparâmetros...")
    
    # Grid de parâmetros
    grid = param_grid if param_grid is not None else PARAM_GRID
    
    # Configurando GridSearchCV
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=grid,
        scoring='neg_mean_squared_error',
        cv=cv,
        n_jobs=-1,
        verbose=verbose
    )
    
    # Executando busca em grid
    grid_search.fit(X_train, y_train)
    
    # Exibindo melhores parâmetros
    print(f"\nMelhores parâmetros encontrados: {grid_search.best_params_}")
    print(f"Melhor score: {-grid_search.best_score_:.4f} (MSE)")
    
    # Retornando melhor modelo
    return grid_search.best_estimator_


def realizar_previsoes(modelo: Pipeline, X_pred: pd.DataFrame) -> np.ndarray:
    """
    Realiza previsões usando o modelo treinado.
    
    Args:
        modelo: Modelo treinado (Pipeline)
        X_pred: DataFrame com features para previsão
        
    Returns:
        Array com as previsões
    """
    if modelo is None:
        print("ERRO: Modelo não fornecido ou inválido")
        return None
    
    try:
        # Realizar previsões
        y_pred = modelo.predict(X_pred)
        return y_pred
        
    except Exception as e:
        print(f"ERRO durante as previsões: {e}")
        return None


def prever_e_adicionar_resultados(df_original: pd.DataFrame, modelo: Pipeline, 
                                 gas_filtro: str, nome_coluna_previsao: str = 'emissao_prevista',
                                 preservar_formato: bool = True) -> pd.DataFrame:
    """
    Realiza previsões e adiciona ao DataFrame original.
    
    Args:
        df_original: DataFrame original com todos os dados
        modelo: Modelo treinado
        gas_filtro: Nome do gás para filtrar
        nome_coluna_previsao: Nome da coluna para armazenar as previsões
        preservar_formato: Se True, preserva o formato original dos números
        
    Returns:
        DataFrame com as previsões adicionadas
    """
    from core.data import filtrar_por_gas
    
    # Cria uma cópia do DataFrame original
    df_output = df_original.copy()
    
    # Inicializa a coluna de previsão com valores vazios
    df_output[nome_coluna_previsao] = None
    
    # Filtra apenas os dados do gás específico
    df_gas = filtrar_por_gas(df_original, gas_filtro)
    
    # Verifica se há dados para prever
    if df_gas.empty:
        print(f"ERRO: Não foram encontrados dados para o gás {gas_filtro}")
        return df_output
    
    print(f"Encontradas {len(df_gas)} linhas para o gás {gas_filtro}")
    
    # Features para previsão (sem a coluna target, se existir)
    X_pred = df_gas.drop(TARGET_COLUMN, axis=1) if TARGET_COLUMN in df_gas.columns else df_gas
    
    # Realizar as previsões
    print(f"\nRealizando previsões para {len(X_pred)} linhas...")
    y_pred = realizar_previsoes(modelo, X_pred)
    
    if y_pred is None:
        print("ERRO: Falha ao realizar previsões")
        return df_output
    
    print(f"Previsões realizadas com sucesso. Shape: {y_pred.shape}")
    
    # Adicionar previsões ao DataFrame original
    indices_gas = df_gas.index
    df_output.loc[indices_gas, nome_coluna_previsao] = y_pred
    
    # Preservar formato original (usar vírgula como separador decimal) se necessário
    if preservar_formato and TARGET_COLUMN in df_original.columns:
        amostra = df_original[TARGET_COLUMN].iloc[0] if len(df_original) > 0 else ''
        usa_virgula = isinstance(amostra, str) and ',' in amostra
        
        if usa_virgula:
            print("Formatando previsões com vírgula como separador decimal...")
            df_output[nome_coluna_previsao] = df_output[nome_coluna_previsao].apply(
                lambda x: str(x).replace('.', ',') if x is not None else None
            )
    
    return df_output