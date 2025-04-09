"""
Módulo para pré-processamento de dados.

Este módulo contém funções para preparar os dados para modelagem,
incluindo divisão treino/teste, imputação de valores faltantes,
codificação de variáveis categóricas, etc.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from typing import Tuple, List, Dict, Optional, Any

from config import TARGET_COLUMN, TEST_SIZE, RANDOM_STATE
from core.data import converter_para_numerico


def preparar_dataset(df: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepara o dataset para modelagem, convertendo tipos de dados e tratando valores nulos.
    
    Args:
        df: DataFrame com os dados
        target_column: Nome da coluna alvo
        
    Returns:
        Tupla com (X, y) - features e variável alvo
    """
    # Criar cópia para não modificar o original
    df_prep = df.copy()
    
    # Converter coluna alvo para tipo numérico
    df_prep[target_column] = converter_para_numerico(df_prep[target_column])
    
    # Remover linhas com valores nulos na coluna alvo
    df_prep = df_prep.dropna(subset=[target_column])
    
    # Separar features e target
    X = df_prep.drop(target_column, axis=1)
    y = df_prep[target_column]
    
    return X, y


def dividir_treino_teste(X: pd.DataFrame, y: pd.Series, 
                        test_size: float = TEST_SIZE, 
                        random_state: int = RANDOM_STATE) -> Tuple:
    """
    Divide os dados em conjuntos de treino e teste.
    
    Args:
        X: DataFrame com as features
        y: Série com a variável alvo
        test_size: Proporção do conjunto de teste
        random_state: Semente aleatória para reprodutibilidade
        
    Returns:
        Tupla com (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"Conjunto de treino: {X_train.shape[0]} amostras")
    print(f"Conjunto de teste: {X_test.shape[0]} amostras")
    
    return X_train, X_test, y_train, y_test


def identificar_tipos_colunas(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Identifica colunas numéricas e categóricas no DataFrame.
    
    Args:
        X: DataFrame com as features
        
    Returns:
        Tupla com (colunas_numericas, colunas_categoricas)
    """
    # Identificar colunas por tipo
    colunas_numericas = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    colunas_categoricas = X.select_dtypes(include=['object']).columns.tolist()
    
    print(f"Colunas numéricas ({len(colunas_numericas)}): {colunas_numericas}")
    print(f"Colunas categóricas ({len(colunas_categoricas)}): {colunas_categoricas}")
    
    return colunas_numericas, colunas_categoricas


def criar_preprocessador(colunas_numericas: List[str], 
                        colunas_categoricas: List[str],
                        estrategia_num: str = 'median',
                        estrategia_cat: str = 'most_frequent',
                        escalar_dados: bool = True) -> ColumnTransformer:
    """
    Cria um preprocessador para transformar colunas numéricas e categóricas.
    
    Args:
        colunas_numericas: Lista de nomes de colunas numéricas
        colunas_categoricas: Lista de nomes de colunas categóricas
        estrategia_num: Estratégia de imputação para colunas numéricas
        estrategia_cat: Estratégia de imputação para colunas categóricas
        escalar_dados: Se True, padroniza as colunas numéricas
        
    Returns:
        ColumnTransformer configurado para pré-processamento
    """
    # Transformador para colunas numéricas
    numeric_steps = []
    numeric_steps.append(('imputer', SimpleImputer(strategy=estrategia_num)))
    
    if escalar_dados:
        numeric_steps.append(('scaler', StandardScaler()))
    
    numeric_transformer = Pipeline(steps=numeric_steps)
    
    # Transformador para colunas categóricas
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy=estrategia_cat)),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combinando transformadores
    transformers = []
    
    if colunas_numericas:
        transformers.append(('num', numeric_transformer, colunas_numericas))
    
    if colunas_categoricas:
        transformers.append(('cat', categorical_transformer, colunas_categoricas))
    
    preprocessor = ColumnTransformer(transformers=transformers)
    
    return preprocessor


def preprocessar_dados_completo(df: pd.DataFrame, target_column: str) -> Tuple:
    """
    Realiza o pré-processamento completo dos dados em uma única função.
    
    Args:
        df: DataFrame com os dados
        target_column: Nome da coluna alvo
        
    Returns:
        Tupla com (X_train, X_test, y_train, y_test, colunas_numericas, colunas_categoricas)
    """
    # Verificar se DataFrame está vazio
    if df.empty:
        print("ERRO: O DataFrame está vazio. Não é possível continuar.")
        return None, None, None, None, None, None
    
    # Preparar dataset
    X, y = preparar_dataset(df, target_column)
    
    # Verificar se temos dados suficientes após preparação
    if len(y) < 10:  # Definir um mínimo arbitrário
        print(f"ERRO: Poucos dados disponíveis após preparação ({len(y)} amostras).")
        return None, None, None, None, None, None
    
    # Identificar tipos de colunas
    colunas_numericas, colunas_categoricas = identificar_tipos_colunas(X)
    
    # Dividir em treino e teste
    X_train, X_test, y_train, y_test = dividir_treino_teste(X, y)
    
    # Verificar qualidade dos dados
    if y_train.isna().any() or y_test.isna().any():
        print("ALERTA: Valores NaN encontrados na variável alvo após divisão.")
    
    return X_train, X_test, y_train, y_test, colunas_numericas, colunas_categoricas