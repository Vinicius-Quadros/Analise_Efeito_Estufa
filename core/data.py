"""
Módulo para processamento de dados.

Este módulo contém funções para trabalhar com dados, incluindo
filtragem por tipo de gás, identificação de gases e análise exploratória.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional, Union, Any

from config import POSSIBLE_GAS_FORMATS


def identificar_gases_no_dataset(df: pd.DataFrame) -> List[Tuple[str, str]]:
    """
    Identifica todos os tipos de gases presentes no dataset e mapeia para nomes padronizados.
    
    Args:
        df: DataFrame com os dados
        
    Returns:
        Lista de tuplas (nome_original, nome_padronizado)
    """
    gases_unicos = df['gas'].unique()
    print(f"Gases encontrados no dataset ({len(gases_unicos)} tipos):")
    
    for gas in gases_unicos:
        print(f"  - {gas}")
    
    # Mapear para nomes padronizados
    gases_mapeados = []
    for gas in gases_unicos:
        gas_padronizado = None
        for nome_padrao, formatos in POSSIBLE_GAS_FORMATS.items():
            if any(formato in gas for formato in formatos):
                gas_padronizado = nome_padrao
                break
        
        if gas_padronizado:
            gases_mapeados.append((gas, gas_padronizado))
        else:
            gases_mapeados.append((gas, gas))
    
    return gases_mapeados


def filtrar_por_gas(df: pd.DataFrame, gas_nome: str) -> pd.DataFrame:
    """
    Filtra o DataFrame para conter apenas dados do gás especificado.
    
    Args:
        df: DataFrame com os dados
        gas_nome: Nome do gás para filtrar (nome padronizado)
        
    Returns:
        DataFrame filtrado
    """
    # Verificar formatos possíveis para este gás
    formatos_possiveis = []
    for nome, formatos in POSSIBLE_GAS_FORMATS.items():
        if nome == gas_nome:
            formatos_possiveis = formatos
            break
    
    if not formatos_possiveis:
        formatos_possiveis = [gas_nome]  # Usar o nome literal se não houver formatos definidos
    
    # Verificar valores únicos na coluna gás
    valores_unicos = df['gas'].unique()
    
    # Tentar encontrar algum formato que exista no dataframe
    formato_encontrado = None
    for formato in formatos_possiveis:
        if formato in valores_unicos:
            formato_encontrado = formato
            break
    
    if formato_encontrado is None:
        # Tentar correspondência parcial
        for valor in valores_unicos:
            for formato in formatos_possiveis:
                if formato in valor:
                    formato_encontrado = valor
                    break
            if formato_encontrado:
                break
    
    if formato_encontrado is None:
        print(f"ERRO: Não foi possível encontrar o gás {gas_nome} nos dados.")
        return df.head(0)  # Retorna DataFrame vazio
    
    # Filtrar para o formato encontrado
    df_gas = df[df['gas'] == formato_encontrado].copy()
    print(f"Dataset filtrado para '{formato_encontrado}': {df_gas.shape[0]} linhas")
    
    return df_gas


def converter_para_numerico(serie: pd.Series) -> pd.Series:
    """
    Converte uma série para valores numéricos, tratando diferentes formatos.
    
    Args:
        serie: Série do pandas a ser convertida
        
    Returns:
        Série convertida para valores numéricos
    """
    if serie.dtype == 'object':
        # Para strings, tenta substituir vírgula por ponto primeiro
        return pd.to_numeric(serie.astype(str).str.replace(',', '.'), errors='coerce')
    else:
        # Para outros tipos, tenta converter diretamente
        return pd.to_numeric(serie, errors='coerce')


def analisar_distribuicao_por_gas(df: pd.DataFrame, coluna_valor: str) -> pd.DataFrame:
    """
    Analisa a distribuição de valores por tipo de gás.
    
    Args:
        df: DataFrame com os dados
        coluna_valor: Nome da coluna com os valores a serem analisados
        
    Returns:
        DataFrame com estatísticas por gás
    """
    # Garantir que a coluna é numérica
    df = df.copy()
    df[coluna_valor] = converter_para_numerico(df[coluna_valor])
    
    # Remover linhas com valores nulos
    df = df.dropna(subset=[coluna_valor])
    
    # Calcular estatísticas por gás
    stats = df.groupby('gas')[coluna_valor].agg([
        ('count', 'count'),
        ('min', 'min'),
        ('max', 'max'),
        ('mean', 'mean'),
        ('median', 'median'),
        ('std', 'std'),
        ('sum', 'sum')
    ]).sort_values('sum', ascending=False)
    
    return stats


def analisar_valores_problematicos(df: pd.DataFrame, coluna: str) -> pd.DataFrame:
    """
    Identifica valores problemáticos em uma coluna (NaN, infinitos, formatos inválidos).
    
    Args:
        df: DataFrame com os dados
        coluna: Nome da coluna a ser analisada
        
    Returns:
        DataFrame com linhas contendo valores problemáticos
    """
    # Cria cópia para não modificar o original
    df_temp = df.copy()
    
    # Tenta converter para numérico para identificar problemas
    if df[coluna].dtype == 'object':
        # Para valores de texto, converte substituindo vírgula por ponto
        df_temp['valor_numerico'] = pd.to_numeric(
            df_temp[coluna].astype(str).str.replace(',', '.'), 
            errors='coerce'
        )
    else:
        df_temp['valor_numerico'] = pd.to_numeric(df_temp[coluna], errors='coerce')
    
    # Identifica linhas com valores problemáticos
    problematicos = df_temp[df_temp['valor_numerico'].isna()].copy()
    
    # Adiciona indicador de problema
    if not problematicos.empty:
        print(f"Encontrados {len(problematicos)} valores problemáticos na coluna '{coluna}'")
        
    return problematicos


def realizar_analise_exploratoria(df: pd.DataFrame, target_column: str, 
                                 target_gas: Optional[str] = None) -> pd.DataFrame:
    """
    Realiza análise exploratória dos dados.
    
    Args:
        df: DataFrame com os dados
        target_column: Nome da coluna alvo
        target_gas: Nome do gás para filtrar (opcional)
        
    Returns:
        DataFrame filtrado e analisado
    """
    # Informações gerais
    print("\nInformações do DataFrame:")
    print(df.info())
    
    print("\nEstatísticas descritivas:")
    print(df.describe())
    
    print("\nValores nulos por coluna:")
    print(df.isnull().sum())
    
    # Verificando valores únicos nas colunas categóricas
    print("\nValores únicos em colunas categóricas:")
    for col in df.select_dtypes(include=['object']).columns:
        print(f"{col}: {df[col].nunique()} valores únicos")
    
    # Distribuição dos gases
    print("\nDistribuição dos gases:")
    print(df['gas'].value_counts())
    
    # Filtrar para o gás específico se solicitado
    if target_gas:
        df_gas = filtrar_por_gas(df, target_gas)
    else:
        df_gas = df.copy()
    
    # Verificar amostra dos valores da coluna alvo
    print(f"\nAmostra dos valores de {target_column}:")
    print(df_gas[target_column].head(10))
    
    # Converter coluna alvo para float se for texto
    if df_gas[target_column].dtype == 'object':
        try:
            df_gas[target_column] = converter_para_numerico(df_gas[target_column])
            print(f"\nValores de {target_column} convertidos com sucesso para float")
        except Exception as e:
            print(f"\nErro ao converter valores: {e}")
            
            # Identificar valores problemáticos
            problematicos = analisar_valores_problematicos(df_gas, target_column)
            if not problematicos.empty:
                print("Primeiros 10 valores problemáticos:")
                print(problematicos.head(10))
    
    # Verificar estatísticas da coluna alvo após conversão
    print(f"\nEstatísticas de {target_column} após processamento:")
    print(df_gas[target_column].describe())
    
    # Verificar se há valores extremos (outliers)
    Q1 = df_gas[target_column].quantile(0.25)
    Q3 = df_gas[target_column].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df_gas[(df_gas[target_column] < (Q1 - 1.5 * IQR)) | 
                     (df_gas[target_column] > (Q3 + 1.5 * IQR))]
    
    print(f"\nDetectados {len(outliers)} possíveis outliers em {target_column}")
    
    return df_gas