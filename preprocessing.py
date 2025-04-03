# Funções de pré-processamento
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from config import TARGET_GAS, TARGET_COLUMN, TEST_SIZE, RANDOM_STATE

def filtrar_n2o(df):
    """
    Filtra o dataframe para conter apenas dados de N2O.
    """
    # Verificar todos os valores únicos da coluna gas
    valores_unicos = df['gas'].unique()
    print(f"\nValores únicos na coluna 'gas':")
    for valor in valores_unicos:
        print(f"  - '{valor}'")
    
    # Tentar encontrar N2O usando correspondência parcial se o filtro exato não funcionar
    if TARGET_GAS not in valores_unicos:
        print(f"\nAviso: Não encontrei exatamente '{TARGET_GAS}', tentando correspondência parcial...")
        possiveis_n2o = [valor for valor in valores_unicos if 'N2O' in valor or 'Nitroso' in valor]
        if possiveis_n2o:
            print(f"Possíveis correspondências para N2O: {possiveis_n2o}")
            # Usar a primeira correspondência encontrada
            gas_a_usar = possiveis_n2o[0]
            print(f"Usando '{gas_a_usar}' como filtro")
            df_n2o = df[df['gas'] == gas_a_usar].copy()
        else:
            print(f"Nenhuma correspondência para N2O encontrada. Usando todo o conjunto de dados.")
            df_n2o = df.copy()
    else:
        df_n2o = df[df['gas'] == TARGET_GAS].copy()
    
    print(f"Dataset filtrado para N2O: {df_n2o.shape[0]} linhas")
    
    return df_n2o

def analise_exploratoria(df):
    """
    Realiza análise exploratória básica dos dados.
    """
    # Informações gerais do dataframe
    print("\nInformações do DataFrame:")
    print(df.info())
    
    # Estatísticas descritivas
    print("\nEstatísticas descritivas:")
    print(df.describe())
    
    # Verificando valores nulos
    print("\nValores nulos por coluna:")
    print(df.isnull().sum())
    
    # Verificando valores únicos nas colunas categóricas
    print("\nValores únicos em colunas categóricas:")
    for col in df.select_dtypes(include=['object']).columns:
        print(f"{col}: {df[col].nunique()} valores únicos")
    
    # Distribuição do gás N2O
    print("\nDistribuição do gás N2O:")
    print(df['gas'].value_counts())
    
    # Filtro para N2O
    df_n2o = filtrar_n2o(df)
    
    # Verificar amostra dos valores de emissão
    print("\nAmostra dos valores de emissão:")
    print(df_n2o[TARGET_COLUMN].head(10))
    
    # Converter emissão para float (substituindo vírgula por ponto)
    df_n2o = df_n2o.copy()
    if df_n2o[TARGET_COLUMN].dtype == 'object':
        try:
            # Substituir vírgula por ponto para conversão para float
            df_n2o[TARGET_COLUMN] = df_n2o[TARGET_COLUMN].str.replace(',', '.').astype(float)
            print("\nValores de emissão convertidos com sucesso para float")
        except ValueError as e:
            print(f"\nErro ao converter valores: {e}")
            print("Primeiros 10 valores problemáticos:")
            problematicos = []
            for i, valor in enumerate(df_n2o[TARGET_COLUMN].head(100)):
                try:
                    float(valor.replace(',', '.'))
                except:
                    problematicos.append(valor)
                    if len(problematicos) >= 10:
                        break
            print(problematicos)
    
    # Verificar distribuição da variável alvo
    plt.figure(figsize=(10, 6))
    try:
        plt.hist(df_n2o[TARGET_COLUMN], bins=30)
        plt.title(f'Distribuição dos valores de {TARGET_COLUMN} para N2O')
        plt.xlabel(f'Valor de {TARGET_COLUMN}')
        plt.ylabel('Frequência')
        plt.savefig('distribuicao_emissao_n2o.png')
    except Exception as e:
        print(f"Erro ao gerar histograma: {e}")
    
    return df_n2o

def preprocessamento(df_n2o):
    """
    Realiza o pré-processamento dos dados para o modelo.
    """
    # Verificar se o DataFrame está vazio
    if df_n2o.empty:
        print("ERRO: O DataFrame filtrado está vazio. Não é possível continuar.")
        print("Sugestão: Verifique o nome exato do gás no arquivo CSV.")
        return None, None, None, None, None, None
    
    # Converte a coluna de emissão para float, tratando possíveis erros
    if df_n2o[TARGET_COLUMN].dtype == 'object':
        print("Convertendo coluna de emissão para valores numéricos...")
        df_n2o[TARGET_COLUMN] = df_n2o[TARGET_COLUMN].str.replace(',', '.').astype(float)
    else:
        df_n2o[TARGET_COLUMN] = pd.to_numeric(df_n2o[TARGET_COLUMN], errors='coerce')
    
    # Trata valores nulos na variável alvo (se houver)
    if df_n2o[TARGET_COLUMN].isnull().sum() > 0:
        print(f"Removendo {df_n2o[TARGET_COLUMN].isnull().sum()} linhas com valores nulos na {TARGET_COLUMN}")
        df_n2o = df_n2o.dropna(subset=[TARGET_COLUMN])
    
    # Seleciona features e target
    X = df_n2o.drop(TARGET_COLUMN, axis=1)
    y = df_n2o[TARGET_COLUMN]
    
    # Identifica colunas numéricas e categóricas
    colunas_numericas = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    colunas_categoricas = X.select_dtypes(include=['object']).columns.tolist()
    
    print(f"Colunas numéricas: {colunas_numericas}")
    print(f"Colunas categóricas: {colunas_categoricas}")
    
    # Divisão em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    
    print(f"Conjunto de treino: {X_train.shape[0]} amostras")
    print(f"Conjunto de teste: {X_test.shape[0]} amostras")
    
    return X_train, X_test, y_train, y_test, colunas_numericas, colunas_categoricas

def criar_preprocessador(colunas_numericas, colunas_categoricas):
    """
    Cria um transformador para pré-processamento dos dados.
    """
    # Transformador para colunas numéricas
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median'))
    ])
    
    # Transformador para colunas categóricas
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combinando transformadores
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, colunas_numericas),
            ('cat', categorical_transformer, colunas_categoricas)
        ]
    )
    
    return preprocessor