# Funções utilitárias
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os
from config import MODEL_FILENAME_TEMPLATE


def carregar_dados(arquivo):
    """
    Carrega os dados do CSV e faz tratamentos iniciais básicos.
    """
    # Carrega o arquivo CSV
    try:
        df = pd.read_csv(arquivo, encoding='utf-8')
    except UnicodeDecodeError:
        # Se falhar com utf-8, tenta outros encodings comuns
        print("Erro com codificação utf-8, tentando latin-1...")
        df = pd.read_csv(arquivo, encoding='latin-1')
    
    # Trata espaços em branco nos nomes das colunas
    df.columns = df.columns.str.strip()
    
    # Trata espaços em branco nos valores
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].str.strip() if df[col].dtype == 'object' else df[col]
    
    # Mostra algumas informações sobre os dados
    print(f"Dados carregados: {df.shape[0]} linhas e {df.shape[1]} colunas")
    
    # Verificar os primeiros valores da coluna 'gas'
    if 'gas' in df.columns:
        print(f"\nValores únicos na coluna 'gas' (primeiros 10):")
        valores_unicos = df['gas'].unique()
        for i, valor in enumerate(valores_unicos[:10]):
            print(f"  {i+1}. '{valor}'")
        
        print(f"\nTotal de valores únicos na coluna 'gas': {len(valores_unicos)}")
    
    return df


def salvar_modelo(modelo, nome_arquivo=MODEL_FILENAME_TEMPLATE):
    """
    Salva o modelo treinado em disco na pasta 'modelos'.
    """
    # Criar pasta 'modelos' se não existir
    os.makedirs('modelos', exist_ok=True)
    
    # Garantir que o nome do arquivo está formatado corretamente
    if isinstance(nome_arquivo, str) and '{}' in nome_arquivo:
        nome_arquivo = nome_arquivo.format('default')
    
    # Adicionar caminho para a pasta 'modelos'
    caminho_completo = os.path.join('modelos', os.path.basename(nome_arquivo))
    
    joblib.dump(modelo, caminho_completo)
    print(f"\nModelo salvo como '{caminho_completo}'")


def carregar_modelo(nome_arquivo=MODEL_FILENAME_TEMPLATE):
    """
    Carrega um modelo previamente salvo.
    """
    # Garantir que o nome do arquivo está formatado corretamente
    if isinstance(nome_arquivo, str) and '{}' in nome_arquivo:
        nome_arquivo = nome_arquivo.format('default')
    
    # Verificar se o arquivo está na pasta 'modelos'
    caminho_modelos = os.path.join('modelos', os.path.basename(nome_arquivo))
    
    if os.path.exists(caminho_modelos):
        modelo = joblib.load(caminho_modelos)
        print(f"\nModelo carregado de '{caminho_modelos}'")
    else:
        modelo = joblib.load(nome_arquivo)
        print(f"\nModelo carregado de '{nome_arquivo}'")
    
    return modelo


def criar_baseline(X_train, X_test, y_train, y_test):
    """
    Cria um modelo baseline simples para comparação.
    """
    # Baseline simples: previsão pela média
    media_emissao = y_train.mean()
    y_pred_baseline = np.ones(len(y_test)) * media_emissao

    # Calculando erro do baseline
    from sklearn.metrics import mean_squared_error
    mse_baseline = mean_squared_error(y_test, y_pred_baseline)
    rmse_baseline = np.sqrt(mse_baseline)

    print("\nDesempenho do modelo baseline:")
    print(f"MSE Baseline: {mse_baseline:.4f}")
    print(f"RMSE Baseline: {rmse_baseline:.4f}")

    return mse_baseline, rmse_baseline