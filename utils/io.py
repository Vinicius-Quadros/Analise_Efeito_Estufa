"""
Utilitários para operações de entrada e saída (I/O).

Este módulo contém funções para carregar dados de arquivos CSV,
salvar e carregar modelos, e outras operações relacionadas a I/O.
"""

import os
import pandas as pd
import joblib
from typing import Optional, Union, Dict, Any


def ensure_directory_exists(directory: str) -> None:
    """
    Garante que um diretório existe, criando-o se necessário.
    
    Args:
        directory: Caminho do diretório a ser verificado/criado
    """
    # Normaliza o caminho para o sistema operacional atual
    directory = os.path.normpath(directory)
    
    # Cria o diretório se não existir, incluindo diretórios pais
    if not os.path.exists(directory):
        try:
            os.makedirs(directory, exist_ok=True)
            print(f"Diretório criado: '{directory}'")
        except Exception as e:
            print(f"Erro ao criar diretório '{directory}': {e}")


def initialize_directories(directories: Dict[str, str]) -> None:
    """
    Inicializa todos os diretórios necessários para o projeto.
    
    Args:
        directories: Dicionário com nomes e caminhos dos diretórios
    """
    for dir_name, dir_path in directories.items():
        ensure_directory_exists(dir_path)
        print(f"Diretório '{dir_path}' verificado.")


def load_csv_data(filepath: str, verbose: bool = True) -> Optional[pd.DataFrame]:
    """
    Carrega dados de um arquivo CSV com tratamento para diferentes encodings.
    
    Args:
        filepath: Caminho do arquivo CSV
        verbose: Se True, exibe informações sobre os dados carregados
        
    Returns:
        DataFrame com os dados carregados ou None em caso de erro
    """
    # Normaliza o caminho para o sistema operacional atual
    filepath = os.path.normpath(filepath)
    
    try:
        # Primeira tentativa: UTF-8
        df = pd.read_csv(filepath, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            # Segunda tentativa: Latin-1
            if verbose:
                print("Erro com codificação utf-8, tentando latin-1...")
            df = pd.read_csv(filepath, encoding='latin-1')
        except Exception as e:
            print(f"Erro ao carregar o arquivo {filepath}: {e}")
            return None
    except Exception as e:
        print(f"Erro ao carregar o arquivo {filepath}: {e}")
        return None
        
    # Limpeza básica dos dados
    df.columns = df.columns.str.strip()
    
    # Limpeza de valores em colunas de texto
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].str.strip() if hasattr(df[col], 'str') else df[col]
    
    if verbose:
        print(f"Dados carregados: {df.shape[0]} linhas e {df.shape[1]} colunas")
        
        # Mostra valores únicos da coluna 'gas' se existir
        if 'gas' in df.columns:
            unique_gases = df['gas'].unique()
            print(f"\nValores únicos na coluna 'gas' (primeiros 10):")
            for i, valor in enumerate(unique_gases[:10]):
                print(f"  {i+1}. '{valor}'")
            print(f"\nTotal de valores únicos na coluna 'gas': {len(unique_gases)}")
    
    return df


def save_model(model: Any, filename_template: str, gas_name: str = 'default', 
              model_dir: str = 'modelos') -> str:
    """
    Salva um modelo treinado em disco.
    
    Args:
        model: Objeto do modelo a ser salvo
        filename_template: Template para o nome do arquivo
        gas_name: Nome do gás para formato do arquivo
        model_dir: Diretório onde o modelo será salvo
        
    Returns:
        Caminho completo onde o modelo foi salvo
    """
    # Criar diretório se não existir
    ensure_directory_exists(model_dir)
    
    # Formatar nome do arquivo
    if '{}' in filename_template:
        filename = filename_template.format(gas_name)
    else:
        filename = filename_template
    
    # Caminho completo - usando os.path.join para compatibilidade cross-platform
    full_path = os.path.join(model_dir, os.path.basename(filename))
    
    # Salvar modelo
    joblib.dump(model, full_path)
    print(f"Modelo salvo como '{full_path}'")
    
    return full_path


def load_model(filename_template: str, gas_name: str = 'default', 
              model_dir: str = 'modelos') -> Any:
    """
    Carrega um modelo previamente salvo.
    
    Args:
        filename_template: Template para o nome do arquivo
        gas_name: Nome do gás para formato do arquivo
        model_dir: Diretório onde o modelo foi salvo
        
    Returns:
        Objeto do modelo carregado
    """
    # Formatar nome do arquivo
    if '{}' in filename_template:
        filename = filename_template.format(gas_name)
    else:
        filename = filename_template
    
    # Verificar se o arquivo está na pasta de modelos
    model_path = os.path.join(model_dir, os.path.basename(filename))
    model_path = os.path.normpath(model_path)
    
    # Normaliza o caminho do arquivo base também
    filename = os.path.normpath(filename)
    
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        print(f"Modelo carregado de '{model_path}'")
    elif os.path.exists(filename):
        model = joblib.load(filename)
        print(f"Modelo carregado de '{filename}'")
    else:
        raise FileNotFoundError(f"Arquivo de modelo não encontrado: '{model_path}' ou '{filename}'")
    
    return model


def save_dataframe(df: pd.DataFrame, filename: str, directory: str = None) -> str:
    """
    Salva um DataFrame em um arquivo CSV.
    
    Args:
        df: DataFrame a ser salvo
        filename: Nome do arquivo
        directory: Diretório onde o arquivo será salvo (opcional)
        
    Returns:
        Caminho completo onde o arquivo foi salvo
    """
    # Determinar caminho completo
    if directory:
        ensure_directory_exists(directory)
        full_path = os.path.join(directory, filename)
    else:
        full_path = filename
    
    # Normaliza o caminho para o sistema operacional atual
    full_path = os.path.normpath(full_path)
    
    # Salvar arquivo
    try:
        df.to_csv(full_path, index=False)
        if os.path.exists(full_path):
            print(f"Arquivo salvo com sucesso em '{full_path}'.")
            print(f"Tamanho: {os.path.getsize(full_path)} bytes")
        else:
            print(f"Erro: Arquivo não foi criado em '{full_path}'")
    except Exception as e:
        print(f"Erro ao salvar arquivo em '{full_path}': {e}")
        return None
    
    return full_path