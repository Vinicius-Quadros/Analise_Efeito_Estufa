"""
Módulo para análise de múltiplos gases.

Este módulo contém funções para treinar modelos para diferentes gases,
realizar previsões e comparar os resultados entre diferentes gases.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, List, Optional, Any, Union

from config import MODEL_FILENAME_TEMPLATE, TARGET_COLUMN, POSSIBLE_GAS_FORMATS
from core.data import identificar_gases_no_dataset, filtrar_por_gas, converter_para_numerico
from core.preprocessing import preprocessar_dados_completo
from core.model import criar_pipeline_xgboost, treinar_e_avaliar_modelo, realizar_previsoes
from utils.io import save_model, save_dataframe, ensure_directory_exists
from utils.visualization import plot_feature_importance, plot_model_comparison, plot_predictions_vs_actual


def treinar_modelos_para_todos_gases(df: pd.DataFrame, 
                                    gases_mapeados: List[Tuple[str, str]],
                                    min_samples: int = 1000) -> Tuple[Dict, Dict]:
    """
    Treina modelos separados para cada tipo de gás no dataset.
    
    Args:
        df: DataFrame com os dados
        gases_mapeados: Lista de tuplas (nome_original, nome_padronizado)
        min_samples: Número mínimo de amostras para treinar um modelo
        
    Returns:
        Tupla com (modelos, metricas)
    """
    modelos = {}
    metricas = {}
    
    for gas_original, gas_padronizado in gases_mapeados:
        print(f"\n{'='*50}")
        print(f"Treinando modelo para {gas_padronizado} (formato original: {gas_original})")
        print(f"{'='*50}")
        
        # Filtrar dados para este gás
        df_gas = df[df['gas'] == gas_original].copy()
        
        if len(df_gas) < min_samples:
            print(f"Poucos dados para {gas_original} ({len(df_gas)} amostras). Pulando...")
            continue
        
        # Preprocessamento
        resultado = preprocessar_dados_completo(df_gas, TARGET_COLUMN)
        
        if resultado[0] is None:
            print(f"Erro no preprocessamento para {gas_original}. Pulando...")
            continue
            
        X_train, X_test, y_train, y_test, colunas_numericas, colunas_categoricas = resultado
        
        # Criar pipeline
        pipeline = criar_pipeline_xgboost(colunas_numericas, colunas_categoricas)
        
        # Treinar e avaliar
        resultado_treino = treinar_e_avaliar_modelo(pipeline, X_train, X_test, y_train, y_test)
        
        if resultado_treino[0] is None:
            print(f"Falha ao treinar modelo para {gas_original}. Pulando...")
            continue
            
        modelo, y_pred, mse, rmse, r2 = resultado_treino
        
        # Visualização das previsões
        try:
            plot_predictions_vs_actual(
                y_test, 
                y_pred,
                title=f'Previsões vs Valores Reais - {gas_padronizado}',
                filename=f'previsoes_vs_reais_{gas_padronizado}.png'
            )
        except Exception as e:
            print(f"Erro ao criar gráfico de previsões: {e}")
        
        # Salvar modelo
        nome_arquivo = MODEL_FILENAME_TEMPLATE.format(gas_padronizado)
        save_model(modelo, nome_arquivo)
        
        # Armazenar modelo e métricas
        modelos[gas_padronizado] = modelo
        metricas[gas_padronizado] = {
            'mse': mse, 
            'rmse': rmse, 
            'r2': r2, 
            'samples': len(df_gas)
        }
    
    return modelos, metricas


def prever_para_todos_gases(df: pd.DataFrame, 
                           modelos: Dict, 
                           gases_mapeados: List[Tuple[str, str]]) -> pd.DataFrame:
    """
    Realiza previsões para todos os gases usando os modelos treinados.
    
    Args:
        df: DataFrame com os dados
        modelos: Dicionário com modelos treinados por gás
        gases_mapeados: Lista de tuplas (nome_original, nome_padronizado)
        
    Returns:
        DataFrame com previsões para todos os gases
    """
    df_resultado = df.copy()
    df_resultado['emissao_prevista'] = None
    
    for gas_original, gas_padronizado in gases_mapeados:
        if gas_padronizado not in modelos:
            print(f"Nenhum modelo disponível para {gas_padronizado}. Pulando...")
            continue
        
        print(f"\nRealizando previsões para {gas_padronizado}...")
        
        # Filtrar para este gás
        df_gas = df[df['gas'] == gas_original].copy()
        if df_gas.empty:
            print(f"Nenhum dado para {gas_original}. Pulando...")
            continue
        
        # Índices para este gás
        indices_gas = df_gas.index
        
        # Features para previsão
        X_pred = df_gas.drop(TARGET_COLUMN, axis=1) if TARGET_COLUMN in df_gas.columns else df_gas
        
        # Realizar previsões
        try:
            y_pred = modelos[gas_padronizado].predict(X_pred)
            
            # Adicionar previsões ao dataframe resultado
            df_resultado.loc[indices_gas, 'emissao_prevista'] = y_pred
            print(f"Previsões adicionadas para {len(indices_gas)} linhas de {gas_padronizado}")
        except Exception as e:
            print(f"Erro ao prever para {gas_padronizado}: {e}")
    
    # Formatar previsões se necessário (usando vírgula como separador decimal)
    if df[TARGET_COLUMN].dtype == 'object' and ',' in str(df[TARGET_COLUMN].iloc[0]):
        df_resultado['emissao_prevista'] = df_resultado['emissao_prevista'].apply(
            lambda x: str(x).replace('.', ',') if x is not None else None
        )
    
    # Salvar resultado
    save_dataframe(df_resultado, 'resultado_todos_gases.csv', 'resultados')
    print("\nResultados salvos em 'resultados/resultado_todos_gases.csv'")
    
    return df_resultado


def analisar_gases(df: pd.DataFrame, target_gas: str = 'N2O') -> Dict:
    """
    Realiza análise comparativa entre diferentes gases, com foco no gás alvo.
    
    Args:
        df: DataFrame com os dados e previsões
        target_gas: Gás alvo para análise detalhada
        
    Returns:
        Dicionário com resumo das análises
    """
    print(f"\n{'='*50}")
    print(f"ANÁLISE COMPARATIVA DE GASES (FOCO EM {target_gas})")
    print(f"{'='*50}")
    
    # Verificar se temos a coluna de previsões
    if 'emissao_prevista' not in df.columns:
        print("ERRO: Não há previsões no DataFrame. Execute a previsão primeiro.")
        return {'erro': 'Sem previsões no DataFrame'}
    
    # Converter colunas para numérico
    df_analise = df.copy()
    
    # Verificar tipos de dados
    print(f"Tipo de dados da coluna '{TARGET_COLUMN}': {df_analise[TARGET_COLUMN].dtype}")
    print(f"Tipo de dados da coluna 'emissao_prevista': {df_analise['emissao_prevista'].dtype}")
    
    # Aplicar conversão de forma segura
    df_analise[TARGET_COLUMN] = converter_para_numerico(df_analise[TARGET_COLUMN])
    df_analise['emissao_prevista'] = converter_para_numerico(df_analise['emissao_prevista'])
    
    # Remover valores nulos após conversão
    df_analise = df_analise.dropna(subset=[TARGET_COLUMN, 'emissao_prevista'])
    
    print(f"Dados após conversão e limpeza: {len(df_analise)} linhas")
    
    # Encontrar o nome original do gás alvo
    gas_alvo_original = None
    for gas, padronizado in identificar_gases_no_dataset(df):
        if padronizado == target_gas:
            gas_alvo_original = gas
            break
    
    if not gas_alvo_original:
        print(f"ERRO: Gás alvo {target_gas} não encontrado no dataset.")
        return {'erro': f'Gás {target_gas} não encontrado'}
    
    # Criar pasta para gráficos
    ensure_directory_exists('graficos')
    
    # Análise 1: Distribuição de emissões por tipo de gás
    plt.figure(figsize=(14, 8))
    sns.boxplot(x='gas', y=TARGET_COLUMN, data=df_analise)
    plt.title('Distribuição de Emissões por Tipo de Gás')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('graficos/distribuicao_emissoes_por_gas.png')
    plt.close()
    
    # Análise 2: Comparação entre valores reais e previstos para o gás alvo
    df_alvo = df_analise[df_analise['gas'] == gas_alvo_original]
    
    if len(df_alvo) > 0:
        plt.figure(figsize=(10, 6))
        plt.scatter(df_alvo[TARGET_COLUMN], df_alvo['emissao_prevista'], alpha=0.5)
        min_value = min(df_alvo[TARGET_COLUMN].min(), df_alvo['emissao_prevista'].min())
        max_value = max(df_alvo[TARGET_COLUMN].max(), df_alvo['emissao_prevista'].max())
        plt.plot([min_value, max_value], [min_value, max_value], 'r--')
        plt.xlabel('Emissão Real')
        plt.ylabel('Emissão Prevista')
        plt.title(f'Comparação entre Valores Reais e Previstos para {target_gas}')
        plt.savefig(f'graficos/comparacao_real_previsto_{target_gas}.png')
        plt.close()
    else:
        print(f"AVISO: Não há dados para o gás alvo {target_gas} após a conversão.")
    
    # Análise 3: Erro de previsão por setor para o gás alvo
    if 'Setor' in df_alvo.columns:
        df_alvo['erro_absoluto'] = abs(df_alvo[TARGET_COLUMN] - df_alvo['emissao_prevista'])
        
        plt.figure(figsize=(14, 8))
        erro_por_setor = df_alvo.groupby('Setor')['erro_absoluto'].mean().sort_values(ascending=False)
        sns.barplot(x=erro_por_setor.index, y=erro_por_setor.values)
        plt.title(f'Erro Médio de Previsão por Setor para {target_gas}')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(f'graficos/erro_por_setor_{target_gas}.png')
        plt.close()
    
    # Análise 4: Comparação da contribuição de cada gás para emissões totais
    emissoes_por_gas = df_analise.groupby('gas')[TARGET_COLUMN].sum()
    
    plt.figure(figsize=(10, 6))
    emissoes_por_gas.sort_values(ascending=False).plot(kind='bar')
    plt.title('Contribuição Total de Emissões por Tipo de Gás')
    plt.ylabel('Soma das Emissões')
    plt.tight_layout()
    plt.savefig('graficos/contribuicao_total_por_gas.png')
    plt.close()
    
    # Análise 5: Evolução temporal das emissões do gás alvo (se houver coluna 'ano')
    if 'ano' in df_alvo.columns:
        emissoes_tempo = df_alvo.groupby('ano')[TARGET_COLUMN].mean()
        previsoes_tempo = df_alvo.groupby('ano')['emissao_prevista'].mean()
        
        plt.figure(figsize=(12, 6))
        plt.plot(emissoes_tempo.index, emissoes_tempo.values, 'b-', label='Real')
        plt.plot(previsoes_tempo.index, previsoes_tempo.values, 'r--', label='Previsto')
        plt.title(f'Evolução Temporal das Emissões de {target_gas}')
        plt.xlabel('Ano')
        plt.ylabel('Emissão Média')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'graficos/evolucao_temporal_{target_gas}.png')
        plt.close()
    
    # Análise 6: Correlação entre diferentes variáveis para o gás alvo
    colunas_numericas = df_alvo.select_dtypes(include=['number']).columns
    
    if len(colunas_numericas) > 1:
        plt.figure(figsize=(12, 10))
        matriz_corr = df_alvo[colunas_numericas].corr()
        sns.heatmap(matriz_corr, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title(f'Matriz de Correlação para {target_gas}')
        plt.tight_layout()
        plt.savefig(f'graficos/matriz_correlacao_{target_gas}.png')
        plt.close()
    
    print(f"Análise concluída! Gráficos salvos na pasta 'graficos/'")
    
    # Calcular métricas para resumo
    try:
        # Evitar divisão por zero
        if len(df_alvo) > 0 and df_alvo[TARGET_COLUMN].sum() != 0:
            percentual_erro = abs(df_alvo[TARGET_COLUMN] - df_alvo['emissao_prevista']).sum() / df_alvo[TARGET_COLUMN].sum() * 100
        else:
            percentual_erro = 0
        
        # Retornar resumo das análises
        resumo = {
            'total_gases': len(df_analise['gas'].unique()),
            'total_emissoes_alvo': df_alvo[TARGET_COLUMN].sum() if len(df_alvo) > 0 else 0,
            'media_emissoes_alvo': df_alvo[TARGET_COLUMN].mean() if len(df_alvo) > 0 else 0,
            'erro_medio_alvo': abs(df_alvo[TARGET_COLUMN] - df_alvo['emissao_prevista']).mean() if len(df_alvo) > 0 else 0,
            'percentual_erro': percentual_erro
        }
        
        print("\nRESUMO DA ANÁLISE:")
        for key, value in resumo.items():
            print(f"  - {key}: {value}")
        
        return resumo
    except Exception as e:
        print(f"Erro ao gerar resumo da análise: {e}")
        return {'erro': str(e)}