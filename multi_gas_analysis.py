# Funções para análise de múltiplos gases
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from preprocessing import filtrar_gas, preprocessamento
from model import criar_pipeline, treinar_e_avaliar, prever_em_novos_dados
from utils import criar_baseline, salvar_modelo, carregar_modelo
from config import POSSIBLE_GAS_FORMATS, MODEL_FILENAME_TEMPLATE, TARGET_COLUMN
from visualization import comparar_com_baseline

def identificar_gases_no_dataset(df):
    """
    Identifica todos os tipos de gases presentes no dataset.
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

def treinar_modelos_para_todos_gases(df, gases_mapeados, min_samples=1000):
    """
    Treina modelos separados para cada tipo de gás no dataset.
    Retorna um dicionário com os modelos treinados.
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
        X_train, X_test, y_train, y_test, colunas_numericas, colunas_categoricas = preprocessamento(df_gas)
        
        if X_train is None:
            print(f"Erro no preprocessamento para {gas_original}. Pulando...")
            continue
        
        # Criar e treinar modelo
        pipeline = criar_pipeline(colunas_numericas, colunas_categoricas)
        modelo, y_pred, mse, rmse = treinar_e_avaliar(pipeline, X_train, X_test, y_train, y_test)
        
        if modelo is None:
            print(f"Falha ao treinar modelo para {gas_original}. Pulando...")
            continue

        # Adicionar esta parte para gerar a comparação com baseline
        mse_baseline, _ = criar_baseline(X_train, X_test, y_train, y_test)
        melhoria = comparar_com_baseline(mse, mse_baseline)
        print(f"Gráfico de comparação com baseline salvo como 'comparacao_baseline.png' para {gas_padronizado}")
        
        # Salvar modelo
        nome_arquivo = MODEL_FILENAME_TEMPLATE.format(gas_padronizado)
        salvar_modelo(modelo, nome_arquivo)
        
        # Armazenar modelo e métricas
        modelos[gas_padronizado] = modelo
        metricas[gas_padronizado] = {'mse': mse, 'rmse': rmse, 'samples': len(df_gas)}
    
    return modelos, metricas

def prever_para_todos_gases(df, modelos, gases_mapeados):
    """
    Realiza previsões para todos os gases usando os modelos treinados.
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
    df_resultado.to_csv('resultado_todos_gases.csv', index=False)
    print("\nResultados salvos em 'resultado_todos_gases.csv'")
    
    return df_resultado

def analisar_gases(df, target_gas='N2O'):
    """
    Realiza análise comparativa entre diferentes gases, com foco no gás alvo.
    """
    print(f"\n{'='*50}")
    print(f"ANÁLISE COMPARATIVA DE GASES (FOCO EM {target_gas})")
    print(f"{'='*50}")
    
    # Verificar se temos a coluna de previsões
    if 'emissao_prevista' not in df.columns:
        print("ERRO: Não há previsões no DataFrame. Execute a previsão primeiro.")
        return
    
    # Converter colunas para numérico
    df_analise = df.copy()
    
    # Verificar tipos de dados
    print(f"Tipo de dados da coluna '{TARGET_COLUMN}': {df_analise[TARGET_COLUMN].dtype}")
    print(f"Tipo de dados da coluna 'emissao_prevista': {df_analise['emissao_prevista'].dtype}")
    
    # Função segura para converter para numérico
    def converter_para_numerico(serie):
        if serie.dtype == 'object':
            # Para strings, tenta substituir vírgula por ponto primeiro
            return pd.to_numeric(serie.astype(str).str.replace(',', '.'), errors='coerce')
        else:
            # Para outros tipos, tenta converter diretamente
            return pd.to_numeric(serie, errors='coerce')
    
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
        return
    
    # Criar pasta para gráficos
    import os
    os.makedirs('graficos', exist_ok=True)
    
    # Análise 1: Distribuição de emissões por tipo de gás
    plt.figure(figsize=(14, 8))
    sns.boxplot(x='gas', y=TARGET_COLUMN, data=df_analise)
    plt.title('Distribuição de Emissões por Tipo de Gás')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('graficos/distribuicao_emissoes_por_gas.png')
    
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
    
    # Análise 4: Comparação da contribuição de cada gás para emissões totais
    emissoes_por_gas = df_analise.groupby('gas')[TARGET_COLUMN].sum()
    
    plt.figure(figsize=(10, 6))
    emissoes_por_gas.sort_values(ascending=False).plot(kind='bar')
    plt.title('Contribuição Total de Emissões por Tipo de Gás')
    plt.ylabel('Soma das Emissões')
    plt.tight_layout()
    plt.savefig('graficos/contribuicao_total_por_gas.png')
    
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
    
    # Análise 6: Correlação entre diferentes variáveis para o gás alvo
    colunas_numericas = df_alvo.select_dtypes(include=['number']).columns
    
    if len(colunas_numericas) > 1:
        plt.figure(figsize=(12, 10))
        matriz_corr = df_alvo[colunas_numericas].corr()
        sns.heatmap(matriz_corr, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title(f'Matriz de Correlação para {target_gas}')
        plt.tight_layout()
        plt.savefig(f'graficos/matriz_correlacao_{target_gas}.png')
    
    print(f"Análise concluída! Gráficos salvos na pasta 'graficos/'")
    
    # Retornar resumo das análises
    resumo = {
        'total_gases': len(df_analise['gas'].unique()),
        'total_emissoes_alvo': df_alvo[TARGET_COLUMN].sum(),
        'media_emissoes_alvo': df_alvo[TARGET_COLUMN].mean(),
        'erro_medio_alvo': abs(df_alvo[TARGET_COLUMN] - df_alvo['emissao_prevista']).mean(),
        'percentual_erro': abs(df_alvo[TARGET_COLUMN] - df_alvo['emissao_prevista']).sum() / df_alvo[TARGET_COLUMN].sum() * 100
    }
    
    print("\nRESUMO DA ANÁLISE:")
    for key, value in resumo.items():
        print(f"  - {key}: {value}")
    
    try:
        # Retornar resumo das análises
        resumo = {
            'total_gases': len(df_analise['gas'].unique()),
            'total_emissoes_alvo': df_alvo[TARGET_COLUMN].sum() if len(df_alvo) > 0 else 0,
            'media_emissoes_alvo': df_alvo[TARGET_COLUMN].mean() if len(df_alvo) > 0 else 0,
            'erro_medio_alvo': abs(df_alvo[TARGET_COLUMN] - df_alvo['emissao_prevista']).mean() if len(df_alvo) > 0 else 0,
            'percentual_erro': abs(df_alvo[TARGET_COLUMN] - df_alvo['emissao_prevista']).sum() / df_alvo[TARGET_COLUMN].sum() * 100 if len(df_alvo) > 0 and df_alvo[TARGET_COLUMN].sum() != 0 else 0
        }
        
        print("\nRESUMO DA ANÁLISE:")
        for key, value in resumo.items():
            print(f"  - {key}: {value}")
        
        return resumo
    except Exception as e:
        print(f"Erro ao gerar resumo da análise: {e}")
        # Retorna um resumo vazio em caso de erro
        return {'erro': str(e)} 