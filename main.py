"""
Arquivo principal de execução do projeto.

Este arquivo contém as funções principais para execução das análises,
servindo como ponto de entrada do programa.
"""

import os
import pandas as pd
from typing import Tuple, Dict, Optional, Any

from config import DATA_FILENAME, TARGET_COLUMN, TARGET_GAS, OUTPUT_DIRS
from utils.io import load_csv_data, initialize_directories
from core.data import filtrar_por_gas, realizar_analise_exploratoria 
from core.preprocessing import preprocessar_dados_completo
from core.model import (
    criar_pipeline_xgboost, 
    treinar_e_avaliar_modelo, 
    criar_baseline,
    otimizar_hiperparametros,
    prever_e_adicionar_resultados
)
from analyses.multi_gas import (
    identificar_gases_no_dataset,
    treinar_modelos_para_todos_gases,
    prever_para_todos_gases,
    analisar_gases
)
from utils.visualization import (
    plot_model_comparison,
    plot_predictions_vs_actual
)
from utils.io import save_model, save_dataframe


def executar_analise_basica() -> Tuple[Any, pd.DataFrame]:
    """
    Executa o pipeline básico de análise e previsão para N2O.
    
    Returns:
        Tupla com (modelo, resultados)
    """
    print("Iniciando análise de previsão de emissões de Óxido Nitroso (N2O)")
    
    # Carrega os dados
    df = load_csv_data(DATA_FILENAME)
    
    if df is None or df.empty:
        print("ERRO: Falha ao carregar dados. Verifique o arquivo CSV.")
        return None, None
    
    # Análise exploratória e filtragem para N2O
    df_n2o = realizar_analise_exploratoria(df, TARGET_COLUMN, TARGET_GAS)
    
    if df_n2o is None or df_n2o.empty:
        print("ERRO: Não foi possível filtrar dados para N2O.")
        return None, None
    
    # Pré-processamento dos dados
    resultado = preprocessar_dados_completo(df_n2o, TARGET_COLUMN)
    
    if resultado[0] is None:
        print("Não foi possível continuar devido à falta de dados.")
        return None, None
        
    X_train, X_test, y_train, y_test, colunas_numericas, colunas_categoricas = resultado
    
    # Cria e treina o modelo
    pipeline = criar_pipeline_xgboost(colunas_numericas, colunas_categoricas)
    resultado_treino = treinar_e_avaliar_modelo(pipeline, X_train, X_test, y_train, y_test)
    
    if resultado_treino[0] is None:
        print("ERRO: Falha ao treinar o modelo. Não é possível continuar.")
        return None, None
        
    modelo, y_pred, mse, rmse, r2 = resultado_treino
    
    # Visualizar previsões vs reais
    try:
        plot_predictions_vs_actual(y_test, y_pred)
    except Exception as e:
        print(f"Erro ao criar gráfico de previsões: {e}")
    
    # Salvar modelo
    try:
        save_model(modelo)
        print("Modelo salvo com sucesso.")
    except Exception as e:
        print(f"Erro ao salvar o modelo: {e}")
    
    # Previsão em novos dados
    print("\n=== Iniciando etapa de previsão em novos dados ===")
    try:
        resultados = prever_e_adicionar_resultados(df, modelo, TARGET_GAS)
        if resultados is None:
            print("ERRO: Não foi possível realizar previsões.")
        else:
            # Salvar resultados
            save_dataframe(resultados, 'resultado_predicao_n2o.csv', 'resultados')
            
            # Também salva um arquivo apenas com as linhas de N2O para análise
            df_n2o_resultado = resultados[resultados['gas'] == TARGET_GAS].copy()
            save_dataframe(df_n2o_resultado, 'apenas_n2o_predicao.csv', 'resultados')
            
            print("Previsões realizadas e salvas com sucesso.")
    except Exception as e:
        print(f"ERRO ao fazer previsões: {str(e)}")
        import traceback
        traceback.print_exc()
        resultados = None
    
    print("\nProcesso básico de análise e previsão finalizado.")
    
    return modelo, resultados


def executar_analise_com_otimizacao() -> Tuple[Any, pd.DataFrame]:
    """
    Executa o pipeline de análise com otimização de hiperparâmetros.
    
    Returns:
        Tupla com (modelo_otimizado, resultados)
    """
    print("Iniciando análise com otimização de hiperparâmetros")
    
    # Carrega os dados
    df = load_csv_data(DATA_FILENAME)
    
    # Análise exploratória e filtragem para N2O
    df_n2o = realizar_analise_exploratoria(df, TARGET_COLUMN, TARGET_GAS)
    
    # Pré-processamento dos dados
    X_train, X_test, y_train, y_test, colunas_numericas, colunas_categoricas = preprocessar_dados_completo(df_n2o, TARGET_COLUMN)
    
    # Cria pipeline
    pipeline = criar_pipeline_xgboost(colunas_numericas, colunas_categoricas)
    
    # Otimiza hiperparâmetros
    modelo_otimizado = otimizar_hiperparametros(pipeline, X_train, y_train)
    
    # Avalia modelo otimizado
    resultado_treino = treinar_e_avaliar_modelo(modelo_otimizado, X_train, X_test, y_train, y_test)
    modelo, y_pred, mse, rmse, r2 = resultado_treino
    
    # Visualizações
    plot_predictions_vs_actual(y_test, y_pred)
    
    # Realizar previsões
    resultados = prever_e_adicionar_resultados(df, modelo_otimizado, TARGET_GAS)
    
    # Salvar modelo
    save_model(modelo_otimizado, gas_name='otimizado')
    
    print("\nProcesso com otimização finalizado.")
    
    return modelo_otimizado, resultados


def executar_analise_com_comparacao() -> Tuple[Any, pd.DataFrame, float]:
    """
    Executa o pipeline de análise com comparação com baseline.
    
    Returns:
        Tupla com (modelo, resultados, melhoria)
    """
    print("Iniciando análise com comparação com baseline")
    
    # Carrega os dados
    df = load_csv_data(DATA_FILENAME)
    
    # Análise exploratória e filtragem para N2O
    df_n2o = realizar_analise_exploratoria(df, TARGET_COLUMN, TARGET_GAS)
    
    # Pré-processamento dos dados
    X_train, X_test, y_train, y_test, colunas_numericas, colunas_categoricas = preprocessar_dados_completo(df_n2o, TARGET_COLUMN)
    
    # Cria e avalia baseline
    mse_baseline, _ = criar_baseline(X_train, X_test, y_train, y_test)
    
    # Cria e treina modelo
    pipeline = criar_pipeline_xgboost(colunas_numericas, colunas_categoricas)
    resultado_treino = treinar_e_avaliar_modelo(pipeline, X_train, X_test, y_train, y_test)
    modelo, y_pred, mse, rmse, r2 = resultado_treino
    
    # Compara com baseline
    melhoria = plot_model_comparison(mse, mse_baseline)
    
    # Visualizações
    plot_predictions_vs_actual(y_test, y_pred)
    
    # Realizar previsões
    resultados = prever_e_adicionar_resultados(df, modelo, TARGET_GAS)
    
    # Salvar modelo
    save_model(modelo)
    
    print(f"\nMelhoria em relação ao baseline: {melhoria:.2f}%")
    print("\nProcesso com comparação finalizado.")
    
    return modelo, resultados, melhoria


def executar_analise_completa() -> Tuple[Any, pd.DataFrame, float]:
    """
    Executa o pipeline completo de análise (otimização + comparação).
    
    Returns:
        Tupla com (modelo_otimizado, resultados, melhoria)
    """
    print("\n" + "="*70)
    print("INICIANDO ANÁLISE COMPLETA (OTIMIZAÇÃO + COMPARAÇÃO)")
    print("="*70)
    
    # Carrega os dados
    df = load_csv_data(DATA_FILENAME)
    
    # Análise exploratória e filtragem para N2O
    df_n2o = realizar_analise_exploratoria(df, TARGET_COLUMN, TARGET_GAS)
    
    # Pré-processamento dos dados
    X_train, X_test, y_train, y_test, colunas_numericas, colunas_categoricas = preprocessar_dados_completo(df_n2o, TARGET_COLUMN)
    
    # Cria e avalia baseline
    mse_baseline, _ = criar_baseline(X_train, X_test, y_train, y_test)
    
    # Cria pipeline
    pipeline = criar_pipeline_xgboost(colunas_numericas, colunas_categoricas)
    
    # Otimiza hiperparâmetros
    modelo_otimizado = otimizar_hiperparametros(pipeline, X_train, y_train)
    
    # Avalia modelo otimizado
    resultado_treino = treinar_e_avaliar_modelo(modelo_otimizado, X_train, X_test, y_train, y_test)
    modelo, y_pred, mse, rmse, r2 = resultado_treino
    
    # Compara com baseline
    melhoria = plot_model_comparison(mse, mse_baseline)
    
    # Visualizações
    plot_predictions_vs_actual(y_test, y_pred)
    
    # Realizar previsões
    resultados = prever_e_adicionar_resultados(df, modelo_otimizado, TARGET_GAS)
    
    # Salvar modelo
    save_model(modelo_otimizado, gas_name='otimizado_completo')
    
    print("\n" + "="*70)
    print(f"RESULTADO FINAL DA ANÁLISE COMPLETA:")
    print("="*70)
    print(f"MSE final do modelo: {mse:.4f}")
    print(f"RMSE final do modelo: {rmse:.4f}")
    print(f"R² final do modelo: {r2:.4f}")
    print(f"Melhoria em relação ao baseline: {melhoria:.2f}%")
    print("="*70)
    print("\nProcesso completo finalizado.")
    
    return modelo_otimizado, resultados, melhoria


def executar_analise_todos_gases() -> Tuple:
    """
    Executa análise e previsão para todos os gases no dataset.
    
    Returns:
        Tupla com (df_resultado, modelos, metricas, analise)
    """
    print("\n" + "="*70)
    print("INICIANDO ANÁLISE PARA TODOS OS GASES")
    print("="*70)
    
    # Inicializa valores default para retorno
    df_resultado = None
    modelos = {}
    metricas = {}
    analise = {}
    
    try:
        # Carrega os dados
        df = load_csv_data(DATA_FILENAME)
        
        if df is None or df.empty:
            print("ERRO: Falha ao carregar dados. Verifique o arquivo CSV.")
            return df_resultado, modelos, metricas, analise
        
        # Identificar gases no dataset
        gases_mapeados = identificar_gases_no_dataset(df)
        
        # Treinar modelos para todos os gases
        modelos, metricas = treinar_modelos_para_todos_gases(df, gases_mapeados)
        
        # Realizar previsões para todos os gases
        df_resultado = prever_para_todos_gases(df, modelos, gases_mapeados)
        
        # Realizar análise comparativa com foco no N2O
        analise = analisar_gases(df_resultado, target_gas='N2O')
        
        # Exibir resumo final
        print("\n" + "="*70)
        print("RESUMO FINAL DA ANÁLISE MULTI-GÁS")
        print("="*70)
        print(f"Total de gases analisados: {len(modelos)}")
        
        # Tabela de métricas
        print("\nMétricas por gás:")
        print("-" * 60)
        print(f"{'Gás':<10} | {'MSE':<12} | {'RMSE':<12} | {'R²':<12} | {'Amostras':<10}")
        print("-" * 60)
        
        for gas, metrica in metricas.items():
            print(f"{gas:<10} | {metrica['mse']:<12.4f} | {metrica['rmse']:<12.4f} | {metrica['r2']:<12.4f} | {metrica['samples']:<10}")
        
        print("-" * 60)
        print("\nArquivos gerados:")
        print(f"- Modelos: {len(modelos)} arquivos .pkl na pasta 'modelos/'")
        print(f"- Resultados: arquivo CSV com previsões para todos os gases")
        print(f"- Gráficos: Diversos gráficos de análise na pasta 'graficos/'")
        
    except Exception as e:
        import traceback
        print(f"Erro durante a execução da análise: {e}")
        traceback.print_exc()
    
    print("\n" + "="*70)
    print("PROCESSO COMPLETO DE ANÁLISE MULTI-GÁS FINALIZADO")
    print("="*70)
    
    return df_resultado, modelos, metricas, analise


def testar_salvamento_csv():
    """
    Função para testar apenas o salvamento de um CSV.
    """
    import pandas as pd
    
    # Criar um DataFrame simples
    df_teste = pd.DataFrame({
        'coluna1': [1, 2, 3],
        'coluna2': ['a', 'b', 'c']
    })
    
    # Tenta salvar
    caminho = save_dataframe(df_teste, 'teste_salvamento.csv')
    
    if caminho:
        print(f"Arquivo de teste salvo com sucesso em: {caminho}")
    else:
        print("Falha ao salvar arquivo de teste.")


if __name__ == "__main__":
    # Inicializar diretórios
    initialize_directories(OUTPUT_DIRS)
    
    # Para executar a análise completa com todos os gases:
    resultado, modelos, metricas, analise = executar_analise_todos_gases()
    
    # Para executar apenas a análise básica:
    # modelo, resultados = executar_analise_basica()
    
    # Para executar a análise com otimização:
    # modelo, resultados = executar_analise_com_otimizacao()
    
    # Para executar a análise com comparação:
    # modelo, resultados, melhoria = executar_analise_com_comparacao()
    
    # Para executar a análise completa:
    # modelo, resultados, melhoria = executar_analise_completa()