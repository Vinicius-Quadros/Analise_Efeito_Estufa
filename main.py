# Script principal
from config import DATA_FILENAME
from utils import carregar_dados, criar_baseline, salvar_modelo
from preprocessing import analise_exploratoria, preprocessamento
from model import criar_pipeline, treinar_e_avaliar, otimizar_hiperparametros, prever_em_novos_dados
from visualization import analisar_importancia_features, comparar_com_baseline

def executar_analise_basica():
    """
    Executa o pipeline básico de análise e previsão.
    """
    print("Iniciando análise de previsão de emissões de Óxido Nitroso (N2O)")
    
    # Carrega os dados
    df = carregar_dados(DATA_FILENAME)
    
    if df is None or df.empty:
        print("ERRO: Falha ao carregar dados. Verifique o arquivo CSV.")
        return None, None
    
    # Análise exploratória e filtragem para N2O
    df_n2o = analise_exploratoria(df)
    
    if df_n2o is None or df_n2o.empty:
        print("ERRO: Não foi possível filtrar dados para N2O.")
        return None, None
    
    # Pré-processamento dos dados
    X_train, X_test, y_train, y_test, colunas_numericas, colunas_categoricas = preprocessamento(df_n2o)
    
    # Verificar se temos dados para continuar
    if X_train is None:
        print("Não foi possível continuar devido à falta de dados.")
        return None, None
    
    # Cria e treina o modelo
    pipeline = criar_pipeline(colunas_numericas, colunas_categoricas)
    modelo, y_pred, mse, rmse = treinar_e_avaliar(pipeline, X_train, X_test, y_train, y_test)
    
    if modelo is None:
        print("ERRO: Falha ao treinar o modelo. Não é possível continuar.")
        return None, None
    
    # Análise de importância das features
    try:
        df_importancia = analisar_importancia_features(modelo, X_train)
    except Exception as e:
        print(f"Erro ao analisar importância das features: {e}")
    
    # Salvar modelo
    try:
        salvar_modelo(modelo)
        print("Modelo salvo com sucesso.")
    except Exception as e:
        print(f"Erro ao salvar o modelo: {e}")
    
    # Previsão em novos dados
    print("\n=== Iniciando etapa de previsão em novos dados ===")
    try:
        resultados = prever_em_novos_dados(modelo, df)
        if resultados is None:
            print("ERRO: Não foi possível realizar previsões.")
        else:
            print("Previsões realizadas e salvas com sucesso.")
    except Exception as e:
        print(f"ERRO ao fazer previsões: {str(e)}")
        import traceback
        traceback.print_exc()
        resultados = None
    
    print("\nProcesso básico de análise e previsão finalizado.")
    
    return modelo, resultados

def executar_analise_com_otimizacao():
    """
    Executa o pipeline de análise com otimização de hiperparâmetros.
    """
    print("Iniciando análise com otimização de hiperparâmetros")
    
    # Carrega os dados
    df = carregar_dados(DATA_FILENAME)
    
    # Análise exploratória e filtragem para N2O
    df_n2o = analise_exploratoria(df)
    
    # Pré-processamento dos dados
    X_train, X_test, y_train, y_test, colunas_numericas, colunas_categoricas = preprocessamento(df_n2o)
    
    # Cria pipeline
    pipeline = criar_pipeline(colunas_numericas, colunas_categoricas)
    
    # Otimiza hiperparâmetros
    modelo_otimizado = otimizar_hiperparametros(pipeline, X_train, y_train)
    
    # Avalia modelo otimizado
    _, y_pred, mse, rmse = treinar_e_avaliar(modelo_otimizado, X_train, X_test, y_train, y_test)
    
    # Análise de importância e previsões
    analisar_importancia_features(modelo_otimizado, X_train)
    resultados = prever_em_novos_dados(modelo_otimizado, df)
    
    # Salvar modelo
    salvar_modelo(modelo_otimizado)
    
    print("\nProcesso com otimização finalizado.")
    
    return modelo_otimizado, resultados

def executar_analise_com_comparacao():
    """
    Executa o pipeline de análise com comparação com baseline.
    """
    print("Iniciando análise com comparação com baseline")
    
    # Carrega os dados
    df = carregar_dados(DATA_FILENAME)
    
    # Análise exploratória e filtragem para N2O
    df_n2o = analise_exploratoria(df)
    
    # Pré-processamento dos dados
    X_train, X_test, y_train, y_test, colunas_numericas, colunas_categoricas = preprocessamento(df_n2o)
    
    # Cria e avalia baseline
    mse_baseline, _ = criar_baseline(X_train, X_test, y_train, y_test)
    
    # Cria e treina modelo
    pipeline = criar_pipeline(colunas_numericas, colunas_categoricas)
    modelo, y_pred, mse, rmse = treinar_e_avaliar(pipeline, X_train, X_test, y_train, y_test)
    
    # Compara com baseline
    melhoria = comparar_com_baseline(mse, mse_baseline)
    
    # Análise de importância e previsões
    analisar_importancia_features(modelo, X_train)
    resultados = prever_em_novos_dados(modelo, df)
    
    # Salvar modelo
    salvar_modelo(modelo)
    
    print(f"\nMelhoria em relação ao baseline: {melhoria:.2f}%")
    print("\nProcesso com comparação finalizado.")
    
    return modelo, resultados, melhoria

def executar_analise_completa():
    """
    Executa o pipeline completo de análise (otimização + comparação).
    """
    print("Iniciando análise completa")
    
    # Carrega os dados
    df = carregar_dados(DATA_FILENAME)
    
    # Análise exploratória e filtragem para N2O
    df_n2o = analise_exploratoria(df)
    
    # Pré-processamento dos dados
    X_train, X_test, y_train, y_test, colunas_numericas, colunas_categoricas = preprocessamento(df_n2o)
    
    # Cria e avalia baseline
    mse_baseline, _ = criar_baseline(X_train, X_test, y_train, y_test)
    
    # Cria pipeline
    pipeline = criar_pipeline(colunas_numericas, colunas_categoricas)
    
    # Otimiza hiperparâmetros
    modelo_otimizado = otimizar_hiperparametros(pipeline, X_train, y_train)
    
    # Avalia modelo otimizado
    _, y_pred, mse, rmse = treinar_e_avaliar(modelo_otimizado, X_train, X_test, y_train, y_test)
    
    # Compara com baseline
    melhoria = comparar_com_baseline(mse, mse_baseline)
    
    # Análise de importância e previsões
    analisar_importancia_features(modelo_otimizado, X_train)
    resultados = prever_em_novos_dados(modelo_otimizado, df)
    
    # Salvar modelo
    salvar_modelo(modelo_otimizado)
    
    print(f"\nMelhoria em relação ao baseline: {melhoria:.2f}%")
    print("\nProcesso completo finalizado.")
    
    return modelo_otimizado, resultados, melhoria

def testar_salvamento_csv():
    """
    Função para testar apenas o salvamento de um CSV.
    """
    import pandas as pd
    import os
    
    # Criar um DataFrame simples
    df_teste = pd.DataFrame({
        'coluna1': [1, 2, 3],
        'coluna2': ['a', 'b', 'c']
    })
    
    # Tenta salvar
    caminho = os.path.abspath('teste_salvamento.csv')
    print(f"Tentando salvar arquivo de teste em: {caminho}")
    
    try:
        df_teste.to_csv(caminho, index=False)
        if os.path.exists(caminho):
            print(f"Arquivo de teste salvo com sucesso! Tamanho: {os.path.getsize(caminho)} bytes")
        else:
            print("Falha ao salvar arquivo de teste.")
    except Exception as e:
        print(f"Erro ao salvar arquivo de teste: {e}")


# [Todo o conteúdo existente do main.py permanece aqui]

# ... [funções existentes] ...

def executar_analise_todos_gases():
    """
    Executa análise e previsão para todos os gases no dataset.
    """
    from multi_gas_analysis import identificar_gases_no_dataset, treinar_modelos_para_todos_gases, prever_para_todos_gases, analisar_gases
    from config import DATA_FILENAME, TARGET_GAS
    
    print("\n" + "="*50)
    print("INICIANDO ANÁLISE PARA TODOS OS GASES")
    print("="*50)
    
    # Inicializa valores default para retorno
    df_resultado = None
    modelos = {}
    metricas = {}
    analise = {}
    
    try:
        # Carrega os dados
        df = carregar_dados(DATA_FILENAME)
        
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
        
    except Exception as e:
        import traceback
        print(f"Erro durante a execução da análise: {e}")
        traceback.print_exc()
    
    print("\nProcesso completo de análise multi-gás finalizado.")
    
    return df_resultado, modelos, metricas, analise

# Modifique apenas esta parte no final do arquivo
if __name__ == "__main__":
    # Você pode descomentar a linha que preferir executar
    
    # Para executar apenas a análise básica (original):
    # modelo, resultados = executar_analise_basica()
    
    # Para executar a análise completa com todos os gases:
    resultado, modelos, metricas, analise = executar_analise_todos_gases()