# Funções para criação, treinamento e avaliação do modelo
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from preprocessing import criar_preprocessador
from config import XGBOOST_PARAMS, PARAM_GRID, TARGET_COLUMN

def criar_pipeline(colunas_numericas, colunas_categoricas):
    """
    Cria um pipeline completo com pré-processamento e modelo XGBoost.
    """
    # Obtendo preprocessador
    preprocessor = criar_preprocessador(colunas_numericas, colunas_categoricas)
    
    # Criando pipeline completo com XGBoost
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', xgb.XGBRegressor(**XGBOOST_PARAMS))
    ])
    
    return pipeline

def treinar_e_avaliar(pipeline, X_train, X_test, y_train, y_test):
    """
    Treina o modelo e avalia o desempenho.
    """
    # Treinamento
    print("Treinando o modelo XGBoost...")
    
    # Armazena os nomes das colunas
    print(f"Colunas sendo usadas para treino: {X_train.columns.tolist()}")
    
    try:
        pipeline.fit(X_train, y_train)
        
        # Avaliação no conjunto de teste
        y_pred = pipeline.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        
        print(f"Erro Quadrático Médio (MSE): {mse:.4f}")
        print(f"Raiz do Erro Quadrático Médio (RMSE): {rmse:.4f}")
        
        # Visualização das previsões vs valores reais
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.xlabel('Valores Reais')
        plt.ylabel('Previsões')
        plt.title('Previsões vs Valores Reais')
        plt.savefig('previsoes_vs_reais.png')
        
        return pipeline, y_pred, mse, rmse
    except Exception as e:
        print(f"Erro durante o treinamento do modelo: {e}")
        
        # Debug adicional
        print(f"Shapes: X_train: {X_train.shape}, y_train: {y_train.shape}")
        print(f"Tipos: X_train dtype: {X_train.dtypes.iloc[0]}, y_train dtype: {y_train.dtype}")
        print(f"Amostra de y_train: {y_train.head()}")
        
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
        
        # Retornar valores default
        return None, None, float('inf'), float('inf')

def otimizar_hiperparametros(pipeline, X_train, y_train):
    """
    Realiza otimização de hiperparâmetros do XGBoost via GridSearchCV.
    """
    print("\nIniciando otimização de hiperparâmetros...")
    
    # Configurando GridSearchCV
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=PARAM_GRID,
        scoring='neg_mean_squared_error',
        cv=3,  # 3-fold cross-validation
        n_jobs=-1,  # Usar todos os processadores disponíveis
        verbose=1
    )
    
    # Executando busca em grid
    grid_search.fit(X_train, y_train)
    
    # Exibindo melhores parâmetros
    print(f"\nMelhores parâmetros encontrados: {grid_search.best_params_}")
    print(f"Melhor score: {-grid_search.best_score_:.4f} (MSE)")
    
    # Retornando melhor modelo
    return grid_search.best_estimator_

def prever_em_novos_dados(pipeline, df_original):
    """
    Realiza previsões para todas as linhas de N2O no DataFrame original
    e retorna o arquivo original acrescido das previsões.
    """
    # Importa configurações
    from config import TARGET_GAS, TARGET_COLUMN
    import os
    
    print("\n===== INICIANDO PROCESSO DE PREVISÃO =====")
    print(f"Formato do DataFrame original: {df_original.shape}")
    
    # Cria uma cópia do DataFrame original para não alterar o original
    df_output = df_original.copy()
    
    # Filtra apenas os dados de N2O para fazer as previsões
    df_n2o = df_original[df_original['gas'] == TARGET_GAS].copy()
    
    # Verifica se há dados para prever
    if df_n2o.empty:
        print(f"ERRO: Não foram encontrados dados para o gás {TARGET_GAS}")
        return None
    
    print(f"Encontradas {len(df_n2o)} linhas com {TARGET_GAS}")
    
    # Features para previsão (sem a coluna target)
    X_pred = df_n2o.drop(TARGET_COLUMN, axis=1) if TARGET_COLUMN in df_n2o.columns else df_n2o
    
    # Verifica existência do pipeline
    if pipeline is None:
        print("ERRO: Pipeline de modelo não foi fornecido ou está inválido")
        return None
        
    # Verifica os tipos de dados
    print(f"\nVerificação de tipos de dados em X_pred:")
    print(X_pred.dtypes.head())
    
    # Realiza as previsões
    print(f"\nRealizando previsões para {len(X_pred)} linhas de {TARGET_GAS}...")
    try:
        y_pred = pipeline.predict(X_pred)
        print(f"Previsões realizadas com sucesso. Shape: {y_pred.shape}")
        print(f"Primeiras 5 previsões: {y_pred[:5]}")
        
        # Cria uma nova coluna no DataFrame original para armazenar as previsões
        # Inicializa a coluna de previsão com valores vazios para todas as linhas
        df_output['emissao_prevista'] = None
        
        # Atribui as previsões apenas às linhas correspondentes ao N2O
        indices_n2o = df_n2o.index
        df_output.loc[indices_n2o, 'emissao_prevista'] = y_pred
        
        # Formata a coluna de previsão para usar vírgula como separador decimal (para manter consistência)
        # Verifica se a coluna 'emissao' original usa vírgula como separador
        usa_virgula = False
        if df_original[TARGET_COLUMN].dtype == 'object':
            amostra = df_original[TARGET_COLUMN].iloc[0] if len(df_original) > 0 else ''
            if ',' in str(amostra):
                usa_virgula = True
        
        # Se o arquivo original usa vírgula como separador, mantém o padrão
        if usa_virgula:
            print("Formatando previsões com vírgula como separador decimal...")
            df_output['emissao_prevista'] = df_output['emissao_prevista'].apply(
                lambda x: str(x).replace('.', ',') if x is not None else None
            )
        
        # Caminho absoluto para salvar o arquivo
        caminho_resultado = os.path.abspath('resultado_predicao_n2o.csv')
        caminho_n2o = os.path.abspath('apenas_n2o_predicao.csv')
        
        # Salva o dataframe completo com as previsões
        print(f"\nSalvando arquivo completo em: {caminho_resultado}")
        df_output.to_csv(caminho_resultado, index=False)
        
        # Verifica se o arquivo foi salvo
        if os.path.exists(caminho_resultado):
            print(f"Arquivo salvo com sucesso. Tamanho: {os.path.getsize(caminho_resultado)} bytes")
        else:
            print("ERRO: O arquivo não foi salvo corretamente")
        
        # Também salva um arquivo apenas com as linhas de N2O para análise
        df_n2o_resultado = df_output[df_output['gas'] == TARGET_GAS].copy()
        df_n2o_resultado.to_csv(caminho_n2o, index=False)
        
        if os.path.exists(caminho_n2o):
            print(f"Arquivo N2O salvo com sucesso. Tamanho: {os.path.getsize(caminho_n2o)} bytes")
        else:
            print("ERRO: O arquivo N2O não foi salvo corretamente")
        
        print("\nPrevisões concluídas e arquivos salvos com sucesso!")
        
        return df_output
    
    except Exception as e:
        print(f"ERRO durante as previsões: {e}")
        import traceback
        traceback.print_exc()
        
        # Tenta salvar um arquivo de log com o erro
        try:
            with open('erro_previsao.log', 'w') as f:
                f.write(f"Erro ao realizar previsões: {str(e)}\n")
                f.write(f"Formato do DataFrame original: {df_original.shape}\n")
                f.write(f"Colunas do DataFrame: {df_original.columns.tolist()}\n")
                f.write(f"Número de linhas N2O: {len(df_n2o)}\n")
        except:
            print("Não foi possível salvar o arquivo de log de erros")
        
        return None