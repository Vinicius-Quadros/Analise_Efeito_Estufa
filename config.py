"""
Arquivo de configurações para o projeto de previsão de emissões de gases.

Este arquivo contém todas as constantes e configurações utilizadas no projeto,
facilitando ajustes sem necessidade de alterar o código-fonte.
"""

# Configurações gerais
RANDOM_STATE = 42
TEST_SIZE = 0.2
DATA_FILENAME = 'br_seeg_emissoes_brasil.csv'
TARGET_COLUMN = 'emissao'

# Diretórios de saída
OUTPUT_DIRS = {
    'MODELS': 'modelos',
    'GRAPHS': 'graficos',
    'RESULTS': 'resultados',
}

# Templates de nome de arquivo
MODEL_FILENAME_TEMPLATE = 'modelo_xgboost_{}.pkl'
GRAPH_FILENAME_TEMPLATE = '{}_graph.png'
RESULT_FILENAME_TEMPLATE = 'resultado_{}.csv'

# Configurações de gases
TARGET_GAS = 'N2O (t)'  # Formato padrão para N2O no dataset

# Mapeamento de nomenclaturas de gases
POSSIBLE_GAS_FORMATS = {
    'N2O': ['N2O (t)', 'N2O(t)', 'N2O (i)', 'N2O(i)', 'Óxido Nitroso (N2O)', 'Oxido Nitroso (N2O)', 'N2O', 'Óxido Nitroso', 'Oxido Nitroso'],
    'CH4': ['CH4 (t)', 'CH4(t)', 'CH4 (i)', 'CH4(i)', 'Metano (CH4)', 'CH4'],
    'CO2': ['CO2 (t)', 'CO2(t)', 'CO2 (i)', 'CO2(i)', 'Dióxido de Carbono (CO2)', 'Dioxido de Carbono (CO2)', 'CO2']
}

# Parâmetros padrão para o XGBoost
XGBOOST_PARAMS = {
    'objective': 'reg:squarederror',
    'n_estimators': 100,
    'learning_rate': 0.1,
    'max_depth': 5,
    'random_state': RANDOM_STATE,
    'n_jobs': -1
}

# Grid de parâmetros para otimização
PARAM_GRID = {
    'model__n_estimators': [100, 200],
    'model__learning_rate': [0.05, 0.1],
    'model__max_depth': [4, 6]
}