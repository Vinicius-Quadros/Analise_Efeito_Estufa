# Arquivo de configurações
RANDOM_STATE = 42
TEST_SIZE = 0.2
MODEL_FILENAME_TEMPLATE = 'modelo_xgboost_{}.pkl'
DATA_FILENAME = 'br_seeg_emissoes_brasil.csv'
TARGET_GAS = 'N2O (t)'  # Atualizado para o formato correto do seu arquivo
TARGET_COLUMN = 'emissao'

# Lista de possíveis nomes alternativos para N2O
POSSIBLE_N2O_NAMES = [
    'N2O (t)',
    'N2O(t)',
    'N2O (i)',
    'N2O(i)',
    'Óxido Nitroso (N2O)',
    'Oxido Nitroso (N2O)',
    'N2O',
    'Óxido Nitroso',
    'Oxido Nitroso'
]

# Lista de possíveis nomes de gases no dataset
POSSIBLE_GAS_FORMATS = {
    'N2O': ['N2O (t)', 'N2O(t)', 'N2O (i)', 'N2O(i)', 'Óxido Nitroso (N2O)', 'Oxido Nitroso (N2O)', 'N2O'],
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


# 'br_seeg_emissoes_brasil.csv' 