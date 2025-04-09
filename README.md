# Projeto de Previsão de Emissões de Gases

Este projeto implementa uma solução de machine learning para previsão de emissões de gases de efeito estufa, com foco principal no Óxido Nitroso (N2O). O sistema permite análises individuais ou comparativas entre diferentes gases, utilizando modelos de regressão baseados em XGBoost.

## Funcionalidades

- Análise exploratória de dados de emissões
- Pré-processamento automático de dados
- Treinamento de modelos XGBoost para previsão de emissões
- Otimização de hiperparâmetros para melhorar performance
- Análise comparativa com modelo baseline
- Suporte para múltiplos gases (N2O, CO2, CH4)
- Visualizações e análises dos resultados
- Exportação de resultados em CSV

## Estrutura do Projeto

```
projeto/
├── config.py                # Configurações do projeto
├── main.py                  # Ponto de entrada da aplicação
├── core/                    # Módulos principais
│   ├── __init__.py
│   ├── data.py              # Carregamento e manipulação de dados
│   ├── preprocessing.py     # Pré-processamento de dados
│   ├── model.py             # Definição e treino de modelos
├── analyses/                # Análises específicas
│   ├── __init__.py
│   └── multi_gas.py         # Análise multi-gás
└── utils/                   # Utilitários
    ├── __init__.py
    ├── io.py                # Operações de entrada/saída
    └── visualization.py     # Funções de visualização
```

## Requisitos

- Python 3.7+
- pandas
- numpy
- scikit-learn
- xgboost
- matplotlib
- seaborn
- joblib

### Dependências Detalhadas

| Biblioteca    | Versão    | Descrição                                            |
|---------------|-----------|------------------------------------------------------|
| pandas        | >=1.3.0   | Manipulação e análise de dados                       |
| numpy         | >=1.20.0  | Computação numérica                                  |
| scikit-learn  | >=1.0.0   | Algoritmos de machine learning                       |
| xgboost       | >=1.5.0   | Implementação do algoritmo XGBoost                   |
| matplotlib    | >=3.4.0   | Visualização de dados                                |
| seaborn       | >=0.11.0  | Visualização estatística baseada no matplotlib       |
| joblib        | >=1.0.0   | Serialização de modelos                              |

## Instalação

1. Clone o repositório:
   ```
   git clone https://github.com/seu-usuario/projeto-previsao-emissoes.git
   cd projeto-previsao-emissoes
   ```

2. Crie um ambiente virtual (opcional, mas recomendado):
   ```
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```

3. Instale as dependências individualmente:
   ```
   pip install pandas>=1.3.0 numpy>=1.20.0 scikit-learn>=1.0.0 xgboost>=1.5.0 matplotlib>=3.4.0 seaborn>=0.11.0 joblib>=1.0.0
   ```

   Ou crie um arquivo `requirements.txt` com o seguinte conteúdo:
   ```
   pandas>=1.3.0
   numpy>=1.20.0
   scikit-learn>=1.0.0
   xgboost>=1.5.0
   matplotlib>=3.4.0
   seaborn>=0.11.0
   joblib>=1.0.0
   ```
   
   E então instale usando:
   ```
   pip install -r requirements.txt
   ```

## Uso

### Dataset

O projeto espera um arquivo CSV com dados de emissões no formato especificado no arquivo `config.py`. O arquivo padrão esperado é `br_seeg_emissoes_brasil.csv` e deve conter pelo menos:

- Uma coluna `gas` indicando o tipo de gás
- Uma coluna `emissao` com os valores de emissão

### Executando Análises

Para executar o projeto, você pode modificar o arquivo `main.py` descomentando a linha correspondente ao tipo de análise que deseja executar:

```python
# Para executar apenas a análise básica:
modelo, resultados = executar_analise_basica()

# Para executar a análise com otimização:
modelo, resultados = executar_analise_com_otimizacao()

# Para executar a análise com comparação:
modelo, resultados, melhoria = executar_analise_com_comparacao()

# Para executar a análise completa:
modelo, resultados, melhoria = executar_analise_completa()

# Para executar a análise completa com todos os gases:
resultado, modelos, metricas, analise = executar_analise_todos_gases()
```

Depois, execute:

```
python main.py
```

### Principais Funções

- `executar_analise_basica()`: Executa análise e previsão básica para N2O
- `executar_analise_com_otimizacao()`: Inclui otimização de hiperparâmetros
- `executar_analise_com_comparacao()`: Compara o modelo com um baseline
- `executar_analise_completa()`: Combina otimização e comparação
- `executar_analise_todos_gases()`: Analisa todos os gases presentes no dataset

## Saídas

O projeto gera automaticamente:

1. **Modelos treinados** na pasta `modelos/`
2. **Gráficos de análise** na pasta `graficos/`
3. **Resultados das previsões** na pasta `resultados/`

## Configurações

As principais configurações podem ser ajustadas no arquivo `config.py`, incluindo:

- `TARGET_GAS`: Gás principal para análise
- `TEST_SIZE`: Proporção de dados para teste
- `RANDOM_STATE`: Semente para reprodutibilidade
- `XGBOOST_PARAMS`: Parâmetros do modelo XGBoost
- `PARAM_GRID`: Grid de parâmetros para otimização

## Personalização

Para analisar gases diferentes ou adicionar novos formatos, modifique o dicionário `POSSIBLE_GAS_FORMATS` no arquivo `config.py`:

```python
POSSIBLE_GAS_FORMATS = {
    'N2O': ['N2O (t)', 'N2O(t)', 'N2O (i)', 'N2O(i)', 'Óxido Nitroso (N2O)', ...],
    'CH4': ['CH4 (t)', 'CH4(t)', 'CH4 (i)', 'CH4(i)', 'Metano (CH4)', ...],
    'CO2': ['CO2 (t)', 'CO2(t)', 'CO2 (i)', 'CO2(i)', 'Dióxido de Carbono (CO2)', ...]
}
```

## Autores

- Viinicius Ribas Quadros
- Lucas Antonio Ribeiro 
- Giovanni Antonietto Rosa
- Gustavo Fortunato
