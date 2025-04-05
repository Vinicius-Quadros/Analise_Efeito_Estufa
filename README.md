# Análise e Previsão de Emissões de Gases - Brasil

## Descrição do Projeto

Este projeto utiliza machine learning para analisar e prever emissões de gases de efeito estufa no Brasil, com foco principal no Óxido Nitroso (N₂O), mas também incluindo capacidade para análise de múltiplos gases como Metano (CH₄) e Dióxido de Carbono (CO₂).

O sistema implementa um pipeline completo de análise de dados que inclui carregamento, pré-processamento, modelagem com XGBoost, otimização de hiperparâmetros, avaliação de desempenho e visualização de resultados.

## Estrutura do Projeto

O projeto está organizado nos seguintes arquivos:

- `main.py`: Script principal que coordena o fluxo de execução
- `config.py`: Configurações e parâmetros do projeto
- `utils.py`: Funções utilitárias (carregamento de dados, salvamento de modelos)
- `preprocessing.py`: Funções para análise exploratória e pré-processamento
- `model.py`: Definição, treinamento e avaliação do modelo XGBoost
- `visualization.py`: Funções para visualização dos resultados
- `multi_gas_analysis.py`: Funções específicas para análise de múltiplos gases

## Dependências

O projeto requer as seguintes bibliotecas Python:

- pandas
- numpy
- scikit-learn
- xgboost
- matplotlib
- seaborn
- joblib

Instale as dependências com:

```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn joblib
```

## Dataset

O projeto utiliza o arquivo `br_seeg_emissoes_brasil.csv` contendo dados históricos de emissões de gases no Brasil. O dataset deve incluir, pelo menos, as seguintes colunas:

- `gas`: Tipo de gás (N₂O, CH₄, CO₂, etc.)
- `emissao`: Valor da emissão (variável alvo)

O sistema é flexível para reconhecer diferentes formatos de nomes para os gases (por exemplo, "N2O (t)", "Óxido Nitroso", etc.).

## Funcionalidades Principais

1. **Análise Básica**
   - Carregamento e exploração de dados
   - Treinamento do modelo para N₂O
   - Avaliação e salvamento do modelo

2. **Análise com Otimização**
   - Otimização de hiperparâmetros via GridSearchCV
   - Avaliação do modelo otimizado

3. **Análise com Comparação**
   - Comparação do modelo com baseline simples (média)
   - Cálculo de melhoria percentual

4. **Análise Completa**
   - Combina otimização e comparação com baseline
   - Pipeline completo de análise e previsão

5. **Análise Multi-Gás**
   - Identificação automática de todos os gases no dataset
   - Treinamento de modelos específicos para cada gás
   - Análise comparativa entre diferentes gases
   - Visualização de tendências e correlações

## Como Usar

### 1. Execução Básica

```python
from main import executar_analise_basica

modelo, resultados = executar_analise_basica()
```

### 2. Análise Multi-Gás

```python
from main import executar_analise_todos_gases

resultado, modelos, metricas, analise = executar_analise_todos_gases()
```

### 3. Execução via Script

```bash
python main.py
```

## Saídas do Modelo

O sistema gera diversos arquivos de saída:

- **Modelos treinados**: Arquivos `.pkl` contendo os modelos para cada gás
- **Previsões**: Arquivos CSV com os dados originais e as previsões do modelo
- **Visualizações**: Gráficos de importância de features, comparações com baseline, etc.

## Personalização

O comportamento do modelo pode ser ajustado modificando os parâmetros em `config.py`:

- `RANDOM_STATE`: Semente para reprodutibilidade
- `TEST_SIZE`: Proporção do conjunto de teste
- `XGBOOST_PARAMS`: Parâmetros do modelo XGBoost
- `PARAM_GRID`: Grid de hiperparâmetros para otimização

## Análise de Resultados

Após a execução, o sistema gera uma pasta `graficos/` contendo visualizações como:

- Distribuição de emissões por tipo de gás
- Comparação entre valores reais e previstos
- Erro médio de previsão por setor
- Contribuição total de emissões por gás
- Evolução temporal das emissões
- Matriz de correlação entre variáveis

O resumo da análise também é exibido no console, fornecendo métricas como erro médio, percentual de erro e contribuição total.

## Limitações e Desenvolvimento Futuro

- O projeto está otimizado para o formato específico do dataset SEEG Brasil
- Para melhorar a performance, considere:
  - Ajustar o pré-processamento para dados específicos
  - Expandir o grid de hiperparâmetros
  - Implementar validação cruzada mais robusta
  - Experimentar modelos de séries temporais para prever tendências futuras

## Autores

- Vinicius Ribas Quadros
- Lucas Antonio RIbeiro 
- Gustavo Fortunato
- Giovanni Antonietto Rosa 