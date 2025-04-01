# Sistema de Previsão de Compras para Restaurantes

Este projeto implementa um sistema de previsão de compras para restaurantes utilizando dois modelos diferentes:
1. Rede Neural LSTM (Long Short-Term Memory)
2. Regressão Linear

O sistema foi desenvolvido para prever a demanda de ingredientes e insumos, ajudando a reduzir desperdícios e otimizar custos operacionais.

## Estrutura do Projeto

- `data_generator.py`: Gera dados de exemplo para testes
- `lstm_model.py`: Implementa o modelo LSTM
- `linear_regression_model.py`: Implementa o modelo de Regressão Linear
- `main.py`: Classe principal que compara os modelos
- `requirements.txt`: Lista de dependências do projeto
- `restaurant_data.csv`: Dados de exemplo (gerado automaticamente)

## Requisitos

- Python 3.8+
- TensorFlow
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn

## Instalação

1. Clone o repositório
2. Instale as dependências:
```bash
pip install -r requirements.txt
```

## Uso

Para executar o projeto:

```bash
python main.py
```

O script irá:
1. Gerar dados de exemplo (se não existirem)
2. Treinar os modelos LSTM e de Regressão Linear
3. Comparar as métricas de performance
4. Gerar um gráfico comparativo das previsões

## Métricas de Avaliação

Os modelos são avaliados usando:
- MSE (Mean Squared Error)
- MAE (Mean Absolute Error)
- R² Score

## Personalização

Você pode modificar os seguintes parâmetros:
- Tamanho da sequência LSTM (padrão: 7 dias)
- Proporção de treino/teste (padrão: 80/20)
- Número de épocas do treinamento LSTM (padrão: 50)
- Tamanho do batch (padrão: 32) 