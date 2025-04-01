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

![Imagem do WhatsApp de 2025-03-31 à(s) 23 43 12_e7f521c8](https://github.com/user-attachments/assets/4f6d8849-d5fa-40db-a876-1ca51b4d046a)
![Imagem do WhatsApp de 2025-03-31 à(s) 23 47 43_e19371f1](https://github.com/user-attachments/assets/71e209ee-39f6-4d0d-9798-b6d751f7bfdd)
![Imagem do WhatsApp de 2025-03-31 à(s) 23 47 54_ac2858f1](https://github.com/user-attachments/assets/6d913055-5bc8-421f-b4c7-d7140c3fecfc)


