# type: ignore
import pandas as pd
import numpy as np
from data_generator import generate_sample_data
from lstm_model import LSTMPredictor
from linear_regression_model import LinearRegressionPredictor
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class ModelComparison:
    def __init__(self, data_path='restaurant_data.csv'):
        self.data_path = data_path
        self.lstm_model = LSTMPredictor()
        self.lr_model = LinearRegressionPredictor()
        
    def load_data(self):
        try:
            df = pd.read_csv(self.data_path)
        except FileNotFoundError:
            print("Generating sample data...")
            df = generate_sample_data()
            df.to_csv(self.data_path, index=False)
        
        # Convert date column to datetime
        df['data'] = pd.to_datetime(df['data'])
        return df
    
    def prepare_data_for_models(self, df, target_column):
        # Split data into train and test (80/20)
        train_size = int(len(df) * 0.8)
        train_data = df[:train_size]
        test_data = df[train_size:]
        
        # Prepare data for LSTM
        X_lstm_train, y_lstm_train = self.lstm_model.prepare_data(train_data, target_column)
        X_lstm_test, y_lstm_test = self.lstm_model.prepare_data(test_data, target_column)
        
        # Prepare data for Linear Regression
        X_lr_train, y_lr_train = self.lr_model.prepare_data(train_data, target_column)
        X_lr_test, y_lr_test = self.lr_model.prepare_data(test_data, target_column)
        
        return {
            'lstm': (X_lstm_train, y_lstm_train, X_lstm_test, y_lstm_test),
            'lr': (X_lr_train, y_lr_train, X_lr_test, y_lr_test)
        }
    
    def train_and_evaluate(self, data_dict, target_column):
        results = {}
        
        # Train and evaluate LSTM
        print("\n🔄 Training LSTM model...")
        self.lstm_model.train(data_dict['lstm'][0], data_dict['lstm'][1])
        lstm_metrics = self.lstm_model.evaluate(data_dict['lstm'][2], data_dict['lstm'][3])
        results['lstm'] = lstm_metrics
        
        # Train and evaluate Linear Regression
        print("🔄 Training Linear Regression model...")
        self.lr_model.train(data_dict['lr'][0], data_dict['lr'][1])
        lr_metrics = self.lr_model.evaluate(data_dict['lr'][2], data_dict['lr'][3])
        results['lr'] = lr_metrics
        
        return results
    
    def plot_results(self, df, target_column, data_dict):
        # Set style
        plt.style.use('seaborn')
        sns.set_palette("husl")
        
        # Get predictions
        lstm_pred = self.lstm_model.predict(data_dict['lstm'][2])
        lr_pred = self.lr_model.predict(data_dict['lr'][2])
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), height_ratios=[2, 1])
        
        # Plot actual values and predictions
        dates = df['data'].values[-len(lstm_pred):]
        actual_values = df[target_column].values[-len(lstm_pred):]
        
        # Main plot
        ax1.plot(dates, actual_values, label='Actual', color='#2ecc71', linewidth=2)
        ax1.plot(dates, lstm_pred, label='LSTM Predictions', color='#3498db', linestyle='--', linewidth=2)
        ax1.plot(dates, lr_pred, label='Linear Regression Predictions', color='#e74c3c', linestyle='--', linewidth=2)
        
        ax1.set_title(f'Previsão de Demanda para {target_column.title()}', fontsize=14, pad=20)
        ax1.set_xlabel('Data')
        ax1.set_ylabel('Quantidade')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Error plot
        lstm_error = np.abs(actual_values - lstm_pred)
        lr_error = np.abs(actual_values - lr_pred)
        
        ax2.plot(dates, lstm_error, label='Erro LSTM', color='#3498db', alpha=0.7)
        ax2.plot(dates, lr_error, label='Erro Regressão Linear', color='#e74c3c', alpha=0.7)
        
        ax2.set_title('Erro Absoluto das Previsões', fontsize=12)
        ax2.set_xlabel('Data')
        ax2.set_ylabel('Erro Absoluto')
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f'predictions_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def print_metric_explanation(self, metric_name):
        explanations = {
            'mse': """
    MSE (Erro Quadrático Médio):
    • Média dos erros ao quadrado entre previsões e valores reais
    • Penaliza mais fortemente erros grandes
    • Quanto menor, melhor
    • Unidade: ao quadrado da unidade original
            """,
            'mae': """
    MAE (Erro Absoluto Médio):
    • Média dos erros absolutos entre previsões e valores reais
    • Mais fácil de interpretar que o MSE
    • Quanto menor, melhor
    • Unidade: mesma do dado original
            """,
            'r2': """
    R² (Coeficiente de Determinação):
    • Varia de 0 a 1 (0% a 100%)
    • Indica quanto da variabilidade dos dados o modelo explica
    • Quanto mais próximo de 1, melhor
    • Valores negativos indicam performance pior que uma linha horizontal
            """
        }
        return explanations.get(metric_name.lower(), "")
    
    def run_comparison(self, target_column='carne'):
        print("\n🚀 Iniciando Análise de Previsão de Demanda")
        print("=" * 50)
        
        # Load and prepare data
        df = self.load_data()
        data_dict = self.prepare_data_for_models(df, target_column)
        
        # Train and evaluate models
        results = self.train_and_evaluate(data_dict, target_column)
        
        # Print results with better formatting
        print("\n📊 Resultados da Comparação de Modelos")
        print("=" * 50)
        print(f"📌 Coluna Alvo: {target_column.title()}")
        
        print("\n🔍 Métricas do Modelo LSTM:")
        print("-" * 30)
        for metric, value in results['lstm'].items():
            print(f"  • {metric.upper()}: {value:.4f}")
            print(self.print_metric_explanation(metric))
        
        print("\n🔍 Métricas do Modelo de Regressão Linear:")
        print("-" * 30)
        for metric, value in results['lr'].items():
            print(f"  • {metric.upper()}: {value:.4f}")
            print(self.print_metric_explanation(metric))
        
        # Plot results
        self.plot_results(df, target_column, data_dict)
        print("\n✨ Gráficos salvos com sucesso!")

if __name__ == "__main__":
    comparison = ModelComparison()
    comparison.run_comparison() 