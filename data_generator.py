import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_sample_data(n_days=365):
    # Generate dates
    dates = [datetime.now() - timedelta(days=x) for x in range(n_days)]
    dates.reverse()
    
    # Generate sample data with some seasonality and trends
    np.random.seed(42)
    
    # Base values for different ingredients
    base_values = {
        'carne': 100,
        'frango': 80,
        'arroz': 50,
        'feijao': 40,
        'tomate': 30,
        'cebola': 20,
        'alho': 10
    }
    
    # Create DataFrame
    data = {'data': dates}
    
    # Generate data for each ingredient
    for ingrediente, base in base_values.items():
        # Add seasonality (weekly and monthly patterns)
        weekly_pattern = np.sin(np.linspace(0, 2*np.pi*n_days/7, n_days))
        monthly_pattern = np.sin(np.linspace(0, 2*np.pi*n_days/30, n_days))
        
        # Add trend (slight increase over time)
        trend = np.linspace(0, 0.2, n_days)
        
        # Generate random noise
        noise = np.random.normal(0, 5, n_days)
        
        # Combine all components
        values = base * (1 + weekly_pattern*0.1 + monthly_pattern*0.05 + trend + noise/100)
        values = np.maximum(values, 0)  # Ensure no negative values
        
        data[ingrediente] = values
    
    df = pd.DataFrame(data)
    
    # Add some derived features
    df['dia_semana'] = df['data'].dt.dayofweek
    df['mes'] = df['data'].dt.month
    df['dia_mes'] = df['data'].dt.day
    df['fim_semana'] = df['dia_semana'].isin([5, 6]).astype(int)
    
    return df

if __name__ == "__main__":
    # Generate sample data
    df = generate_sample_data()
    
    # Save to CSV
    df.to_csv('restaurant_data.csv', index=False)
    print("Sample data generated and saved to 'restaurant_data.csv'") 