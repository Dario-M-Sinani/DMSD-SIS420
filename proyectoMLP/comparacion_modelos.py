import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import requests
import warnings
warnings.filterwarnings('ignore')

from ems_predict_mlp import get_sucre_historical_weather, AmbulanceDemandMLP

# ==========================================
# Funciones Base
# ==========================================
def load_and_filter_data(filepath):
    df = pd.read_csv(filepath)
    return df[df['title'].str.startswith('EMS')]

def extract_time_features(df):
    df['timeStamp'] = pd.to_datetime(df['timeStamp'])
    df['Hour'] = df['timeStamp'].dt.hour
    df['DayOfWeek'] = df['timeStamp'].dt.dayofweek
    df['Month'] = df['timeStamp'].dt.month
    return df

def create_geospatial_grid(df, grid_size=50):
    lat_bins = np.linspace(df['lat'].min(), df['lat'].max(), grid_size + 1)
    lng_bins = np.linspace(df['lng'].min(), df['lng'].max(), grid_size + 1)
    df['lat_idx'] = np.digitize(df['lat'], lat_bins) - 1
    df['lng_idx'] = np.digitize(df['lng'], lng_bins) - 1
    df['Zone_ID'] = df['lat_idx'] * grid_size + df['lng_idx']
    df = df.drop(['lat_idx', 'lng_idx'], axis=1)
    return df

def add_synthetic_features_local(demand_df, real_weather_df):
    np.random.seed(42)
    demand_df['Holiday_Type'] = np.random.choice([0, 1, 2, 3, 4, 5], size=len(demand_df), p=[0.89, 0.04, 0.02, 0.02, 0.02, 0.01])
    if real_weather_df is not None:
        demand_df = pd.merge(demand_df, real_weather_df, on=['Month', 'DayOfWeek', 'Hour'], how='left')
        demand_df['Weather_Severity'] = demand_df['Weather_Severity'].fillna(0)
    else:
        demand_df['Weather_Severity'] = 0.1
    
    holiday_impact = demand_df['Holiday_Type'].map({0: 0, 1: 5, 2: 10, 3: 15, 4: 25, 5: 20})
    demand_df['Demand'] = demand_df['Demand'] + holiday_impact + (demand_df['Weather_Severity'] * 10)
    return demand_df

class EMSDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

def train_and_eval(demand_df, model_name):
    # Features y Target
    X = demand_df[['Month', 'DayOfWeek', 'Hour', 'Holiday_Type', 'Weather_Severity']].values
    y = demand_df['Demand'].values
    
    # Normalización
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0) + 1e-8
    X_scaled = (X - mean) / std
    
    # DataLoaders
    indices = np.random.permutation(len(X_scaled))
    split = int(0.8 * len(X_scaled))
    train_idx, test_idx = indices[:split], indices[split:]
    
    train_loader = DataLoader(EMSDataset(X_scaled[train_idx], y[train_idx]), batch_size=128, shuffle=True)
    test_loader = DataLoader(EMSDataset(X_scaled[test_idx], y[test_idx]), batch_size=128, shuffle=False)
    
    # Entrenar (sólo 10 épocas para prueba rápida)
    model = AmbulanceDemandMLP(input_dim=5, dropout_rate=0.2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    criterion = nn.MSELoss()
    
    for epoch in range(10):
        model.train()
        for X_b, y_b in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            optimizer.step()
            
    # Evaluar
    model.eval()
    all_preds, all_reals = [], []
    with torch.no_grad():
        for X_b, y_b in test_loader:
            all_preds.extend(model(X_b).numpy().flatten())
            all_reals.extend(y_b.numpy().flatten())
            
    all_reals_arr = np.array(all_reals)
    all_preds_arr = np.array(all_preds)
    
    mae = np.mean(np.abs(all_reals_arr - all_preds_arr))
    ss_res = np.sum((all_reals_arr - all_preds_arr) ** 2)
    ss_tot = np.sum((all_reals_arr - np.mean(all_reals_arr)) ** 2)
    r2 = (1 - (ss_res / (ss_tot + 1e-8))) * 100
    
    print(f"\n--- Resultados para {model_name} ---")
    print(f"R2 Score (Precisión) : {r2:.2f}%")
    print(f"MAE (Error Absoluto) : {mae:.2f} incidentes")
    print(f"Promedio de Demanda Real en este dataset: {np.mean(all_reals_arr):.2f}")

# ==========================================
# Ejecución Comparativa
# ==========================================
if __name__ == "__main__":
    print("Iniciando Prueba Comparativa...")
    df = load_and_filter_data('911.csv')
    df = extract_time_features(df)
    
    # Obtener clima real una vez
    real_weather = get_sucre_historical_weather()
    
    # ---------------------------------------------------------
    # MODELO A: EL ACTUAL (Agrupado por Zone_ID pero sin usarlo)
    # ---------------------------------------------------------
    print("\nPreparando Modelo A (Actual - Con Zone_ID oculto)...")
    df_grid = create_geospatial_grid(df.copy(), grid_size=50)
    demand_df_A = df_grid.groupby(['Zone_ID', 'Month', 'DayOfWeek', 'Hour']).size().reset_index(name='Demand')
    demand_df_A = add_synthetic_features_local(demand_df_A, real_weather)
    
    train_and_eval(demand_df_A, "Modelo A (Actual)")
    
    # ---------------------------------------------------------
    # MODELO B: EL CORREGIDO (Agrupado Globalmente para la ciudad)
    # ---------------------------------------------------------
    print("\nPreparando Modelo B (Corregido - Demanda Global de la Ciudad)...")
    # Agrupamos saltándonos el Zone_ID, obteniendo el total de incidentes por hora en toda la ciudad
    demand_df_B = df.groupby(['Month', 'DayOfWeek', 'Hour']).size().reset_index(name='Demand')
    demand_df_B = add_synthetic_features_local(demand_df_B, real_weather)
    
    train_and_eval(demand_df_B, "Modelo B (Global Corregido)")
