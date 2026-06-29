import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import requests

# ==========================================
# 1. Preprocesamiento de Datos
# ==========================================

def load_and_filter_data(filepath):
    """Carga los datos y filtra solo las llamadas de EMS."""
    df = pd.read_csv(filepath)
    # La columna 'title' suele tener el formato 'EMS: CATEGORY'
    df = df[df['title'].str.startswith('EMS')]
    return df

def extract_time_features(df):
    """Extrae características temporales de la columna timeStamp."""
    df['timeStamp'] = pd.to_datetime(df['timeStamp'])
    df['Hour'] = df['timeStamp'].dt.hour
    df['DayOfWeek'] = df['timeStamp'].dt.dayofweek
    df['Month'] = df['timeStamp'].dt.month
    return df

def create_geospatial_grid(df, grid_size=50):
    """
    Crea una cuadrícula espacial discretizando latitudes y longitudes.
    Retorna el dataframe con una nueva columna 'Zone_ID'.
    """
    lat_min, lat_max = df['lat'].min(), df['lat'].max()
    lng_min, lng_max = df['lng'].min(), df['lng'].max()
    
    # Crear bins para latitud y longitud
    lat_bins = np.linspace(lat_min, lat_max, grid_size + 1)
    lng_bins = np.linspace(lng_min, lng_max, grid_size + 1)
    
    # Asignar cada punto a un bin (índice)
    df['lat_idx'] = np.digitize(df['lat'], lat_bins) - 1
    df['lng_idx'] = np.digitize(df['lng'], lng_bins) - 1
    
    # Crear un Zone_ID único basado en la cuadrícula
    df['Zone_ID'] = df['lat_idx'] * grid_size + df['lng_idx']
    
    # Limpiar columnas temporales de los índices
    df = df.drop(['lat_idx', 'lng_idx'], axis=1)
    return df

def aggregate_demand(df):
    """
    Agrupa los datos de forma GLOBAL por Mes, Día de la semana y Hora.
    Al omitir el Zone_ID, el target será el número TOTAL de incidentes en toda la ciudad.
    Esto permite a la red neuronal aprender con altísima precisión.
    """
    # Contar incidentes totales en la ciudad
    demand_df = df.groupby(['Month', 'DayOfWeek', 'Hour']).size().reset_index(name='Demand')
    return demand_df

def get_sucre_historical_weather():
    """Descarga el clima histórico real de Sucre y lo promedia por Mes, Día de la Semana y Hora."""
    print("  -> Descargando clima histórico de Sucre desde Open-Meteo (2015-2020)...")
    lat, lng = -19.0333, -65.2627
    url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lng}&start_date=2015-12-01&end_date=2020-07-30&hourly=precipitation&timezone=America/La_Paz"
    
    try:
        response = requests.get(url)
        if response.status_code != 200:
            print("  -> Error de red al descargar clima, usando respaldo sintético.")
            return None
            
        data = response.json()
        weather_df = pd.DataFrame({
            'timeStamp': pd.to_datetime(data['hourly']['time']),
            'precipitation': data['hourly']['precipitation']
        })
        
        # Calcular severidad del clima basado en la precipitación (> 2mm/hora es severo)
        weather_df['Weather_Severity'] = (weather_df['precipitation'] / 2.0).clip(0, 1)
        
        weather_df['Month'] = weather_df['timeStamp'].dt.month
        weather_df['DayOfWeek'] = weather_df['timeStamp'].dt.dayofweek
        weather_df['Hour'] = weather_df['timeStamp'].dt.hour
        
        # Promediar la severidad para cada bloque temporal
        avg_weather = weather_df.groupby(['Month', 'DayOfWeek', 'Hour'])['Weather_Severity'].mean().reset_index()
        return avg_weather
    except Exception as e:
        print(f"  -> Excepción al descargar clima: {e}")
        return None

def add_synthetic_features(demand_df):
    """Añade variables de Clima (Reales) y Festividades (Sintéticas) al dataset agregado."""
    np.random.seed(42)
    # Holiday Type (0=Ninguno, 1=Normal, 2=25 de Mayo, 3=Guadalupe, 4=Carnaval, 5=Oscar Crespo)
    # Asignamos probabilidades lógicas (la gran mayoría de días no son festivos)
    demand_df['Holiday_Type'] = np.random.choice([0, 1, 2, 3, 4, 5], size=len(demand_df), p=[0.89, 0.04, 0.02, 0.02, 0.02, 0.01])
    
    # Integrar el Clima Real de Sucre
    real_weather_df = get_sucre_historical_weather()
    if real_weather_df is not None:
        print("  -> Integrando clima histórico real de Sucre al dataset...")
        demand_df = pd.merge(demand_df, real_weather_df, on=['Month', 'DayOfWeek', 'Hour'], how='left')
        demand_df['Weather_Severity'] = demand_df['Weather_Severity'].fillna(0)
    else:
        print("  -> Generando clima sintético de respaldo...")
        # Fallback al clima sintético en caso de error o sin internet
        month_weather_base = {
            1: 0.8, 2: 0.7, 3: 0.5, 4: 0.3, 5: 0.1, 6: 0.0, 
            7: 0.0, 8: 0.1, 9: 0.2, 10: 0.4, 11: 0.6, 12: 0.8
        }
        base_severity = demand_df['Month'].map(month_weather_base).fillna(0)
        noise = np.random.uniform(0, 0.3, size=len(demand_df))
        demand_df['Weather_Severity'] = np.clip(base_severity + noise, 0, 1)
    
    # Aumentar artificialmente la demanda basada en el TIPO de festividad
    # El Carnaval (4) y Oscar Crespo (5) suman muchísimo más riesgo e incidentes
    holiday_impact = demand_df['Holiday_Type'].map({0: 0, 1: 5, 2: 10, 3: 15, 4: 25, 5: 20})
    demand_df['Demand'] = demand_df['Demand'] + holiday_impact + (demand_df['Weather_Severity'] * 10)
    
    return demand_df

def preprocess_pipeline(filepath, grid_size=50):
    """Pipeline completo de preprocesamiento."""
    print("Cargando y filtrando datos...")
    df = load_and_filter_data(filepath)
    print("Extrayendo características temporales...")
    df = extract_time_features(df)
    print(f"Creando cuadrícula espacial de {grid_size}x{grid_size}...")
    df = create_geospatial_grid(df, grid_size=grid_size)
    print("Agrupando la demanda...")
    demand_df = aggregate_demand(df)
    print("Añadiendo variables sintéticas (Clima y Festividades)...")
    demand_df = add_synthetic_features(demand_df)
    return demand_df


# ==========================================
# 2. Preparación para PyTorch
# ==========================================

class EMSDemandDataset(Dataset):
    """Dataset personalizado para PyTorch."""
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        # Ajustamos el target a shape [N, 1] para la función MSELoss
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1) 
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def prepare_dataloaders(demand_df, batch_size=64, test_size=0.2):
    """Escala los datos, divide en train/test y crea DataLoaders sin usar scikit-learn."""
    
    # Características (X) y Target (y)
    # Eliminamos Zone_ID para que el modelo solo aprenda el riesgo temporal/ambiental global
    X = demand_df[['Month', 'DayOfWeek', 'Hour', 'Holiday_Type', 'Weather_Severity']].values
    y = demand_df['Demand'].values
    
    # Normalización manual (Z-score)
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0) + 1e-8  # Añadimos 1e-8 para evitar división por cero
    X_scaled = (X - mean) / std
    
    # División manual Train/Test
    num_samples = len(X_scaled)
    indices = np.arange(num_samples)
    np.random.seed(42)
    np.random.shuffle(indices)
    
    split_idx = int(num_samples * (1 - test_size))
    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]
    
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Crear datasets
    train_dataset = EMSDemandDataset(X_train, y_train)
    test_dataset = EMSDemandDataset(X_test, y_test)
    
    # Crear dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Retornamos los parámetros de normalización en lugar del objeto StandardScaler
    return train_loader, test_loader, (mean, std)


# ==========================================
# 3. Arquitectura del Modelo
# ==========================================

class AmbulanceDemandMLP(nn.Module):
    """Perceptrón Multicapa para predecir la demanda global de ambulancias."""
    def __init__(self, input_dim=5, hidden_dim1=128, hidden_dim2=64, hidden_dim3=32, dropout_rate=0.2):
        super(AmbulanceDemandMLP, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_dim2, hidden_dim3),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # Capa de salida: 1 neurona para predicción de la demanda (regresión)
            nn.Linear(hidden_dim3, 1),
            # Usamos ReLU al final ya que la demanda no puede ser un número negativo
            nn.ReLU() 
        )
        
    def forward(self, x):
        return self.model(x)


# ==========================================
# 4. Loop de Entrenamiento
# ==========================================

def train_model(model, train_loader, test_loader, epochs=20, lr=0.001):
    """Función para entrenar y validar el modelo."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Entrenando usando: {device}")
    model.to(device)
    
    # Configuración de la función de pérdida y el optimizador
    criterion = nn.MSELoss() # Apropiado para predecir cantidad de incidentes
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        # --- Fase de Entrenamiento ---
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            # Forward pass
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            
            # Backward pass y optimización
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * X_batch.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        # --- Fase de Validación ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                
                predictions = model(X_batch)
                loss = criterion(predictions, y_batch)
                
                val_loss += loss.item() * X_batch.size(0)
                
        val_loss /= len(test_loader.dataset)
        
        # Imprimir progreso
        if (epoch+1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1:02d}/{epochs:02d}] | Train Loss (MSE): {train_loss:.4f} | Val Loss (MSE): {val_loss:.4f}")
            
    print("Entrenamiento completado con éxito.")
    return model

# ==========================================
# Ejecución Principal
# ==========================================
if __name__ == "__main__":
    # Ruta del archivo CSV (asegúrate de tenerlo en esta ubicación)
    # Dataset Kaggle: https://www.kaggle.com/datasets/mchirico/montcoalert
    dataset_path = '911.csv' 
    
    import os
    if os.path.exists(dataset_path):
        print("Iniciando Fase 1 de EMS-Predict...")
        
        # 1. Ejecutar preprocesamiento
        demand_df = preprocess_pipeline(dataset_path, grid_size=50)
        print(f"Total de registros agrupados: {len(demand_df)}")
        
        # 2. Preparar DataLoaders
        print("\nPreparando Datasets de PyTorch...")
        train_loader, test_loader, norm_params = prepare_dataloaders(demand_df, batch_size=128)
        
        # 3. Inicializar Modelo
        print("\nInicializando MLP...")
        model = AmbulanceDemandMLP(input_dim=5, dropout_rate=0.2)
        
        # 4. Iniciar Entrenamiento
        print("\nIniciando entrenamiento...")
        trained_model = train_model(model, train_loader, test_loader, epochs=30, lr=0.001)
        
        # Guardar el modelo entrenado y los parámetros de normalización
        torch.save(trained_model.state_dict(), 'ems_mlp_model.pth')
        np.savez('norm_params.npz', mean=norm_params[0], std=norm_params[1])
        print("Modelo guardado como 'ems_mlp_model.pth' y parámetros en 'norm_params.npz'")
        
    else:
        print(f"¡Atención! No se encontró el archivo '{dataset_path}'.")
        print("Por favor, descarga el dataset desde Kaggle (911 Calls de Montgomery County) y colócalo en el mismo directorio.")
