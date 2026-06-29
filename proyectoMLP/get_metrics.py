import pandas as pd
import numpy as np
import torch
import warnings
warnings.filterwarnings('ignore')
from ems_predict_mlp import preprocess_pipeline, prepare_dataloaders, AmbulanceDemandMLP

print("Cargando y procesando datos para calcular métricas...")
demand_df = preprocess_pipeline('911.csv', grid_size=50)
train_loader, test_loader, norm_params = prepare_dataloaders(demand_df, batch_size=128)

model = AmbulanceDemandMLP(input_dim=5, dropout_rate=0.2)
model.load_state_dict(torch.load('ems_mlp_model.pth', map_location='cpu'))
model.eval()

all_preds = []
all_reals = []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        preds = model(X_batch)
        all_preds.extend(preds.numpy().flatten())
        all_reals.extend(y_batch.numpy().flatten())

all_reals_arr = np.array(all_reals)
all_preds_arr = np.array(all_preds)

mae = np.mean(np.abs(all_reals_arr - all_preds_arr))
ss_res = np.sum((all_reals_arr - all_preds_arr) ** 2)
ss_tot = np.sum((all_reals_arr - np.mean(all_reals_arr)) ** 2)
r2 = (1 - (ss_res / (ss_tot + 1e-8))) * 100

print(f"R2 Score (Precisión): {r2:.2f}%")
print(f"MAE (Error Absoluto Medio): {mae:.2f} incidentes")
