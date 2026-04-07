import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

def preparar_nhanes():
    print("--- Procesando Dataset 3: NHANES ---")
    try:
        # Cargar archivos XPT
        bax = pd.read_sas('BAX_L.XPT')
        bpx = pd.read_sas('BPXO_L.XPT')
        bmx = pd.read_sas('BMX_L.XPT')
        lux = pd.read_sas('LUX_L.XPT')

        # Unión (Merge) por SEQN
        df = bax.merge(bpx, on='SEQN', how='inner') \
                .merge(bmx, on='SEQN', how='inner') \
                .merge(lux, on='SEQN', how='inner')

        # Selección de variables críticas según el README
        cols = ['SEQN', 'BAXSTAT', 'BPXOSY1', 'BMXBMI', 'LUXSMED']
        df = df[cols]

        # Limpieza básica: Eliminar filas donde el IMC o la presión sean nulos
        df = df.dropna(subset=['BMXBMI', 'BPXOSY1'])
        
        df.to_csv('nhanes_final_limpio.csv', index=False)
        print("✅ NHANES guardado como 'nhanes_final_limpio.csv'")
    except Exception as e:
        print(f"❌ Error en NHANES: {e}")

def preparar_uci_adult(file_path):
    print("\n--- Procesando Dataset 5: Adult Income ---")
    # Este dataset suele venir sin cabeceras
    cols = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 
            'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 
            'hours-per-week', 'native-country', 'income']
    
    df = pd.read_csv(file_path, names=cols, sep=', ', engine='python')
    
    # Manejo de nulos encubiertos '?'
    df = df.replace('?', np.nan)
    df['workclass'] = df['workclass'].fillna('Unknown')
    df['occupation'] = df['occupation'].fillna('Unknown')
    
    # Codificación del Target
    df['income'] = df['income'].apply(lambda x: 1 if x == '>50K' else 0)
    
    df.to_csv('adult_limpio.csv', index=False)
    print("✅ Adult Income guardado como 'adult_limpio.csv'")

def preparar_credit_approval(file_path):
    print("\n--- Procesando Dataset 6: Credit Approval (CRX) ---")
    df = pd.read_csv(file_path, header=None)
    
    # Reemplazar '?' por NaN
    df = df.replace('?', np.nan)
    
    # El target es la última columna
    df.iloc[:, -1] = df.iloc[:, -1].apply(lambda x: 1 if x == '+' else 0)
    
    # Imputación simple para valores perdidos
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].median())
            
    df.to_csv('credit_approval_limpio.csv', index=False)
    print("✅ Credit Approval guardado como 'credit_approval_limpio.csv'")

# Ejecución principal
if __name__ == "__main__":
    preparar_nhanes()
    # Descomenta las líneas de abajo y pon la ruta de tus archivos locales
    # preparar_uci_adult('adult.data')
    # preparar_credit_approval('crx.data')