import pandas as pd
import numpy as np
import os
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def procesar_ames(file_path='dataset1/AmesHousing.xls'):
    """
    Dataset 1: Ames Housing
    Transformaciones: Imputación por mediana agrupada, One-Hot Encoding.
    """
    print("--- Procesando Dataset 1: Ames Housing ---")
    try:
        df = pd.read_excel(file_path)
        
        # 1. Imputación de nulos en LotFrontage por la mediana de su respectivo Barrio (Neighborhood)
        # Justificación: Diferentes barrios tienen distintos tamaños estándar de lotes (ej. un suburbio frente a una zona rústica).
        df['Lot Frontage'] = df.groupby('Neighborhood')['Lot Frontage'].transform(lambda x: x.fillna(x.median()))
        
        # En caso de quedar nulos remanentes si un barrio entero carecía de dato
        df['Lot Frontage'] = df['Lot Frontage'].fillna(0)
        
        # 2. Encoding de variables categóricas (Neighborhood)
        # Justificación: Los modelos ML estocásticos requieren transformar variables nominales matemáticamente; 
        # OHE (Dummies) es vital ya que los barrios no tienen un orden jerárquico dictado.
        df = pd.get_dummies(df, columns=['Neighborhood'], drop_first=True)
        
        output_name = 'ames_ready.csv'
        df.to_csv(output_name, index=False)
        print(f"✅ Éxito: Se exportó {output_name}")
    except Exception as e:
        print(f"❌ Error al procesar Ames Housing: {e}")

def procesar_mimic(file_path_adm='dataset2/mimic-iii-clinical-database-demo-1.4/ADMISSIONS.csv', 
                   file_path_diag='dataset2/mimic-iii-clinical-database-demo-1.4/DIAGNOSES_ICD.csv'):
    """
    Dataset 2: MIMIC-III Demo
    Transformaciones: Fechas a Días (Length of Stay), reducción de ICD9.
    """
    print("\n--- Procesando Dataset 2: MIMIC-III Demo ---")
    try:
        df_adm = pd.read_csv(file_path_adm)
        
        # 1. Conversión de fechas a 'Días de estadía' (LOS)
        # Forzar columnas a mayúscula por variaciones de versión en MIMIC
        df_adm.columns = df_adm.columns.str.upper()
        
        # Justificación: Los algoritmos no operan con crudos 'datetime'. La duración real (tiempo de vida o internación) sirve como un excelente Feature o de Target predictivo.
        df_adm['ADMITTIME'] = pd.to_datetime(df_adm['ADMITTIME'])
        df_adm['DISCHTIME'] = pd.to_datetime(df_adm['DISCHTIME'])
        df_adm['LOS_dias'] = (df_adm['DISCHTIME'] - df_adm['ADMITTIME']).dt.total_seconds() / (24 * 3600)
        
        # 2. Reducción de ICD9
        try:
            df_diag = pd.read_csv(file_path_diag)
            df_diag.columns = df_diag.columns.str.upper()
            # Justificación: Agrupar por los primeros 3 dígitos colapsa enfermedades de muy baja frecuencia (cardinalidad alta),
            # previniendo la maldición de la dimensionalidad en modelos futuros.
            df_diag['ICD9_BASE'] = df_diag['ICD9_CODE'].astype(str).str[:3]
            df = df_adm.merge(df_diag[['HADM_ID', 'ICD9_BASE']].drop_duplicates(subset=['HADM_ID']), on='HADM_ID', how='left')
        except FileNotFoundError:
            print("⚠️ No se encontró la tabla de diagnósticos. Se procesa solo ADMISSIONS.")
            df = df_adm
            
        output_name = 'mimic_ready.csv'
        df.to_csv(output_name, index=False)
        print(f"✅ Éxito: Se exportó {output_name}")
    except Exception as e:
        print(f"❌ Error al procesar MIMIC-III: {e}")

def procesar_nhanes(dir_path='dataset3/'):
    """
    Dataset 3: NHANES 2021-2023
    Transformaciones: Merge secuencial y filtro de métricas.
    """
    print("\n--- Procesando Dataset 3: NHANES ---")
    try:
        # Carga
        bax = pd.read_sas(os.path.join(dir_path, 'BAX_L.XPT'))
        bpx = pd.read_sas(os.path.join(dir_path, 'BPXO_L.XPT'))
        bmx = pd.read_sas(os.path.join(dir_path, 'BMX_L.XPT'))
        lux = pd.read_sas(os.path.join(dir_path, 'LUX_L.XPT'))

        # 1. Unión Múltiple (Merge)
        # Justificación: En estudios como NHANES la interoperabilidad se da por paciente (SEQN).
        # Unir estas tablas revela la interdependencia sistémica (tensión arterial vs grado de grasa en hígado).
        df = bax.merge(bpx, on='SEQN', how='inner') \
                .merge(bmx, on='SEQN', how='inner') \
                .merge(lux, on='SEQN', how='inner')

        output_name = 'nhanes_ready.csv'
        df.to_csv(output_name, index=False)
        print(f"✅ Éxito: Se exportó {output_name}")
    except Exception as e:
        print(f"❌ Error al procesar NHANES: Archivos insuficientes - {e}")

def procesar_bike_sharing(file_path='dataset4/sampleSubmission.csv'):
    """
    Dataset 4: Bike Sharing
    Transformaciones: Extracción de Datetime, Transformada Logarítmica.
    """
    print("\n--- Procesando Dataset 4: Bike Sharing ---")
    try:
        df = pd.read_csv(file_path)
        
        # 1. Extracción Estacional
        # Justificación: Al aislar fracciones de temporalidad, los árboles de decisión son capaces de agrupar franjas climáticas o el comportamiento punta de horas laborales.
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['hora'] = df['datetime'].dt.hour
        df['dia'] = df['datetime'].dt.day
        df['mes'] = df['datetime'].dt.month
        
        # 2. Log-Transform
        # Justificación: La demanda de recursos es no-negativa y normalmente es sesgada con picos asimétricos.
        # Log(1+x) suaviza el 'Outlierness' de días saturados e insta normalidad.
        if 'count' in df.columns:
            df['log_count'] = np.log1p(df['count'])
            
        output_name = 'bikesharing_ready.csv'
        df.to_csv(output_name, index=False)
        print(f"✅ Éxito: Se exportó {output_name}")
    except Exception as e:
        print(f"❌ Error al procesar Bike Sharing: {e}")

def procesar_adult_income(file_path='dataset5/adult.data'):
    """
    Dataset 5: Adult Income
    Transformaciones: Imputación textual explícita, Binarización Target.
    """
    print("\n--- Procesando Dataset 5: Adult Income ---")
    try:
        cols = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 
                'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 
                'hours-per-week', 'native-country', 'income']
        df = pd.read_csv(file_path, names=cols, sep=r',\s+', engine='python') # Puede separar con coma y espacio
        
        # 1. Tratamiento Nulos Encubiertos
        # Justificación: En encuestas censales el '?' corrompe el flujo estructural.
        # Asignar 'Unknown' aporta una señal clave de regresión al modelo (Ej. desempleados evaden declarar estado laboral).
        df = df.replace('?', 'Unknown')
        
        # 2. Binarización
        df['income_bin'] = df['income'].apply(lambda x: 1 if x == '>50K' else 0)
        
        output_name = 'adult_ready.csv'
        df.to_csv(output_name, index=False)
        print(f"✅ Éxito: Se exportó {output_name}")
    except Exception as e:
        print(f"❌ Error al procesar Adult Income: {e}")

def procesar_crx(file_path='dataset6/crx.data'):
    """
    Dataset 6: Credit Approval (CRX)
    Transformaciones: Imputación mixta de variables cifradas, MinMaxScaler.
    """
    print("\n--- Procesando Dataset 6: Credit Approval ---")
    try:
        df = pd.read_csv(file_path, header=None)
        df = df.replace('?', np.nan)
        
        # Limpieza crucial: Forzar conversión a numérico porque los '?' convirtieron columnas enteras en 'object'
        for col in df.columns:
            try:
                df[col] = df[col].astype(float)
            except ValueError:
                pass # Si falla es categórica pura (caracteres)
            
        scaler = MinMaxScaler()
        
        # Variables continuas y discretas combinadas por ofuscación bancaria
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                # Imputación mediana frente a media simple, es invulnerable a distorsión de outliers de crédito.
                df[col] = df[col].fillna(df[col].median())
                
                # Escalado MinMax Vectorial
                # Justificación: Las redes neuronales o SVM logran convergencia superior si todos los montos fiduciarios pesan entre 0 y 1.
                df[col] = scaler.fit_transform(df[[col]])
            else:
                if col == 15: # Nuestra variable de output
                    df[col] = df[col].apply(lambda x: 1 if x == '+' else 0)
                else:
                    # Imputación por la moda en cualitativas. 
                    # Justificación: Mantiene el balance de proporciones asumiendo el perfil prevalente predecible.
                    try:
                        df[col] = df[col].fillna(df[col].mode()[0])
                    except:
                        pass
                
        output_name = 'crx_ready.csv'
        df.to_csv(output_name, index=False)
        print(f"✅ Éxito: Se exportó {output_name}")
    except Exception as e:
        print(f"❌ Error al procesar CRX: {e}")

def procesar_australian(file_path='dataset7/australian.dat'):
    """
    Dataset 7: Statlog (Australian Credit)
    Transformaciones: Clipping Quantil 1/99, Estandarización de varianza.
    """
    print("\n--- Procesando Dataset 7: Statlog Australian ---")
    try:
        # Carecemos de Headers formales
        df = pd.read_csv(file_path, header=None, sep=r'\s+')
        
        scaler = StandardScaler()
        num_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in num_cols:
            if df[col].nunique() > 2: # Excluyendo las pre-codificadas en formato binario
                # 1. Outlier Clipping
                # Justificación: Recortar perfiles atípicos con montos irracionales (clip a límites) preserva a las redes numéricas del arrastre de la gradiente por penalización extrema.
                lower = df[col].quantile(0.01)
                upper = df[col].quantile(0.99)
                df[col] = df[col].clip(lower, upper)
                
                # 2. StandardScaler Z-Score
                # Justificación: Centra las distribuciones en el eje 0 (media empírica inter-atribucional estable).
                df[col] = scaler.fit_transform(df[[col]])
                
        output_name = 'australian_ready.csv'
        df.to_csv(output_name, index=False)
        print(f"✅ Éxito: Se exportó {output_name}")
    except Exception as e:
        print(f"❌ Error al procesar Australian: {e}")

def procesar_breast_cancer(file_path='dataset8/breast-cancer-wisconsin.data'):
    """
    Dataset 8: Breast Cancer Wisconsin
    Transformaciones: Remoción quirúrgica de ruidos y features no descriptivos (ID).
    """
    print("\n--- Procesando Dataset 8: Breast Cancer ---")
    try:
        cols = ['ID', 'Clump_Thickness', 'U_Cell_Size', 'U_Cell_Shape', 'Marginal_Adhesion', 
                'Single_Epithelial_Size', 'Bare_Nuclei', 'Bland_Chromatin', 'Normal_Nucleoli', 
                'Mitoses', 'Class']
        df = pd.read_csv(file_path, names=cols, na_values='?')
        
        # 1. Eliminación de ID
        # Justificación: Los identificadores inyectan falso aprendizaje por el sesgo espurio y la correlacion transaccional.
        df = df.drop(columns=['ID'])
        
        # 2. Eliminación (Listwise-Drop) de filas en Bare_Nuclei
        # Justificación: Son exámenes celulares vivos. Inyectar o inventar (imputar) características mitóticas a núcleos invisibles es una falta ética grave que corrompe la fiabilidad predictiva oncológica; lo mejor es descartar esas minúsculas filas.
        df = df.dropna(subset=['Bare_Nuclei'])
        df['Bare_Nuclei'] = df['Bare_Nuclei'].astype(float)
        
        output_name = 'breastcancer_ready.csv'
        df.to_csv(output_name, index=False)
        print(f"✅ Éxito: Se exportó {output_name}")
    except Exception as e:
        print(f"❌ Error al procesar Breast Cancer: {e}")

def procesar_meningitis(file_path='dataset9/mening missing 12.csv'):
    """
    Dataset 9: Meningitis (Missing 12)
    Transformaciones: KNN-Imputer predictivo para laboratorio.
    """
    print("\n--- Procesando Dataset 9: Meningitis ---")
    try:
        df = pd.read_csv(file_path)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # 1. KNN Imputer
        # Justificación: La ausencia de ciertos exámenes de LCR (líquido cefalorraquídeo) interrumpe la evaluación de etiología bacteriológica.
        # Imputar basándose holísticamente en los 5 pacientes clónicos más congruentes vectorialmente rescata el tejido fenotípico de forma brillante que una 'Falsa Media Global' arruinaría.
        imputer = KNNImputer(n_neighbors=5)
        df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
        
        output_name = 'meningitis_ready.csv'
        df.to_csv(output_name, index=False)
        print(f"✅ Éxito: Se exportó {output_name}")
    except Exception as e:
        print(f"❌ Error al procesar Meningitis: {e}")

if __name__ == "__main__":
    print("🚀 Iniciando Pipeline Data Engineering ETL de los 9 Datasets...\n")
    
    procesar_ames()
    procesar_mimic() 
    procesar_nhanes() 
    procesar_bike_sharing()
    procesar_adult_income()
    procesar_crx()
    procesar_australian()
    procesar_breast_cancer()
    procesar_meningitis()
    
    print("\n🏁 Tarea Finalizada. Archivos *_ready.csv consolidados en la raíz.")
