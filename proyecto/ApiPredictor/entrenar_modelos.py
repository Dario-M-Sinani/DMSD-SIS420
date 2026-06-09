import pandas as pd
from prophet import Prophet
import pickle
import os

# ==============================================================================
# EJEMPLO DE ESTRUCTURA DEL ARCHIVO CSV (historico_ventas.csv)
# ==============================================================================
# El archivo debe ser un CSV separado por comas (,) con encabezados.
# Nombres obligatorios de columnas (el orden no importa):
# codigo_producto, fecha, cantidad_vendida, promo, feria, lluvia
#
# Ejemplo de datos:
# codigo_producto,fecha,cantidad_vendida,promo,feria,lluvia
# OUR10002030,2023-01-01,150,1,0,1
# OUR10002030,2023-02-01,200,0,1,0
# BIO130350,2023-01-01,80,0,0,1
# ==============================================================================

def entrenar_modelos(ruta_csv='historico_ventas.csv', carpeta_salida='modelos_entrenados'):
    """
    Lee datos históricos de ventas, adapta los nombres de los regresores a su versión genérica
    y entrena/guarda un modelo Prophet (.pkl) individual por cada producto.
    """
    print("Iniciando proceso de entrenamiento automático de modelos...")
    
    if not os.path.exists(ruta_csv):
        print(f"Error: No se encontró el archivo de datos históricos en '{ruta_csv}'.")
        print("Por favor, crea el archivo CSV basándote en el formato comentado arriba.")
        return

    # Crear carpeta de salida si no existe
    if not os.path.exists(carpeta_salida):
        os.makedirs(carpeta_salida)
        print(f"Se creó el directorio de salida: {carpeta_salida}")

    print(f"Cargando datos desde {ruta_csv}...")
    try:
        df = pd.read_csv(ruta_csv)
    except Exception as e:
        print(f"Error al leer el CSV: {e}")
        return

    # Verificar que las columnas necesarias existan en el CSV original subido por el usuario
    columnas_esperadas = ['codigo_producto', 'fecha', 'cantidad_vendida', 'promo', 'feria', 'lluvia']
    if not all(col in df.columns for col in columnas_esperadas):
        print(f"Error: El archivo CSV no contiene todas las columnas requeridas.")
        print(f"Esperadas: {columnas_esperadas}")
        print(f"Encontradas: {list(df.columns)}")
        return

    # Transformación Estructural de Pandas: Renombrar para Prophet y para nuestro uso genérico
    df_preparado = df.rename(columns={
        'fecha': 'ds',
        'cantidad_vendida': 'y',
        'promo': 'Factor_Comercial_1',
        'feria': 'Factor_Eventos_2',
        'lluvia': 'Factor_Estacional_3'
    })

    # Asegurar que la columna ds sea tratada como fecha
    df_preparado['ds'] = pd.to_datetime(df_preparado['ds'])

    # Obtener la lista de productos únicos para iterar
    productos = df_preparado['codigo_producto'].unique()
    print(f"\nSe encontraron {len(productos)} productos distintos en el dataset.")

    for producto in productos:
        print(f"\n[+] Entrenando modelo para: {producto}")
        
        # Aislar los datos del producto actual
        df_prod = df_preparado[df_preparado['codigo_producto'] == producto].copy()
        
        # Prophet suele requerir al menos unos cuantos puntos de datos; aquí forzamos un mínimo
        if len(df_prod) < 2:
            print(f"  -> Advertencia: No hay suficientes datos históricos para el producto {producto} (Solo {len(df_prod)} filas). Se omite el entrenamiento.")
            continue

        # 1. Instanciar el modelo de Machine Learning
        modelo = Prophet(
            yearly_seasonality=True,  # Para captar ciclos largos
            weekly_seasonality=False, # Desactivado por ser datos típicamente mensuales
            daily_seasonality=False   # No hay grano diario
        )

        # 2. Agregar los Regresores Externos (Nombres Genéricos)
        modelo.add_regressor('Factor_Comercial_1')
        modelo.add_regressor('Factor_Eventos_2')
        modelo.add_regressor('Factor_Estacional_3')

        try:
            # 3. Entrenar (Ajustar modelo)
            modelo.fit(df_prod)
            
            # 4. Guardar Serializado (.pkl)
            nombre_archivo = f"modelo_{producto}.pkl"
            ruta_guardado = os.path.join(carpeta_salida, nombre_archivo)
            
            with open(ruta_guardado, 'wb') as f:
                pickle.dump(modelo, f)
                
            print(f"  -> Modelo guardado exitosamente en: {ruta_guardado}")
        except Exception as e:
            print(f"  -> Error al entrenar o guardar el modelo para {producto}: {e}")

    print("\nProceso de entrenamiento completado.")

if __name__ == "__main__":
    # Ejecutar la función principal. Se asumirá que historico_ventas.csv está junto al script
    entrenar_modelos()
