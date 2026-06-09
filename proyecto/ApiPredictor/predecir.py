import pickle
import pandas as pd
from prophet import Prophet  # Es necesario para cargar los modelos
import os

def predecir_venta(nombre_producto, factor_comercial, factor_eventos, factor_estacional, carpeta_modelos='modelos_entrenados'):
    """
    Carga un modelo de producto específico y predice el próximo mes.
    """
    
    # 1. Construir la ruta al archivo del modelo
    nombre_archivo = f"modelo_{nombre_producto}.pkl"
    ruta_modelo = os.path.join(carpeta_modelos, nombre_archivo)

    # 2. Verificar si el modelo existe
    if not os.path.exists(ruta_modelo):
        print(f"Error: No se encontró el archivo del modelo en: {ruta_modelo}")
        print("Asegúrate de que el nombre del producto sea correcto y el archivo .pkl exista.")
        return None

    # 3. Cargar el modelo
    try:
        with open(ruta_modelo, 'rb') as f:
            modelo = pickle.load(f)
    except Exception as e:
        print(f"Error al cargar el modelo '{ruta_modelo}': {e}")
        return None

    # 4. Preparar el DataFrame futuro
    # El modelo necesita un DataFrame con las fechas Y los regresores
    
    # Obtener el primer día del próximo mes
    hoy = pd.Timestamp('today')
    primer_dia_mes_actual = hoy.to_period('M').to_timestamp()
    proximo_mes = (primer_dia_mes_actual + pd.DateOffset(months=1))

    datos_futuros = {
        'ds': [proximo_mes],
        'Factor_Comercial_1': [factor_comercial],
        'Factor_Eventos_2': [factor_eventos],
        'Factor_Estacional_3': [factor_estacional]
    }
    df_futuro = pd.DataFrame(datos_futuros)

    # 5. Predecir
    try:
        forecast = modelo.predict(df_futuro)
        # Devolver la fila de predicción (la única que hay)
        return forecast.iloc[0]
    except Exception as e:
        print(f"Error al predecir con el modelo: {e}")
        return None

# =======================================================================
# --- PUNTO DE ENTRADA PRINCIPAL ---
# =======================================================================
if __name__ == "__main__":
    
    print("--- Simulador de Escenarios de Ventas (Local) ---")

    # --- 1. CONFIGURACIÓN ---
    # El nombre de la carpeta donde descomprimiste tus modelos
    CARPETA_CON_MODELOS = 'modelos_entrenados' 

    # --- 2. ENTRADA DEL USUARIO ---
    # ¡Aquí es donde simulas!
    # Cambia estos valores para probar diferentes escenarios
    
    PRODUCTO_A_PREDECIR = 'OUR10002030' # <-- Cambia el producto
    FACTOR_COMERCIAL_PROXIMO_MES = 1
    FACTOR_EVENTOS_PROXIMO_MES = 0
    FACTOR_ESTACIONAL_PROXIMO_MES = 0
    
    # --- 3. EJECUTAR PREDICCIÓN ---
    prediccion = predecir_venta(
        PRODUCTO_A_PREDECIR,
        FACTOR_COMERCIAL_PROXIMO_MES,
        FACTOR_EVENTOS_PROXIMO_MES,
        FACTOR_ESTACIONAL_PROXIMO_MES,
        carpeta_modelos=CARPETA_CON_MODELOS
    )

    # --- 4. MOSTRAR RESULTADO ---
    if prediccion is not None:
        print("\n" + "="*40)
        print("--- PREDICCIÓN PARA EL PRÓXIMO MES ---")
        print(f"Producto: \t{PRODUCTO_A_PREDECIR}")
        print(f"Escenario: \tFactor_Comercial={FACTOR_COMERCIAL_PROXIMO_MES}, Factor_Eventos={FACTOR_EVENTOS_PROXIMO_MES}, Factor_Estacional={FACTOR_ESTACIONAL_PROXIMO_MES}")
        print(f"Fecha: \t\t{prediccion['ds'].date()}")
        print(f"Ventas Estimadas: {prediccion['yhat']:.2f}")
        print(f"Rango Esperado: \t{prediccion['yhat_lower']:.2f} (min) - {prediccion['yhat_upper']:.2f} (max)")
        print("="*40)

    # --- 5. BONUS: Predecir varios productos a la vez ---
    print("\n\n--- BONUS: Reporte Rápido para Varios Productos ---")
    # Define los productos que quieres revisar
    productos_a_revisar = ['OUR10002030', 'OUR10003726', 'BIO130350'] 
    
    resultados = []
    for prod in productos_a_revisar:
        pred = predecir_venta(prod, FACTOR_COMERCIAL_PROXIMO_MES, FACTOR_EVENTOS_PROXIMO_MES, FACTOR_ESTACIONAL_PROXIMO_MES, CARPETA_CON_MODELOS)
        if pred is not None:
            resultados.append({
                'Producto': prod,
                'Ventas_Estimadas': pred['yhat']
            })
    
    if resultados:
        df_resultados = pd.DataFrame(resultados).sort_values(by='Ventas_Estimadas', ascending=False)
        print(df_resultados.to_string(index=False, float_format="%.2f"))