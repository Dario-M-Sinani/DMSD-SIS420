import pandas as pd
import numpy as np
import os

# Mapeo estructurado para iterar y generar datos lógicos
productos_agro = {
    'SINCROCIP': {'nombre': 'Cera Estampada de Abeja (Pack x50)', 'base': 550, 'meses_pico': [9, 10, 11]},
    'SINCROFORTE': {'nombre': 'Suplemento Proteico Colmenas 20kg', 'base': 250, 'meses_pico': [5, 6, 7, 8]},
    'RESOLUTOR': {'nombre': 'Tratamiento Orgánico Varroa', 'base': 110, 'meses_pico': [3, 4, 9, 10]},
    'OUROVIT-500': {'nombre': 'Nutriente Foliar Estimulante 5L', 'base': 250, 'meses_pico': [8, 9, 10]},
    'CIPROLAC': {'nombre': 'Sanitizante de Material Apícola', 'base': 500, 'meses_pico': [3, 4, 11, 12]},
    'SUPERHION-5L': {'nombre': 'Control Ecológico de Plagas (5L)', 'base': 500, 'meses_pico': [10, 11, 12, 1]},
    'IMPACTO-25': {'nombre': 'Repelente de Ácaros Concentrado', 'base': 750, 'meses_pico': [4, 5, 9]},
    'IMPACTO-100': {'nombre': 'Repelente de Ácaros Concentrado (Medio)', 'base': 520, 'meses_pico': [4, 5, 9]},
    'DOXIFIN': {'nombre': 'Antibiótico Apícola Loque', 'base': 420, 'meses_pico': [2, 3, 4]},
    'OUROVAC-AFT': {'nombre': 'Envases de Vidrio para Miel (Caja x100)', 'base': 1500, 'meses_pico': [11, 12, 1, 2]},
    'SUPERHION-1L': {'nombre': 'Control Ecológico de Plagas (1L)', 'base': 635, 'meses_pico': [10, 11, 12, 1]},
    'OUROVIT-250': {'nombre': 'Nutriente Foliar Estimulante 2.5L', 'base': 740, 'meses_pico': [8, 9, 10]},
    'OUROTETRA': {'nombre': 'Sanitizador de Suelos Orgánico', 'base': 1400, 'meses_pico': [1, 2, 3, 4]},
    'IMPACTO-1000': {'nombre': 'Repelente de Ácaros Industrial (1L)', 'base': 550, 'meses_pico': [4, 5, 9]}
}

def generar_datos_agro_apicola(ruta_salida='historico_ventas.csv'):
    print("Generando dataset histórico para el sector Agro/Apícola (+1000 registros)...")
    
    # Rango de fechas: 18 años de historial mensual (216 meses)
    fechas = pd.date_range(start='2008-01-01', end='2025-12-01', freq='MS')
    
    registros = []
    np.random.seed(7)  # Semilla para consistencia analítica
    
    for producto, info in productos_agro.items():
        base_ventas = info['base']
        meses_pico = info['meses_pico']
        
        # Crecimiento lineal estimado (2% al año aproximadamente sobre la base)
        crecimiento_anual = base_ventas * 0.02
        
        for fecha in fechas:
            promo = 1 if np.random.rand() > 0.80 else 0
            feria = 1 if np.random.rand() > 0.90 else 0
            
            es_temporada_alta = 1 if fecha.month in meses_pico else 0
            
            anios_transcurridos = fecha.year - 2008
            ventas_estimadas = base_ventas + (anios_transcurridos * crecimiento_anual)
            
            if es_temporada_alta == 1:
                ventas_estimadas += (base_ventas * 0.45)
            if promo == 1:
                ventas_estimadas += (base_ventas * 0.20)
            if feria == 1:
                ventas_estimadas += (base_ventas * 0.15)
                
            ruido_climatico = np.random.normal(0, base_ventas * 0.07)
            ventas_finales = max(5, int(ventas_estimadas + ruido_climatico))
            
            registros.append({
                'codigo_producto': producto,
                'fecha': fecha.strftime('%Y-%m-%d'),
                'cantidad_vendida': ventas_finales,
                'promo': promo,
                'feria': feria,
                'lluvia': es_temporada_alta  # Mapeado a Factor Estacional
            })
            
    df_agro = pd.DataFrame(registros)
    df_agro.to_csv(ruta_salida, index=False)
    
    print(f"¡Éxito! Dataset agropecuario guardado en '{ruta_salida}' con {len(df_agro)} filas.")
    print("\nPrimeras filas del historial generado (Muestra):")
    print(df_agro.head(12))

if __name__ == '__main__':
    generar_datos_agro_apicola()