import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Desactivar la ventana emergente (solo guarda el archivo)
import matplotlib
matplotlib.use('Agg') 

# 1. Configuración de la simulación
np.random.seed(None) 
n = 10000

# Parámetros aleatorios del Hardware (Simulando Gigabytes)
# b: RAM Base entre 8GB y 16GB
b = np.random.uniform(8, 16) 
# m: Incremento por unidad de carga (0.01 a 0.05 GB por unidad)
m = np.random.uniform(0.01, 0.05) 

# 2. Creación de datos (Carga vs RAM en GB)
X = np.random.uniform(100, 2000, n)
ruido = np.random.normal(0, 0.5, n) # Variación de 500MB aprox.
Y_real = (m * X) + b + ruido
Y_predicha = (m * X) + b

# 3. Guardar Dataset
df = pd.DataFrame({'Carga_Trabajo_X': X, 'RAM_Consumida_GB_Y': Y_real})
df.to_csv('reporte_datacenter.csv', index=False)

# 4. EXPLICACIÓN RESUMIDA EN CONSOLA
print("\n" + "="*60)
print("   SISTEMA DE PREDICCIÓN DE INFRAESTRUCTURA (IA)")
print("="*60)
print(f" ECUACIÓN: y = {m:.4f} * x + {b:.2f}")
print("-"*60)
print(f" > UNIDAD DE MEDIDA: Gigabytes (GB)")
print(f" > b ({b:.2f} GB): Es la RAM base que ya usa el servidor solo")
print(f"             por estar encendido (Sistema y Modelos).")
print(f" > m ({m:.4f} GB): Es el peso del proceso. Por cada unidad de")
print(f"             carga, se consumen {m:.4f} GB adicionales.")
print("="*60)
print(" ACCIÓN: Gráfico guardado como 'grafico_prediccion.png'")
print("="*60 + "\n")

# 5. Generar y guardar el gráfico
plt.figure(figsize=(10, 6))
plt.scatter(X, Y_real, color='lightsteelblue', s=1, alpha=0.3, label='Datos reales del Servidor')
plt.plot(X, Y_predicha, color='navy', linewidth=2, label='Inferencia: y = mx + b')

plt.title('Consumo de Memoria RAM en Data Center de IA')
plt.xlabel('Carga de Trabajo (Unidades X)')
plt.ylabel('Memoria RAM (GB)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.4)

plt.savefig('grafico_prediccion.png')