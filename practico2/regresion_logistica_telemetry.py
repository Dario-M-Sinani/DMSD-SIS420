# ==============================================================================
# Regresión Logística - Predicción de Frenado (Frena o No Frena)
# ==============================================================================
# Este script implementa el algoritmo de Regresión Logística
# para predecir si el conductor de un auto en el juego Forza Motorsport está 
# frenando (1) o no frenando (0) en base a su 'Velocidad' (speed) y 
# las RPM del motor (current_engine_rpm).
# Dataset utilizado: telemetry-rio-5-laps.csv

# ------------------------------------------------------------------------------
# 1. Importación de Librerías
# ------------------------------------------------------------------------------
import os
import numpy as np           # Computación vectorial y matemática
import pandas as pd          # Manipulación y carga del dataset CSV
from matplotlib import pyplot # Trazado de gráficos
from scipy import optimize    # Módulo de optimización (para reemplazar el descenso de gradiente manual)
# %matplotlib inline         # Descomentar si se ejecuta en Jupyter Notebook / Colab

# ------------------------------------------------------------------------------
# 2. Carga y Preparación de Datos
# ------------------------------------------------------------------------------
ruta_dataset = 'telemetry-rio-5-laps.csv'

print("Cargando el dataset de telemetría...")
try:
    data = pd.read_csv(ruta_dataset)
except FileNotFoundError:
    print(f"Error: No se encontró el archivo {ruta_dataset}")
    exit()

print("\n================= ANÁLISIS DE DATOS DEL DATASET =================")
print(f"1. Tamaño del dataset original: {data.shape[0]} filas y {data.shape[1]} columnas.")
print("\n2. Tipos de datos en el dataset (resumen):")
print(data.dtypes.value_counts())

# Columnas que vamos a utilizar vs las que anularemos
columnas_seleccionadas = ['speed', 'current_engine_rpm', 'brake']
columnas_anuladas = [col for col in data.columns if col not in columnas_seleccionadas]

print(f"\n3. Selección de características:")
print(f"   -> Se seleccionarán {len(columnas_seleccionadas)} columnas relevantes para el modelo:")
print(f"      {columnas_seleccionadas}")
print(f"   -> Se anularán/descartarán las {len(columnas_anuladas)} columnas restantes.")
print(f"      (Ejemplo de anuladas: {columnas_anuladas[:5]}...)")

# El dataset original tiene 70+ columnas. Para este ejercicio de clasificación bi-dimensional, 
# seleccionaremos dos características numéricas relevantes (X):
#   - speed: Velocidad del vehículo
#   - current_engine_rpm: Revoluciones por minuto del motor
# Como variable objetivo (y), predeciremos si el conductor está frenando o no.
# La columna 'brake' del dataset tiene valores de 0 a 255. 
# Crearemos nuestra variable binaria 'y': 1 si 'brake' > 0 (Frenando), 0 si 'brake' == 0 (No Frenando)

# Filtramos filas que tengan valores nulos en estas columnas por seguridad
filas_antes = data.shape[0]
data = data.dropna(subset=columnas_seleccionadas)
filas_despues = data.shape[0]

print(f"\n4. Limpieza de datos (nulos):")
print(f"   -> Se eliminaron {filas_antes - filas_despues} filas que contenían valores nulos en las columnas seleccionadas.")
print("=================================================================\n")

# Para acelerar el entrenamiento y no saturar la gráfica, podemos tomar una muestra
# aleatoria de 1000 registros (opcional, pero recomendado para datasets masivos como este)
data_sample = data.sample(n=1000, random_state=42)

# Extraemos X (Velocidad, RPM) e y (0 o 1) de la muestra
X = data_sample[['speed', 'current_engine_rpm']].values
y = (data_sample['brake'] > 0).astype(int).values

m = y.size # Cantidad de ejemplos

print(f"Total de datos cargados para entrenamiento (muestra): {m}")
print(f"Desglose de 'y' (1 = Frenando, 0 = No Frenando): \n{pd.Series(y).value_counts()}")

# ------------------------------------------------------------------------------
# 3. Visualización de los Datos
# ------------------------------------------------------------------------------
def plotData(X, y):
    """
    Grafica los puntos de datos X y y en una figura.
    Utiliza '+' para los ejemplos positivos (Frenando) 
    y 'o' amarillos para los negativos (No Frenando).
    """
    fig = pyplot.figure(figsize=(8, 6))

    # Encuentra los índices de los ejemplos positivos (y==1) y negativos (y==0)
    pos = (y == 1)
    neg = (y == 0)

    # Grafica los ejemplos
    pyplot.plot(X[pos, 0], X[pos, 1], 'k+', lw=2, ms=8, label='Frenando (1)')
    pyplot.plot(X[neg, 0], X[neg, 1], 'ko', mfc='y', ms=8, mec='k', mew=1, label='No Frenando (0)')
    
    pyplot.xlabel('Velocidad (speed)')
    pyplot.ylabel('RPM del motor (current_engine_rpm)')
    pyplot.title('Distribución de Frenado vs (Velocidad y RPM)')
    pyplot.legend()
    pyplot.grid(True)

print("\nGenerando gráfica de distribución de datos...")
plotData(X, y)
# pyplot.show() # Descomentar para pausar y ver la gráfica aquí 

# ------------------------------------------------------------------------------
# 4. Implementación - Función Sigmoide
# ------------------------------------------------------------------------------
# En clasificación, necesitamos que las predicciones estén entre 0 y 1 (probabilidades).
# Esto lo logramos pasando nuestro modelo lineal a través de la función Sigmoide.

def sigmoid(z):
    """
    Calcula la sigmoide de una entrada z (z puede ser un escalar, vector o matriz).
    Retorna un valor entre 0 y 1.
    """
    z = np.array(z)
    g = 1 / (1 + np.exp(-z))
    return g

# Prueba rápida de la sigmoide (debería dar 0.5 si z=0)
print(f"\nPrueba Función Sigmoide: sigmoid(0) = {sigmoid(0)}")

# ------------------------------------------------------------------------------
# 5. Función de Costo y Gradiente
# ------------------------------------------------------------------------------
# Agregamos la columna de unos a X para el término de intercepción (theta_0)
m, n = X.shape
X_ready = np.concatenate([np.ones((m, 1)), X], axis=1)

def costFunction(theta, X, y):
    """
    Calcula el costo y el gradiente de la regresión logística.
    """
    m = y.size
    
    # 1. Calculamos la hipótesis usando la función sigmoide: h_theta(X) = g(X * theta)
    h = sigmoid(X.dot(theta))
    
    # Prevenimos log(0) que daría NaN sumando un valor minúsculo epsilon
    epsilon = 1e-15
    
    # 2. Calculamos el Costo (J) usando la fórmula logarítmica de entropía cruzada
    J = (1 / m) * np.sum(-y * np.log(h + epsilon) - (1 - y) * np.log(1 - h + epsilon))
    
    # 3. Calculamos el Gradiente (derivadas parciales)
    grad = (1 / m) * (h - y).dot(X)
    
    return J, grad

# Probamos la función con thetas iniciales en 0
initial_theta = np.zeros(X_ready.shape[1])
cost, grad = costFunction(initial_theta, X_ready, y)
print(f'Costo con theta inicial (zeros): {cost:.3f}')
print(f'Gradiente en theta inicial: {grad}')

# ------------------------------------------------------------------------------
# 6. Optimización de Parámetros usando scipy.optimize
# ------------------------------------------------------------------------------
# En lugar de hacer el bucle de descenso de gradiente manual, usamos una librería 
# experta que lo hará más rápido y encontrará thetas óptimos automáticamente.

historial_costo = []
historial_p = [] # Guardaremos las predicciones 'p' en cada iteración

def callback_optimizacion(theta_actual):
    """
    Función que se llama en cada iteración de la optimización
    para guardar el costo y las predicciones históricas 'p'.
    """
    # Guardar costo
    costo_actual, _ = costFunction(theta_actual, X_ready, y)
    historial_costo.append(costo_actual)
    
    # Guardar predicciones 'p' históricas
    probabilidades = sigmoid(X_ready.dot(theta_actual))
    p = np.round(probabilidades)
    historial_p.append(p)

print('\nOptimizando parámetros con scipy.optimize.minimize...')
options= {'maxiter': 1000} # Máximo de iteraciones permitidas

# Llamamos al optimizador (usando el algoritmo Truncated Newton 'TNC')
res = optimize.minimize(costFunction,      # Nuestra función a minimizar
                        initial_theta,     # Thetas iniciales (ceros)
                        (X_ready, y),      # Los datos extra que recibe costFunction
                        jac=True,          # Le indicamos que costFunction también devuelve el gradiente
                        method='TNC',      # Algoritmo de optimización escogido
                        callback=callback_optimizacion, # Pasamos la función callback
                        options=options)

# Obtenemos el costo minimizado y el array de Thetas optimizados
cost = res.fun
theta_optimizado = res.x

print(f'\nCosto mínimo encontrado: {cost:.3f}')
print(f'Theta optimizados: {theta_optimizado}')

# ------------------------------------------------------------------------------
# 7. Evaluación y Predicción Final del Modelo
# ------------------------------------------------------------------------------
def predict(theta, X):
    """
    Predice si la etiqueta es 0 (No frena) o 1 (Frena) para casos X,
    usando un umbral de decisión del 50% (0.5).
    """
    # Calculamos las probabilidades
    probabilidades = sigmoid(X.dot(theta))
    # Redondeamos: >= 0.5 será 1, < 0.5 será 0
    p = np.round(probabilidades)
    return p

# A) Precisión General del Modelo
# Comparamos nuestras predicciones sobre X_ready contra los valores reales 'y'
predicciones = predict(theta_optimizado, X_ready)
precision = np.mean(predicciones == y) * 100
print(f'\nPrecisión del modelo en el set de entrenamiento: {precision:.2f}%')

# B) Predicción de un Caso Nuevo
# Supongamos que en un momento de la carrera, tenemos:
velocidad_test = 25.0 # M/s (aprox 90 km/h)
rpm_test = 8000.0     # RPM altas
X_test = np.array([1, velocidad_test, rpm_test]) # Agregamos el 1 de intercepción manual

prob_frenado = sigmoid(np.dot(X_test, theta_optimizado))
pred_frenado = predict(theta_optimizado, X_test.reshape(1, -1))[0] # Reshape por ser 1 sola fila

print('\n--------------------- PREDICCIÓN ---------------------')
print(f'Datos del coche en carrera:')
print(f' - Velocidad:   {velocidad_test} m/s')
print(f' - RPMs Motor:  {rpm_test}')
print(f'>> Probabilidad de que el conductor frene: {prob_frenado * 100:.2f}%')
print(f'>> Predicción del sistema: {"FRENANDO (1)" if pred_frenado == 1 else "ACELERANDO o NEUTRO (0)"}')
print('------------------------------------------------------')

# Mostrar gráfica de datos junto con la frontera de decisión
print('\nGraficando frontera de decisión...')
# Seleccionar los valores mínimo y máximo de la característica X1 (Velocidad) para trazar una línea
plot_x = np.array([np.min(X[:, 0]) - 2, np.max(X[:, 0]) + 2])
# Calcular los correspondientes valores de X2 (RPM) usando la ecuación de la frontera (theta0 + theta1*X1 + theta2*X2 = 0)
plot_y = (-1. / theta_optimizado[2]) * (theta_optimizado[1] * plot_x + theta_optimizado[0])

# Graficar la línea de la frontera de decisión en color azul
pyplot.plot(plot_x, plot_y, 'b-', label='Frontera de Decisión')
pyplot.legend()

# Mostrar la primera gráfica
pyplot.show()

# ------------------------------------------------------------------------------
# 8. Gráficas de Aprendizaje (Costo y Precisión vs Iteraciones)
# ------------------------------------------------------------------------------
print('\nGenerando gráficas de la curva de aprendizaje (Costo y Precisión)...')
iteraciones = np.arange(1, len(historial_costo) + 1)

# Calculamos la precisión histórica basándonos en el historial de 'p'
historial_precision = [np.mean(p == y) * 100 for p in historial_p]

fig, (ax1, ax2) = pyplot.subplots(1, 2, figsize=(14, 5))

# Gráfico 1: Costo (J) vs Iteraciones
ax1.plot(iteraciones, historial_costo, 'b-', linewidth=2)
ax1.set_xlabel('Iteraciones')
ax1.set_ylabel('Costo (J)')
ax1.set_title('Historial de Costo durante la Optimización')
ax1.grid(True)

# Gráfico 2: Precisión vs Iteraciones
ax2.plot(iteraciones, historial_precision, 'g-', linewidth=2)
ax2.set_xlabel('Iteraciones')
ax2.set_ylabel('Precisión (%)')
ax2.set_title('Historial de Precisión durante la Optimización')
ax2.grid(True)

pyplot.tight_layout()

# Mostrar gráficas conjuntas
pyplot.show()
