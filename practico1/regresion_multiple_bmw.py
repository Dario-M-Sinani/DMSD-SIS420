# ==============================================================================
# Regresión Lineal Múltiple - Predicción de Precios de Vehículos BMW
# ==============================================================================
# Este script implementa el algoritmo de Regresión Lineal con Múltiples Variables
# desde cero (usando descenso por el gradiente) para predecir el precio de un
# vehículo BMW en base a múltiples características (Año, Tamaño del Motor, y Kilometraje).

# ------------------------------------------------------------------------------
# 1. Importación de Librerías
# ------------------------------------------------------------------------------
import os
import numpy as np           # Computación vectorial y operaciones matemáticas
import pandas as pd          # Para manipulación y carga del dataset CSV fácilmente
from matplotlib import pyplot # Librería para trazado de gráficos
# %matplotlib inline         # Descomentar si se ejecuta en Jupyter Notebook / Colab para embeber gráficas

# ------------------------------------------------------------------------------
# 2. Carga de Datos
# ------------------------------------------------------------------------------
# Al visualizar los datos originales, tenemos muchas columnas, algunas con texto.
# Para regresión lineal múltiple estándar, necesitamos características numéricas.
# El archivo .csv contiene, entre otras, las siguientes columnas:
# Model, Year, Region, Color, Fuel_Type, Transmission, Engine_Size, Mileage_KM, Price_USD...
#
# Usaremos Pandas para cargar el CSV porque maneja fácilmente la cabecera y el texto,
# y luego extraeremos solo las columnas numéricas que nos interesan (nuestras variables X).

ruta_dataset = 'BMW sales data (2010-2024).csv'

# Intentamos cargar los datos
print("Cargando el dataset...")
try:
    data = pd.read_csv(ruta_dataset)
except FileNotFoundError:
    # Si estás en Google Colab, podrías necesitar cambiar la ruta, por ejemplo:
    # ruta_dataset = '/content/gdrive/MyDrive/Colab Notebooks/.../BMW sales data (2010-2024).csv'
    print(f"Error: No se encontro el archivo en la ruta {ruta_dataset}")
    exit()

# Extraemos nuestras variables (X) y nuestro objetivo (y)
# X (Características):
#   - Year: Año del vehículo
#   - Engine_Size: Tamaño del motor
#   - Mileage_KM: Kilometraje recorrido
# y (Objetivo):
#   - Price_USD: Precio del vehículo en dólares
X = data[['Year', 'Engine_Size_L', 'Mileage_KM']].values
y = data['Price_USD'].values
m = y.size # Cantidad de ejemplos de entrenamiento

print(f"Total de datos de entrenamiento (m): {m}")
print('\nMostrando los primeros 10 registros extraidos:')
print('{:>8s}{:>14s}{:>14s}{:>14s}'.format('Año', 'Motor_L', 'Kilometraje', 'Precio_USD'))
print('-' * 52)
for i in range(10):
    print('{:8.0f}{:14.1f}{:14.0f}{:14.0f}'.format(X[i, 0], X[i, 1], X[i, 2], y[i]))

# ------------------------------------------------------------------------------
# 3. Normalización de Características
# ------------------------------------------------------------------------------
# Al observar los datos, notamos que el 'Año' ronda los 2016, el 'Motor_L' es un 
# valor pequeño (ej. 3.5), y el 'Kilometraje' es un valor muy grande (ej. 150000).
# Las características tienen DIFERENTES MAGNITUDES. 
# Si no las normalizamos, el Descenso por el Gradiente tardará mucho en converger 
# o incluso divergir. Transformamos cada valor a una escala similar (-1 a 1).

def featureNormalize(X):
    """
    Normaliza las características en X.
    Devuelve la versión normalizada X_norm, junto con la media (mu) 
    y la desviación estándar (sigma) calculadas.
    """
    X_norm = X.copy()
    
    # Calculamos la media (promedio) de cada columna (axis=0)
    mu = np.mean(X, axis=0)
    
    # Calculamos la desviación estándar de cada columna (axis=0)
    # La desviación estándar mide cuánta variación hay respecto a la media.
    sigma = np.std(X, axis=0)
    
    # Aplicamos la fórmula de normalización: (Valor - Media) / Desviación_Estándar
    X_norm = (X - mu) / sigma
    
    return X_norm, mu, sigma

print('\nNormalizando características...')
X_norm, mu, sigma = featureNormalize(X)

print('Media de cada columna (Año, Motor, Kilometraje):', mu)
print('Desviación Estándar de cada columna:', sigma)

# Después de normalizar, AGREGAMOS LA COLUMNA DE UNOS (Intercepto x_0 = 1)
# Esto es necesario para la ecuación matricial de la recta: y = theta_0*1 + theta_1*x_1 + ...
X_ready = np.concatenate([np.ones((m, 1)), X_norm], axis=1)

# ------------------------------------------------------------------------------
# 4. Funciones de Costo y Descenso por el Gradiente
# ------------------------------------------------------------------------------

def computeCostMulti(X, y, theta):
    """
    Calcula el costo J para regresión lineal con múltiples variables.
    Es la medida de qué tan equivocadas están nuestras predicciones actuales.
    """
    m = y.shape[0]
    # np.dot(X, theta) son las predicciones (hipótesis).
    # Se resta 'y' para obtener los errores. Se eleva al cuadrado y se suma.
    J = (1/(2 * m)) * np.sum(np.square(np.dot(X, theta) - y))
    return J

def gradientDescentMulti(X, y, theta, alpha, num_iters):
    """
    Ejecuta el algoritmo de Descenso por el Gradiente para aprender theta.
    Iterativamente ajusta los parámetros theta para minimizar el costo J.
    """
    m = y.shape[0]
    theta = theta.copy()
    J_history = [] # Guardará el costo en cada iteración para poder graficarlo

    for i in range(num_iters):
        # Fórmula vectorizada del Descenso por el Gradiente:
        # theta = theta - alpha * (1/m) * sum((h(x) - y) * x)
        errores = np.dot(X, theta) - y
        theta = theta - (alpha / m) * (errores.dot(X))
        
        # Guardamos el costo de esta iteración
        J_history.append(computeCostMulti(X, y, theta))

    return theta, J_history

# ------------------------------------------------------------------------------
# 5. Entrenamiento del Modelo
# ------------------------------------------------------------------------------
# Configuramos la "tasa de aprendizaje" (alpha) y la cantidad de iteraciones (pasos)
alpha = 0.03   # Tamaño del paso que damos hacia el mínimo. (Puedes probar 0.1, 0.01)
num_iters = 500 # Número de veces que repetiremos la actualización

# Inicializamos todos los Thetas en cero.
# Tenemos 4 thetas: theta_0 (intercepto), theta_1 (Año), theta_2 (Motor), theta_3 (Km)
theta_inicial = np.zeros(4) 

print('\nEjecutando el Descenso por el Gradiente...')
theta_optimizado, J_history = gradientDescentMulti(X_ready, y, theta_inicial, alpha, num_iters)

print(f'\nParámetros Theta optimizados encontrados por el gradiente:')
print(f'Theta 0 (Intercepto) : {theta_optimizado[0]:.2f}')
print(f'Theta 1 (Año)        : {theta_optimizado[1]:.2f}')
print(f'Theta 2 (Motor)      : {theta_optimizado[2]:.2f}')
print(f'Theta 3 (Kilometraje): {theta_optimizado[3]:.2f}')

# ------------------------------------------------------------------------------
# 6. Graficar la Convergencia
# ------------------------------------------------------------------------------
# Esto nos ayuda a "visualizar" cómo el modelo fue aprendiendo. 
# La curva debería ir disminuyendo y luego estabilizarse (converger).
pyplot.figure(figsize=(8, 5))
pyplot.plot(np.arange(len(J_history)), J_history, lw=2, color='red')
pyplot.xlabel('Número de Iteraciones')
pyplot.ylabel('Costo J')
pyplot.title('Comportamiento del Descenso por el Gradiente')
pyplot.grid(True)
# Descomentar si no usas %matplotlib inline en jupyter/colab
pyplot.show() 

# ------------------------------------------------------------------------------
# 7. Predicción de un Caso Nuevo
# ------------------------------------------------------------------------------
# Explicación Final: ¿Para qué sirve todo esto? ¡Para PREDECIR!
# Digamos que entra un nuevo auto a la concesionaria. 
# No sabemos a qué precio venderlo. Ingresamos sus datos:

año_test = 2021
motor_test = 3.0
kilometraje_test = 45000

# IMPORTANTE: Debemos aplicarle EXACTAMENTE LA MISMA NORMALIZACIÓN que usamos en el entrenamiento.
X_test_norn = np.array([año_test, motor_test, kilometraje_test])
X_test_norn = (X_test_norn - mu) / sigma

# Le agregamos el 1 inicial (Intercepto)
X_test_listo = np.insert(X_test_norn, 0, 1)

# Aplicamos la ecuación de regresión: Precio = Theta * X
precio_estimado = np.dot(X_test_listo, theta_optimizado)

print('\n--------------------- PREDICCIÓN ---------------------')
print(f'Vehículo a predecir:')
print(f' - Año: {año_test}')
print(f' - Tamaño Motor (L): {motor_test}')
print(f' - Kilometraje recorrido: {kilometraje_test} KM')
print(f'>> PRECIO DE VENTA ESTIMADO: ${precio_estimado:,.2f} USD')
print('------------------------------------------------------')
