import os
import numpy as np
import pandas as pd
from matplotlib import pyplot

# Configuración para graficar
# %matplotlib inline (Esto causaba el error, solo funciona en Jupyter Notebooks)

# 1. Cargar datos
# Nota: Asegúrate de que la ruta al archivo CSV sea correcta en tu entorno
path = 'BMW sales data (2010-2024).csv' 
df = pd.read_csv(path)

# Seleccionamos las columnas numéricas relevantes
# X: Año y Kilometraje (Mileage_KM)
# y: Precio (Price_USD)
X = df[['Year', 'Mileage_KM']].values
y = df['Price_USD'].values
m = y.size

print(f"Total de registros cargados: {m}")

# Imprimir los primeros 10 datos para verificar
print(f"{'Year':>10s} {'Mileage':>12s} {'Price':>12s}")
print('-' * 36)
for i in range(10):
    print(f"{X[i, 0]:10.0f} {X[i, 1]:12.0f} {y[i]:12.0f}")

# 2. Función de Normalización de características
def featureNormalize(X):
    X_norm = X.copy()
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    X_norm = (X - mu) / sigma
    return X_norm, mu, sigma

# Normalizar X
X_norm, mu, sigma = featureNormalize(X)

# Añadir columna de unos (intercepto) a X_norm
X_ready = np.concatenate([np.ones((m, 1)), X_norm], axis=1)

print('\nMedia calculada:', mu)
print('Desviación estándar calculada:', sigma)

# 3. Funciones de Costo y Descenso por el Gradiente
def computeCostMulti(X, y, theta):
    m = y.shape[0]
    J = (1/(2 * m)) * np.sum(np.square(np.dot(X, theta) - y))
    return J

def gradientDescentMulti(X, y, theta, alpha, num_iters):
    m = y.shape[0]
    theta = theta.copy()
    J_history = []

    for i in range(num_iters):
        # Actualización simultánea de theta
        theta = theta - (alpha / m) * (np.dot(X, theta) - y).dot(X)
        J_history.append(computeCostMulti(X, y, theta))

    return theta, J_history

# 4. Configuración y Entrenamiento
alpha = 0.01  # Puedes probar con 0.1, 0.01, 0.001
num_iters = 1200

# Inicializa theta (3 elementos: intercepto + 2 variables)
theta = np.zeros(3)
theta, J_history = gradientDescentMulti(X_ready, y, theta, alpha, num_iters)

# 5. Graficar la convergencia
pyplot.figure()
pyplot.plot(np.arange(len(J_history)), J_history, lw=2, color='blue')
pyplot.xlabel('Número de iteraciones')
pyplot.ylabel('Costo J')
pyplot.title('Convergencia del Descenso por el Gradiente')

print(f'\ntheta calculado: {theta}')

# 6. Predicción de ejemplo
# Estimar el precio para un BMW del año 2022 con 30,000 KM
año_test = 2022
kilometraje_test = 30000

# Creamos el vector de entrada [1, año, kilometraje]
X_sample = np.array([año_test, kilometraje_test])
# Normalizamos el sample usando la media y desviación del dataset original
X_sample_norm = (X_sample - mu) / sigma
# Agregamos el 1 del intercepto
X_array = np.insert(X_sample_norm, 0, 1)

price = np.dot(X_array, theta)

print(f'\n--- PREDICCIÓN ---')
print(f'Precio estimado para un BMW año {año_test} con {kilometraje_test} KM: ${price:,.2f}')

# Mostrar el gráfico (necesario al no usar %matplotlib inline)
pyplot.show()