import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def calcularCosto(theta, X, y):
    m = y.size
    h = sigmoid(X.dot(theta))
    # Se agrega un valor minúsculo (1e-15) para evitar el error log(0)
    J = (1 / m) * np.sum(-y * np.log(h + 1e-15) - (1 - y) * np.log(1 - h + 1e-15))
    return J

def calcularGradiente(theta, X, y):
    m = y.size
    h = sigmoid(X.dot(theta))
    return (1 / m) * (X.T.dot(h - y))

df = pd.read_csv('breastcancer_ready.csv')

# Preparamos X (características) e y (clase: 2 benigno -> 0, 4 maligno -> 1)
X = df.drop('Class', axis=1).values
y = df['Class'].map({2: 0, 4: 1}).values

m_train = int(len(y) * 0.75)
X_train, y_train = X[:m_train], y[:m_train]
X_test, y_test = X[m_train:], y[m_train:]

mu, sigma = X_train.mean(0), X_train.std(0)
X_train = (X_train - mu) / sigma
X_test = (X_test - mu) / sigma

X_train = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
X_test = np.hstack([np.ones((X_test.shape[0], 1)), X_test])

initial_theta = np.zeros(X_train.shape[1])
res = optimize.minimize(fun=calcularCosto, 
                        x0=initial_theta, 
                        args=(X_train, y_train), 
                        jac=calcularGradiente, 
                        method='TNC')

theta_final = res.x

probabilidades = sigmoid(X_test.dot(theta_final))
predicciones = (probabilidades >= 0.5).astype(int)
exactitud = np.mean(predicciones == y_test) * 100

print(f"Exactitud final en el conjunto de prueba: {exactitud:.2f}%")

plt.figure(figsize=(10, 6))
plt.scatter(range(len(y_test)), y_test, color='blue', label='Real', alpha=0.5)
plt.scatter(range(len(y_test)), probabilidades, color='red', marker='x', label='Probabilidad Predicha', alpha=0.5)
plt.axhline(0.5, color='black', linestyle='--')
plt.title('Clasificación de Cáncer: Realidad vs Predicción')
plt.ylabel('Malignidad (0=No, 1=Si)')
plt.legend()
plt.show()

# --- GRÁFICA 2: Histograma de Probabilidades Predichas ---
plt.figure(figsize=(10, 6))
plt.hist(probabilidades[y_test == 0], bins=20, alpha=0.7, color='blue', label='Real: Benigno (0)')
plt.hist(probabilidades[y_test == 1], bins=20, alpha=0.7, color='red', label='Real: Maligno (1)')
plt.title('Distribución de Probabilidades Predichas por Clase Real')
plt.xlabel('Probabilidad Predicha de Malignidad')
plt.ylabel('Frecuencia (Número de muestras)')
plt.axvline(0.5, color='black', linestyle='--', label='Umbral de Decisión (0.5)')
plt.legend()
plt.show()

