import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def calcularCosto(theta, X, y):
    m = y.size
    h = sigmoid(X.dot(theta))
    # Costo de regresión logística
    J = (1 / m) * np.sum(-y * np.log(h + 1e-15) - (1 - y) * np.log(1 - h + 1e-15))
    return J

def calcularGradiente(theta, X, y):
    m = y.size
    h = sigmoid(X.dot(theta))
    # Derivada para el optimizador
    grad = (1 / m) * (X.T.dot(h - y))
    return grad

df = pd.read_csv('mimic_ready.csv')
X = df[['LOS_dias', 'ICD9_BASE']].values
y = df['HOSPITAL_EXPIRE_FLAG'].values

m_total = len(y)
m_train = int(m_total * 0.75)

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
precision = np.mean(predicciones == y_test) * 100

print(f"Precisión Final en Test: {precision:.2f}%")

plt.figure(figsize=(8, 5))
plt.plot(y_test, 'bo', label='Real (Vivo/Muerto)')
plt.plot(probabilidades, 'rx', label='Probabilidad Predicha')
plt.axhline(0.5, color='gray', linestyle='--')
plt.title('MIMIC: Realidad vs Predicción de Probabilidad')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.hist(probabilidades[y_test == 0], bins=20, alpha=0.7, color='blue', label='Real: Supervivencia (0)')
plt.hist(probabilidades[y_test == 1], bins=20, alpha=0.7, color='red', label='Real: Fallecimiento (1)')
plt.title('Distribución de Probabilidades Predichas por Clase Real')
plt.xlabel('Probabilidad Predicha de Fallecimiento')
plt.ylabel('Frecuencia (Número de pacientes)')
plt.axvline(0.5, color='black', linestyle='--', label='Umbral de Decisión (0.5)')
plt.legend()
plt.show()

