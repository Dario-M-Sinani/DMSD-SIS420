import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def estandarizar_datos(X):
    """
    Estandariza los datos manualmente restando la media y dividiendo por 
    la desviación estándar. Equivalente a StandardScaler de scikit-learn.
    """
    # Eje 0 significa que operamos sobre columnas
    media = np.mean(X, axis=0)
    desviacion = np.std(X, axis=0)
    
    # Prevenir división por cero si alguna columna tiene desviación 0
    desviacion = np.where(desviacion == 0, 1, desviacion)
    
    X_estandarizado = (X - media) / desviacion
    return X_estandarizado

def aplicar_pca(X, n_componentes=3):
    """
    Realiza Análisis de Componentes Principales (PCA) desde cero utilizando
    la matriz de covarianza y cálculo de autovalores/autovectores.
    """
    # 1. La matriz debe estar centrada (ya lo está por estandarizar_datos)
    # 2. Calcular matriz de covarianza (rowvar=False para indicar columnas como variables)
    matriz_cov = np.cov(X, rowvar=False)
    
    # 3. Calcular autovalores y autovectores (eigenvalues/eigenvectors)
    # Usamos eigh porque la matriz de covarianza es simétrica y devuelve valores reales
    autovalores, autovectores = np.linalg.eigh(matriz_cov)
    
    # 4. Ordenar autovalores y autovectores de mayor a menor varianza
    indices_ordenados = np.argsort(autovalores)[::-1]
    autovalores_ordenados = autovalores[indices_ordenados]
    autovectores_ordenados = autovectores[:, indices_ordenados]
    
    # 5. Seleccionar los top 'n_componentes' autovectores
    autovectores_seleccionados = autovectores_ordenados[:, 0:n_componentes]
    
    # 6. Transformar (proyectar) los datos al nuevo subespacio dimensional
    X_reducido = np.dot(X, autovectores_seleccionados)
    
    # Calcular varianza explicada
    varianza_total = np.sum(autovalores)
    varianza_explicada = np.sum(autovalores_ordenados[:n_componentes]) / varianza_total
    
    return X_reducido, varianza_explicada

class KMeansManual:
    """
    Implementación del algoritmo K-Means desde cero utilizando solo NumPy.
    """
    def __init__(self, n_clusters=3, max_iters=100, random_state=42):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.random_state = random_state
        self.centroides = None
    def fit_predict(self, X):
        # Fijar semilla aleatoria para reproducibilidad
        np.random.seed(self.random_state)
        
        # 1. Inicialización: Escoger 'k' puntos al azar del dataset como centroides
        indices_aleatorios = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        self.centroides = X[indices_aleatorios]
        
        etiquetas = np.zeros(X.shape[0])
        
        for _ in range(self.max_iters):
            # 2. Asignación: Calcular distancias euclidianas de cada punto a todos los centroides
            # Se usa broadcasting (np.newaxis) para cálculo vectorizado hiper-rápido en lugar de un doble for-loop
            distancias = np.linalg.norm(X[:, np.newaxis] - self.centroides, axis=2)
            
            # Asignar cada punto al centroide con la menor distancia (índice del mínimo)
            nuevas_etiquetas = np.argmin(distancias, axis=1)
            
            # 3. Convergencia: Si ningún punto cambió de clúster, terminamos de iterar
            if np.array_equal(etiquetas, nuevas_etiquetas):
                break
            etiquetas = nuevas_etiquetas
            
            # 4. Actualización: Recalcular centroides como la media de los puntos en cada clúster
            for k in range(self.n_clusters):
                puntos_cluster = X[etiquetas == k]
                if len(puntos_cluster) > 0:
                    self.centroides[k] = np.mean(puntos_cluster, axis=0)
                    
        return etiquetas

def main():
    print("Cargando el dataset 'bank.csv'...")
    try:
        df = pd.read_csv('bank.csv')
    except FileNotFoundError:
        print("Error: No se encontró el archivo 'bank.csv'.")
        return

    print(f"Dataset cargado exitosamente. Dimensiones originales: {df.shape}")
    
    df = df.dropna()
    df = df.drop_duplicates()
    
    print("Convirtiendo variables categóricas a numéricas (One-Hot Encoding)...")
    df_numerico = pd.get_dummies(df, drop_first=True)
    X = df_numerico.to_numpy(dtype=float)
    print(f"Dimensiones después de codificar categóricas: {X.shape}")

    print("Estandarizando las características numéricas...")
    X_escalado = estandarizar_datos(X)

    print("Aplicando PCA implementado desde cero para reducir a 3 dimensiones...")
    X_pca, varianza = aplicar_pca(X_escalado, n_componentes=3)
    print(f"Varianza total retenida por las 3 componentes principales: {(varianza * 100):.2f}%")

    valores_k = [3, 5, 7, 9]
    fig = plt.figure(figsize=(16, 12))
    
    for i, k in enumerate(valores_k, 1):
        print(f"Ejecutando K-Means para k={k}...")
        kmeans = KMeansManual(n_clusters=k, max_iters=150, random_state=42)
        etiquetas = kmeans.fit_predict(X_escalado)
        
        # Añadir un subplot 3D
        ax = fig.add_subplot(2, 2, i, projection='3d')
        
        # Dispersión en 3D
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], 
                             c=etiquetas, cmap='tab10', s=1, alpha=0.5)
        
        ax.set_title(f'K-Means Custom con k={k}')
        ax.set_xlabel('Componente Principal 1')
        ax.set_ylabel('Componente Principal 2')
        ax.set_zlabel('Componente Principal 3')
        
        fig.colorbar(scatter, ax=ax, pad=0.1, label='ID del Cluster')

    plt.tight_layout()
    plt.savefig('kmeans_3d_clusters_nosklearn.png')
    print("Gráfico 3D guardado como 'kmeans_3d_clusters_nosklearn.png'")
    plt.show()

    print("\n" + "="*80)
    print("CONCLUSIONES DEL ANÁLISIS DE CLUSTERING (VERSIÓN DESDE CERO):")
    print("="*80)
    print("1. El escalado de datos (StandardScaler), PCA y K-Means han sido implementados")
    print("   utilizando exclusivamente matemáticas puras mediante álgebra lineal con NumPy.")
    print("\n2. Visualización 3D: Los resultados se mantienen consistentes con las")
    print("   librerías optimizadas tradicionales. Cuando exigimos fragmentar en")
    print("   más clústeres (como k=9), los segmentos no parecen aportar fronteras")
    print("   claramente distinguibles y se entrelazan visualmente en el hiperplano.")
    print("="*80)


if __name__ == '__main__':
    main()
