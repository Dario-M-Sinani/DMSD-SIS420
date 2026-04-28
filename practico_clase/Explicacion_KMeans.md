# Análisis de Clustering con K-Means: Proyecto de Segmentación de Clientes

Este documento detalla el análisis no supervisado (clustering) realizado sobre el conjunto de datos de marketing bancario (`bank.csv`). El objetivo del proyecto fue segmentar a los clientes en diferentes grupos (clústeres) utilizando el algoritmo K-Means, y reducir sus dimensiones para una visualización en 3D. 

El proyecto se basa en una **Implementación desde Cero**, programando todas las matemáticas (Estandarización, PCA, K-Means) puramente con álgebra lineal mediante `NumPy` y herramientas de manipulación de datos con `Pandas`.

---

## 📊 Descripción Detallada del Dataset (`bank.csv`)

El archivo `bank.csv` utilizado contiene datos de campañas de marketing directo (llamadas telefónicas) de una institución bancaria portuguesa.

### Estructura de los Datos
* **Registros (Filas)**: Aproximadamente 11,162 clientes.
* **Características (Columnas)**: 17 variables que combinan datos demográficos, financieros e históricos de la campaña.

### Variables Principales
1. **Datos Demográficos**:
   - `age`: Edad del cliente (Numérica).
   - `job`: Tipo de trabajo (Categórica: admin., blue-collar, entrepreneur, management, etc.).
   - `marital`: Estado civil (Categórica: married, divorced, single).
   - `education`: Nivel educativo (Categórica: primary, secondary, tertiary, unknown).
2. **Datos Financieros/Bancarios**:
   - `default`: ¿Tiene crédito en mora? (Categórica binaria: yes, no).
   - `balance`: Saldo medio anual en euros (Numérica).
   - `housing`: ¿Tiene préstamo hipotecario? (Categórica binaria: yes, no).
   - `loan`: ¿Tiene un préstamo personal? (Categórica binaria: yes, no).
3. **Datos de Contacto de Campaña**:
   - `contact`: Medio de comunicación (Categórica: cellular, telephone, unknown).
   - `day`, `month`: Día y mes del último contacto (Numéricas/Categóricas).
   - `duration`: Duración del último contacto en segundos (Numérica).
4. **Datos Históricos**:
   - `campaign`: Número de contactos realizados en la campaña actual.
   - `pdays`: Días que pasaron desde que el cliente fue contactado en una campaña anterior.
   - `previous`: Número de contactos previos a esta campaña.
   - `poutcome`: Resultado de la campaña de marketing anterior.
5. **Variable Objetivo (En contextos supervisados)**:
   - `deposit`: ¿Suscribió un depósito a plazo fijo? (yes, no). *Nota: Al ser un algoritmo de clustering no supervisado (K-Means), esta etiqueta no guía el modelo, sino que sirve para ver cómo se agrupan las personas naturalmente.*

---

## 🛠️ ¿Qué se hizo en el Código? (Metodología)

Para aplicar correctamente K-Means, se desarrolló un pipeline de datos estructurado en ambos scripts:

### 1. Limpieza y Manejo de Variables Categóricas (One-Hot Encoding)
K-Means utiliza la distancia euclidiana (la línea recta entre puntos) para agrupar. Las distancias no pueden calcularse sobre palabras como "married" o "single". 
* **Acción**: Se aplicó la función `pd.get_dummies(drop_first=True)` para transformar las columnas de texto en columnas numéricas binarias (0 y 1). El parámetro `drop_first=True` previene la colinealidad (evita crear columnas redundantes).

### 2. Estandarización de los Datos
Las variables tienen escalas muy distintas (ej: `age` va de 20 a 90, pero `balance` va de -1000 a 80000). Sin ajustar esto, la variable `balance` dominaría por completo el algoritmo y arruinaría los clústeres.
* **Acción**: Se calculó matemáticamente la estandarización aplicando la fórmula `(X - Media) / Desviación_Estándar`.

### 3. Reducción de Dimensionalidad (PCA)
Tras el *One-Hot Encoding*, el dataset pasó a tener más de 40 columnas (dimensiones). Los humanos no podemos visualizar gráficos en 40D.
* **Acción**: Se utilizó el Análisis de Componentes Principales (PCA). Se calculó la matriz de covarianza y se extrajeron sus autovalores y autovectores, quedándonos solo con los 3 ejes que acumulan la mayor cantidad de información y varianza del dataset original.
* **Resultado**: Transformación a coordenadas 3D (PC1, PC2, PC3).

### 4. Algoritmo K-Means y Visualización 3D interactiva/estática
Finalmente, se programó la asignación de clientes a diferentes $k$-grupos:
* **Acción**: Se probaron los valores $k \in \{3, 5, 7, 9\}$. Los centroides se inicializaron aleatoriamente sobre puntos del dataset. Se calcularon las distancias utilizando *broadcasting* ultrarrápido con Numpy, y se iteró recalculando el promedio geométrico (la media) hasta la convergencia.
* **Resultado**: Se crearon sub-gráficos en un solo plano 3D para cada modelo. Las gráficas resultantes fueron guardadas en la imagen `kmeans_3d_clusters_nosklearn.png`.

---

## 🎯 Conclusión y Elección del Mejor Número de Clústeres

Implementar K-Means desde cero permitió entender que debajo de la "caja negra" de los algoritmos tradicionales, el agrupamiento es puramente una minimización de distancias euclidianas asistida por álgebra matricial.

### 🎲 Explicación de los Gráficos 3D ($k=3, 5, 7, 9$)
Las gráficas en `kmeans_3d_clusters_nosklearn.png` muestran a los clientes proyectados en las 3 Componentes Principales más importantes (PC1, PC2 y PC3) halladas mediante PCA:
- **Para $k=3$ y $k=5$**: Se observan agrupaciones (nubes de puntos del mismo color) bien aglomeradas y diferenciadas en zonas específicas del espacio 3D. Las fronteras son lógicas.
- **Para $k=7$ y $k=9$**: Los colores comienzan a mezclarse excesivamente ("sobre-segmentación"). Un mismo grupo natural de clientes está siendo partido por la mitad arbitrariamente solo para forzar el número de clústeres exigido por el modelo.

### 🏆 ¿Cuál es el mejor grupo/valor de k?
Según la lógica de negocio y la visualización, **el mejor valor estimado ronda $k=4$ o $k=5$**. 
**Justificación**: En el gráfico 3D para $k=5$, vemos que los clientes se agrupan en cinco "nubes" distintas, permitiéndole al banco crear perfiles de marketing específicos sin caer en la fragmentación irracional y caótica que se observa en las subgráficas de $k=7$ o $k=9$.
