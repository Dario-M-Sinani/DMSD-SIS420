# Guía de Defensa: Métricas de Clasificación Binaria con PyTorch

Este documento proporciona una explicación detallada paso a paso del cuadernillo `metricas_clasificacion_pytorch.ipynb`, diseñada para que puedas defender con solidez cada decisión técnica, el uso de librerías y la interpretación de los resultados ante un jurado o en una presentación académica.

---

## 1. Uso de Librerías (Importaciones)

Al inicio del cuadernillo se importan varias librerías, cada una con un propósito específico dentro del flujo de Machine Learning:

*   **Manipulación de Datos y Matemáticas:**
    *   `numpy (np)`: Se usa para realizar operaciones matemáticas eficientes sobre matrices y vectores. Es la base sobre la que trabajan muchas otras librerías.
    *   `pandas (pd)`: Esencial para la carga del dataset (`.csv`) y la manipulación de datos tabulares (DataFrames), lo que facilita la exploración, limpieza y segmentación de los datos.
*   **Visualización:**
    *   `matplotlib.pyplot (plt)`: Librería principal para crear gráficos de líneas, barras y diagramas 2D (por ejemplo, las curvas de pérdida y curvas ROC).
    *   `seaborn (sns)`: Construida sobre Matplotlib, proporciona una interfaz de alto nivel para dibujar gráficos estadísticos más atractivos visualmente, como la Matriz de Confusión.
*   **Deep Learning (PyTorch):**
    *   `torch`: Framework principal de tensores y operaciones en GPU/CPU.
    *   `torch.nn (nn)`: Contiene las clases base para construir redes neuronales (como `nn.Linear` para las capas y `nn.BCELoss` para la función de pérdida).
    *   `torch.optim (optim)`: Contiene los algoritmos de optimización, como SGD (Stochastic Gradient Descent), para actualizar los pesos del modelo.
    *   `Dataset, DataLoader, TensorDataset`: Utilidades para manejar conjuntos de datos grandes, dividirlos en "lotes" (batches) y barajarlos (shuffle) durante el entrenamiento.
*   **Machine Learning (Scikit-Learn - `sklearn`):**
    *   Se utiliza para tareas clásicas que PyTorch no trae por defecto: partición de datos (`train_test_split`), escalado numérico (`StandardScaler`), codificación de texto a números (`LabelEncoder`) y, crucialmente, el cálculo de todas las métricas de evaluación (Accuracy, F1-Score, Matriz de Confusión, ROC-AUC).

**Dato de defensa:** *¿Por qué usar PyTorch y Sklearn juntos?* PyTorch es excelente para definir arquitecturas y entrenar modelos que aprovechen hardware (GPU) para cálculos masivos y descenso de gradiente. Sin embargo, Sklearn es el estándar de la industria para preprocesamiento y cálculo de métricas tradicionales, por lo que combinarlas es la mejor práctica y demuestra madurez técnica.

---

## 2. Configuración y Reproducibilidad

El bloque de configuración se ve así:
```python
torch.manual_seed(42)
np.random.seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```
*   **Semillas (Seed):** Fijamos la semilla en `42` en Numpy y PyTorch. Es vital para la investigación académica. Esto garantiza que cada vez que ejecutemos el código, la inicialización aleatoria de los pesos y la mezcla de los datos sea **exactamente la misma**, permitiendo reproducir resultados consistentes frente al jurado.
*   **Dispositivo (Device):** Se detecta automáticamente si la computadora tiene una tarjeta gráfica NVIDIA compatible (`cuda`). Si no, usa el procesador (`cpu`). Hace que este cuadernillo corra en cualquier sistema.

---

## 3. Carga, Exploración y Preprocesamiento del Dataset

### Exploración
Trabajamos con el dataset `mening_missing_12.csv`. Durante la exploración validamos cuántas columnas hay, sus tipos de datos y, lo más importante, se visualiza la distribución de la variable objetivo (`Risk_Level`) para entender la proporción de casos críticos vs no críticos.

### Preprocesamiento (ETL)
*   **Manejo de Valores Faltantes (Nulos):** En el contexto médico, limpiar la data es obligatorio.
    *   *Numéricos:* Rellenados con la **mediana** (su ventaja sobre el promedio es que no se ve afectada drásticamente por pacientes con valores atípicos/extremos).
    *   *Categóricos:* Rellenados con la **moda** (el diagnóstico más frecuente de toda la base de datos).
*   **Label Encoding:** Modelos matemáticos en PyTorch no pueden operar sobre palabras ("Male", "Female"). `LabelEncoder` numera cada categoría única (0, 1, 2...).
*   **Binarización de Variable Objetivo:** Para usar "Binary Cross Entropy", la salida debe ser 0 o 1. Transformamos `Risk_Level` asumiendo clase Positiva (`1`) al `'High Risk'`, y a cualquier otra clase se le asigna Negativa (`0`).
*   **Escalado (`StandardScaler`):** Ajustamos todas las variables numéricas a una media de 0 y desviación de 1.
    *   *Defensa:* Las redes neuronales son matemáticamente muy sensibles a la escala. Si la "glucosa" tiene rangos de cientos y los "leucocitos" de miles, y otras cosas son decimales, el entrenamiento tardará exponencialmente más o nunca convergerá hacia el mínimo ideal del gradiente. Normalizar homogeneiza la influencia inicial de todos los atributos.

---

## 4. Construcción del Modelo (Perceptrón en PyTorch)

```python
class Perceptron(nn.Module):
    def __init__(self, input_size: int):
        ...
        self.linear = nn.Linear(input_size, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.linear(x)).squeeze(1)
```

*   **¿Qué es?** Un *Perceptrón* de una sola capa lineal (`nn.Linear`). Recibe el número de columnas (características) y lo comprime hacia 1 único valor contínuo mediante multiplicación matricial de pesos.
*   **Inicialización He/Kaiming:** Una estrategia de PyTorch recomendada teóricamente que evita problemas de activación inicial inestable.
*   **Activación Sigmoide (`torch.sigmoid`):** Esto es fundamental. Envuelve ese número de salida en la función matemática Sigmoide $f(x) = \frac{1}{1+e^{-x}}$, la cual obligatoriamente exprime la respuesta para que se sitúe estrictamente entre $0.0$ y $1.0$. Esto es lo que nosotros interpretamos lógicamente como la **"Probabilidad de ser Alto Riesgo (1)"**.

---

## 5. Entrenamiento del Modelo

Usamos un optimizador y una función de pérdida (Cost Function).
*   **Función de Costo (`BCELoss` / Binary Cross Entropy):** Penaliza duramente al modelo si emite una respuesta incorrecta con mucha confianza. Por ejemplo, predecir con probabilidad del 99% a un paciente sano siendo éste en realidad grave.
*   **Optimizador (`SGD` / Stochastic Gradient Descent):** El encargado de alterar y actualizar los "pesos matemáticos" al final de cada iteración basándose en el vector de derivadas calculadas por PyTorch. Se configuró con **momentum** para superar óptimos locales.
*   **Ciclo de Aprendizaje:** Repetimos por múltiples épocas. En cada pasada, se genera predicción, se compara con error `loss.backward()`, y se modifican los pesos con `optimizer.step()`. Para ello se usa `DataLoader`, procesando datos de a "grupos reducidos" (batches de 32 datos), lo que hace el entrenamiento más escalable en memoria respecto al cálculo masivo con arrays estáticos.

---

## 6. Métricas de Evaluación (Criterio para Medicina)

Esta sección de tu código evalúa utilizando `@torch.no_grad()`, lo cual "apaga" las derivadas innecesarias de memoria ya que solo vamos a predecir sobre datos de Test.

### Accuracy (Exactitud)
*   **Interpretación:** Porcentaje frío de respuestas atinadas: Acertados / Total.
*   **Defensa:** Es engañosa en medicina. Si solo 5% de tus pacientes tienen meningococos y tú dices que ningún paciente los tiene, lograrás un modelo mediocre inútil con un impresionante "95% de exactitud". Por eso evaluamos el resto.

### Matriz de Confusión
Divide aciertos y errores en 4 cuadrantes.
*   **Verdaderos Positivos (TP):** Predijo High Risk ✅ y la realidad era High Risk ✅ (Bien).
*   **Verdaderos Negativos (TN):** Predijo Sano ❌ y era Sano ❌ (Bien).
*   **Falsos Positivos (FP):** Predijo High Risk ✅ pero en realidad era Sano ❌ (Alarma Cautelar / Falsa Alarma - Costo bajo).
*   **Falsos Negativos (FN):** Predijo Sano ❌ pero en realidad era High Risk ✅ (**Peligro inminente, omisión de auxilio - Costo extremo**).

### Precision y Recall (Sensibilidad)
*   **Precision ($TP / TP + FP$):** Si saltó la alarma del sistema, qué tan fiable es la alarma (pocos falsos positivos).
*   **Recall u Sensibilidad ($TP / TP + FN$):** De la cantidad total mundial de pacientes de nivel Rojo (Riesgo total), tu modelo a cuántos logró detectar, y a cuántos dejó escapar (falsos negativos).
*   **Defensa Académica / Médica:** "En un contexto clínico priorizamos un alto índice de *Recall* a costa de la *Precisión*. Es totalmente preferible someter a pruebas adicionales de laboratorios extra a un paciente sano (falso positivo, bajando precision) que mandar erradamente a su casa con paracetamol a un paciente de Meningitis Aguda con Alta Urgencia (falso negativo, bajando recall)".

### F1-Score
Una compensación balanceada. Es la media armónica. Otorga un veredicto definitivo de calidad sin los picos sesgados de un extremo o del otro.

### Curva ROC y AUC (Área Bajo La Curva)
Mientras las anteriores métricas asumían un punto de corte seco del 50% de probabilidad (es decir, el modelo predice 51% y decimos 'enfermo'), el ROC analiza todo.
*   Si nuestro modelo da 40% de "seguridad de que es grave", un Threshold=0.5 no lo diagnosticaría como Riesgo.
*   Para medicina, el código permite bajar este límite. Con Threshold o Umbral = $0.20$, disparamos alarma ante la menor duda, subiendo la sensibilidad al precio de más falsas alarmas. ROC mide el equilibrio a lo largo de todos estos umbrales del 0 al 1.
*   **AUC ($Area \ Under \ Curve$):** Mientras más cercada a $1.0$, mejor discierne entre enfermo y sano tu perceptrón. Un valor de $0.5$ significaría lo mismo que arrojar un dado.

---

## 7. Cierre de la Defensa

"En resumen, en este proyecto se probó una modernización tecnológica, migrando un modelo de un script secuencial simple al poderoso framework estandarizado **PyTorch**, con lo cual queda preparado el terreno de red para aumentar la dimensionalidad en un eventual futuro sin modificar los pilares (dataset/loss). Además se demostró madurez analística al comprender que en la Inteligencia Artificial médica, optimizar la métrica básica *Accuracy* es mala praxis académica si no consideramos el abanico total con la visualización matricial para minimizar el impacto crítico fatal de los casos falsos-negativos (*Recall*)."
