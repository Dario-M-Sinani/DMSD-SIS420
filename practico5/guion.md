# Guía Detallada: Transición a PyTorch y Redes Neuronales (Práctico 5)

## 1. Introducción: ¿Qué es PyTorch y por qué lo usamos?

**PyTorch** es una de las bibliotecas de aprendizaje automático (Machine Learning) y aprendizaje profundo (Deep Learning) más populares y potentes desarrolladas por el equipo de investigación de Inteligencia Artificial de Facebook (Meta).

**¿Por qué estamos haciendo esta transición?**
En los prácticos anteriores (`LAB2` y `LAB3`), implementamos modelos matemáticos "a mano" o utilizando bibliotecas de optimización clásicas como `scipy.optimize`. Aunque esto es fundamental para entender las matemáticas detrás del aprendizaje automático, tiene limitaciones severas a gran escala:
1. **Escalabilidad:** Calcular derivadas parciales y programar el gradiente descendente manualmente es insostenible en modelos con miles o millones de parámetros.
2. **Aceleración por Hardware:** PyTorch permite ejecutar cálculos matriciales masivos en tarjetas de video (GPUs) casi sin esfuerzo, reduciendo drásticamente los tiempos de entrenamiento.
3. **Diferenciación Automática (Autograd):** PyTorch calcula automáticamente las derivadas matemáticas complejas para ajustar los modelos (Backpropagation), facilitando la creación de Arquitecturas de Redes Neuronales.

## 2. Contexto de los Datasets

### Dataset 1: `BMW sales data (2010-2024).csv` (Utilizado en regesiones)
- **Contexto:** Es un conjunto de datos que contiene información histórica de 50,000 vehículos BMW vendidos, incluyendo características como el año de fabricación, el tamaño del motor en litros, el kilometraje y finalmente su precio en dólares.
- **Objetivo:** Predecir el **precio** de un vehículo a partir de sus características. Al tratar de predecir un valor numérico continuo, estamos frente a un problema de **Regresión**.

### Dataset 2: `telemetry-rio-5-laps.csv` (Utilizado en clasificación)
- **Contexto:** Son datos de telemetría de alta frecuencia extraídos del simulador de carreras Forza Motorsport. Contiene mediciones precisas del estado del coche (revoluciones por minuto (RPM), velocidad, ángulos, aceleración, etc.) tomadas durante 5 vueltas a un circuito de carreras (Rio).
- **Objetivo:** Predecir si el conductor está **frenando (1)** o **no frenando (0)** basándose en la velocidad y el régimen del motor (RPM). Al intentar predecir una categoría entre dos opciones discretas posibles, estamos ante un problema de **Clasificación Binaria**.

## 3. ¿Por qué usar un Perceptrón Multicapa (MLP) sobre modelos clásicos?

En ejercicios anteriores, utilizamos **Regresión Lineal** y **Regresión Logística**. Estos modelos asumen que existe una relación estrictamente *lineal* (como un plano inclinado en 3D o una línea recta en 2D) entre los datos de entrada y el resultado esperado.

Sin embargo, el mundo real suele ser más complejo.
Un **Perceptrón Multicapa (MLP)** es una **Red Neuronal Artificial Básica**. Consiste en múltiples "capas" de neuronas matemáticas:
1. **Capa de Entrada:** Recibe los datos originales.
2. **Capas Ocultas:** Toman los datos de entrada y aplican una función matemática no lineal (en nuestro caso, la función **ReLU**). Esto permite a la red modelar e inferir relaciones complejas, interacciones ocultas y excepciones a la regla en los datos.
3. **Capa de Salida:** Entrega la predicción consolidada final.

**Ejemplo Práctico:** Un coche antiguo con pocos kilómetros podría ser considerado un artículo de colección clásico y costar desproporcionadamente más, rompiendo la regla de "más viejo = más barato". Un MLP es capaz de entender este tipo de excepciones gracias a la no-linealidad.

## 4. Análisis Detallado de las Implementaciones (Notebooks)

### A. Ejecución: `bmw_perceptron_pytorch.ipynb` (Regresión de Precios)

- **El Problema Computacional:** Estimar un valor numérico continuo (Precio en USD) a partir de un arreglo de Características (Año, Tamaño del Motor, Kilometraje).
- **Arquitectura de la Red:**
  - **Entrada:** 3 neuronas (una para cada variable introducida).
  - **Capa Oculta:** 16 neuronas con función de activación **ReLU**. (Añade la posibilidad de identificar no-linealidades).
  - **Salida:** 1 neurona lineal pura. No se utiliza ninguna función de activación en el tramo de salida porque el precio predecido puede adoptar cualquier rango de valor numérico.
- **Optimizador y Función de Pérdida (Loss):**
  - **Función de Pérdida:** Se usa **MSELoss** (Error Cuadrático Medio, Mean Squared Error). Es la forma convencional de medir el fracaso en regresión. Castiga cuadráticamente (en gran proporción) cuando la predicción dista mucho del precio real originario.
  - **Optimizador:** Se emplea **Adam**. Un algoritmo dinámico avanzado que adapta automáticamente la "tasa de aprendizaje" (el tamaño de cada paso de corrección en el entramiento). Evita estancamientos.
- **Mecánica de Entrenamiento:** Utiliza el clásico ciclo de aprendizaje de PyTorch: 
  1. *Forward*: Pase frontal, calcula predicción.
  2. *Loss*: Computa la magnitud del error.
  3. *Backward*: Usa cálculo derivado automático para medir cómo cada peso afectaba al error.
  4. *Step*: Corrige sutilmente los pesos para hacerlo mejor la próxima vez.

### B. Ejecución: `frenado_mlp_pytorch.ipynb` (Clasificación de Frenado)

- **El Problema Computacional:** Clasificación Binaria. Separar dos estados únicos, si un coche frena (estado 1) o no frena (estado 0), usando Velocidad y RPM del motor de la telemetría dinámica.
- **Mejoras Metodológicas respecto a prácticas anteriores:** Se reemplaza por completo la evaluación global, para implementar un subsistema `Dataset` + `DataLoader` provisto nativamente por PyTorch. Esto subdivide los cientos de miles de registros en pequeños lotes o "mini-batches", lo que erradica la saturación de memoria RAM y favorece la generalización dinámica del patrón del piloto.
- **Arquitectura de la Red:**
  - **Entrada:** 2 neuronas elementales (Velocidad y RPM).
  - **Capa Oculta 1:** 32 neuronas paramétricas (mediadas por la activación ReLU).
  - **Capa Oculta 2:** 16 neuronas paramétricas (con ReLU). Al poseer dos capas, la red cruza el umbral de "Deep Learning" simple (red de cierta profundidad) logrando engranar patrones invisibles de las mecánicas físicas automovilísticas.
  - **Salida:** 1 neurona. Frente al caso BMW (Donde un precio puede ser $24500), el coche frena o no frena. Para mapear la decisión, se usa una función de activación final llamada **Sigmoid**. La Sigmoid "comprime" todo conocimiento adquirido predecido a un margen estricto entre el valor constante `0 y 1`, proporcionando no solo la predicción final, sino una **probabilidad abstracta confiable** (ej. "Tengo 89% de seguridad de que en este microsegundo, el piloto estaba tocando el freno").
- **Optimizador, Función de Pérdida y Análisis Estadístico:**
  - **Función de Pérdida:** Se migra a un entorno de **BCELoss** (Entropía Cruzada Binaria, Binary Cross Entropy). Es la solución asintótica para problemas probabilísticos binarios; evalúa la diferencia logarítmica entre curvas de distribución.
  - **Evaluación Final de Rigor:** Simplemente ver bajar un número de pérdida no garantiza que esté aprendiendo. Por ello, apartamos y congelamos un 20% del conjunto de pruebas original como Test Set para someter a examen exhaustivo a la Red Neuronal, de modo que evalué su precisión y represente el dictámen en una estructurada **Matriz de Confusión**. Esta permite un análisis riguroso de falsos positivos y falsos negativos en frenadas.

## 5. Cierre Resumido

Transportar los ejercicios al ecosistema PyTorch no se limita únicamente a dejar de calcular derivadas matemáticas a mano. Esta maniobra transicional significa una optimización abismal en el tratamiento de estructuras multivariables irregulares frente a la rígida regresión lineal o logística tradicional. Facilita la adopción de herramientas estructurales listas como DataLoaders, funciones asintóticas preempaquetadas y manejo directo de GPUs para la experimentación con Datasets modernos, dotando de fundamento al resto de laboratorios venideros a medida que se escale el nivel de los datos computados.
