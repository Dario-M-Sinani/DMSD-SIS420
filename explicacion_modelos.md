# Explicación Detallada de Modelos de Machine Learning

Este documento explica en detalle los propósitos, el funcionamiento y las gráficas generadas por los dos scripts desarrollados: `regresion_logistica_telemetry.py` y `regresion_multiple_bmw.py`.

---

## 1. Regresión Logística (`regresion_logistica_telemetry.py`)

### ¿Para qué sirve?
Este script implementa un modelo de **Clasificación Binaria** utilizando el algoritmo de **Regresión Logística**. Su objetivo es predecir una de dos posibles categorías: si un jugador en el juego Forza Motorsport está **Frenando (1)** o **No Frenando (0)** en un momento dado, basándose en la **Velocidad** del vehículo y las **Revoluciones por Minuto (RPM)** del motor.

### ¿Cómo funciona paso a paso?
1. **Carga de Datos:** Lee los datos de telemetría desde un CSV. Ya que el dataset es muy grande, extrae una muestra aleatoria (1000 registros) de las columnas `speed` (Velocidad), `current_engine_rpm` (RPM) y crea la variable objetivo `y` a partir de la columna `brake` (1 si presiona el freno, 0 si no).
2. **Función Sigmoide:** A diferencia de la regresión lineal que devuelve números arbitrarios, este modelo usa la función sigmoide para "aplastar" el resultado matemático y convertirlo en un rango entre `0` y `1`, lo cual se interpreta como una **probabilidad**.
3. **Función de Costo y Gradiente:** Calcula qué tan equivocadas están las predicciones actuales utilizando la Entropía Cruzada (logarítmica).
4. **Optimización Avanzada:** En lugar de hacer un bucle manual para encontrar los mejores parámetros matemáticos (thetas), utiliza una librería experta `scipy.optimize.minimize` que realiza este cálculo de forma mucho más rápida y precisa.
5. **Predicción y Evaluación:** Evalúa la precisión del modelo en los datos de prueba y, finalmente, toma datos ficticios nuevos de carrera para predecir si el conductor frenará o no.

### Explicación del Gráfico: Distribución de Datos y Frontera de Decisión
* **Ejes:** El eje X representa la Velocidad y el eje Y representa las RPM del motor.
* **Puntos:** Cada punto en la gráfica es un momento registrado en el juego. 
  * Las cruces negras (`+`) representan los momentos donde el conductor estaba **Frenando**.
  * Los círculos amarillos (`o`) representan los momentos donde el conductor **No estaba Frenando** (estaba acelerando o en neutro).
* **Frontera de Decisión (Línea Azul):** Esta línea recta es el resultado inteligente de nuestro modelo. Es la barrera matemática que el algoritmo descubrió para separar ambos comportamientos. Cualquier dato o punto nuevo que caiga de un lado de la línea será clasificado por el sistema como "Frenando", y si cae del otro lado será clasificado como "No Frenando".

---

## 2. Regresión Lineal Múltiple (`regresion_multiple_bmw.py`)

### ¿Para qué sirve?
Este script implementa un modelo de predicción continua utilizando **Regresión Lineal Múltiple**. A diferencia del script anterior que predecía categorías, este modelo predice un **valor numérico específico**: estimar o predecir el **Precio de Venta ($ USD)** de un automóvil marca BMW, en base a tres variables: el **Año** del vehículo, el **Tamaño del Motor (Litros)** y su **Kilometraje recorrido**.

### ¿Cómo funciona paso a paso?
1. **Carga de Datos:** Toma la información general del CSV de ventas de BMW y aísla solamente las características numéricas que nos interesan (Año, Motor_L, Mileage_KM) y el Precio.
2. **Normalización de Características:** Este es el paso más crítico. Como el Año está en los miles (ej. 2016), el motor en unidades pequeñas (ej. 3.0) y los kilómetros en cientos de miles (ej. 150000), el algoritmo se confundiría sin una escala equitativa. Esta función matemática estandariza todas las variables para que tengan un rango similar (generalmente entre -1 y 1) y el algoritmo pueda aprender más rápido y sin sesgos.
3. **Descenso por el Gradiente Manual:** Configura la ecuación de predicción y ejecuta un bucle manual (`for`) que se repite cientos de veces (`num_iters=500`). En cada iteración, el algoritmo da un pequeño "paso" (`alpha=0.03`, Tasa de aprendizaje) para ajustar los multiplicadores matemáticos (Thetas) intentando reducir el error de cálculo cada vez más.
4. **Predicción:** Una vez que el algoritmo termina de aprender, toma los datos de un vehículo completamente nuevo (Ej. un BMW 2021, motor 3.0 y 45,000Km), **le aplica la misma normalización** y nos devuelve cuál debería ser su precio justo en el mercado.

### Explicación del Gráfico: Comportamiento del Descenso por el Gradiente
* **¿Qué estamos viendo?** Esta gráfica nos sirve como "monitor" de la salud cerebral (aprendizaje) de nuestra IA durante el entrenamiento.
* **Ejes:** El eje X representa el "Número de Iteraciones" (los intentos o pasos que dio el modelo para aprender). El eje Y representa el "Costo J" (cuantifica qué qué tan equivocadas y alejadas de la realidad estaban las predicciones del modelo en ese paso).
* **Interpretación de la Curva:** 
  * Al principio (iteración 0), el costo es muy alto porque el modelo no sabe nada y se equivoca brutalmente con los precios.
  * A medida que avanza a la derecha (más intentos), vemos que la línea roja **cae drásticamente**. Esto significa que el algoritmo "Descenso por el Gradiente" funciona perfectamente, reduciendo el error en cada intento.
  * Hacia el final, la línea se estabiliza de manera plana y horizontal. A esto se le llama **Convergencia**. Significa que el algoritmo llegó a su límite de aprendizaje máximo con los datos proporcionados y encontró los mejores números posibles para realizar sus predicciones. Si la recta nunca se aplanara o se fuera hacia arriba, indicaría que nuestro `alpha` (Tasa de aprendizaje) estaba mal configurado.
