# 📚 Guía Detallada de los 9 Datasets 

Este documento proporciona una radiografía completa de cada uno de los datasets incluidos en el proyecto. Para cada conjunto de datos se explica qué es, para qué puede ser utilizado, su morfología (filas y columnas) y particularidades importantes para su análisis con Inteligencia Artificial.

---

## 1. Ames Housing Dataset (Predicción de Precios de Vivienda)
* **¿Qué es?**: Un registro muy extenso de las características físicas, ubicación y estado de las viviendas vendidas en la ciudad de Ames, Iowa. Fue creado como una alternativa más moderna y rica frente al antiguo dataset de Boston Housing.
* **¿Para qué se podría usar?**: 
  - **Uso principal**: Modelos de regresión lineal múltiple, Random Forest Regression o Gradient Boosting para predecir la variable continua del precio de venta (`SalePrice`).
  - **Retos técnicos**: Ideal para practicar el tratamiento de valores nulos estructurales (por ejemplo, cuando "Nulo" no es un error, sino que significa "No tiene piscina" o "No tiene garaje") y lidiar con alta multicolinealidad entre métricas (tamaño del lote de terreno vs área construida).
* **Cuántas columnas tiene**: 82 columnas detalladas (mezcla de categóricas nominales, ordinales y numéricas continuas).
* **Cuántos ejemplos tiene**: 2,930 registros detallados de viviendas.
* **¿Qué representa exactamente un ejemplo (fila)?**: Cada fila es **una casa individual** del mundo real vendida en Ames. Un registro típico luciría así en la data: *"Casa construida en 1995, ubicada en el barrio North Ames, sin piscina, con 2000 pies cuadrados de área habitable, calificada con un 7/10 en calidad, y vendida finalmente por $215,000"*.

* **Explicación de sus columnas principales**:  
  Las **82 columnas** existen porque detallan de manera sumamente minuciosa cada aspecto concebible de una casa que afecta su precio. Se dividen en dimensiones espaciales, calificaciones de calidad, antigüedad, y comodidades (sótanos, piscinas, garajes, tipo de techo).

| Columna Principal | Descripción de la Variable |
|---|---|
| `SalePrice` | **[Variable Objetivo]** Precio final de venta de la casa. |
| `Neighborhood` | Barrio o vecindario donde está ubicada la casa. |
| `OverallQual` | Calificación global de 1 a 10 sobre los materiales y acabados. |
| `GrLivArea` | Superficie habitable sobre el nivel del suelo (en pies cuadrados). |
| `GarageType` / `LotFrontage` | Tipo de garaje y tamaño de la fachada (contienen nulos estructurales importantes). |


---

## 2. MIMIC-III Clinical Database (Base de Datos Clínicos - Demo)
* **¿Qué es?**: Una muestra (Demo) de una base de datos relacional inmensa que guarda información de pacientes internados en la Unidad de Cuidados Intensivos (UCI) del centro médico Beth Israel Deaconess Medical Center.
* **¿Para qué se podría usar?**: 
  - **Uso principal**: Predecir el tiempo de estancia en el hospital (`Length of Stay`), predecir la mortalidad intrahospitalaria o realizar NLP (Procesamiento de Lenguaje Natural) con las notas de los médicos.
  - **Retos técnicos**: Requiere un manejo avanzado de SQL o `pandas` para cruzar múltiples tablas (pacientes, admisiones, eventos de laboratorio). Se usa mucho para realizar codificación guiada por objetivo (Target Encoding) agrupando códigos de la CIE-9 (ICD-9).
* **Cuántas columnas tiene**: Varía enormemente al ser relacional. La tabla base `PATIENTS` tiene 8 columnas, mientras `ADMISSIONS` tiene 19, conectadas a través de docenas de características extraídas.
* **Cuántos ejemplos tiene**: La versión de demostración contiene datos de **100 pacientes únicos** que se expanden a miles de registros secundarios relativos a la evolución temporal de sus signos vitales.
* **¿Qué representa exactamente un ejemplo (fila)?**: Al ser una base relacional, depende de la tabla. En `PATIENTS`, una fila es **una persona** (ej: *"Mujer, nacida en 1954"*). En `ADMISSIONS`, una fila es **un ingreso al hospital** (ej: *"Ingreso de urgencia a las 03:00 AM por falla cardíaca"*). En `LABEVENTS`, una fila es **un solo examen de sangre** sacado en un minuto específico a ese paciente.

* **Explicación de sus columnas principales**:  
  Al ser una base de datos relacional, las columnas son extraídas de diferentes tablas (como `PATIENTS` y `ADMISSIONS`) mediante *JOINs* de SQL o pandas:

| Columna Principal | Selección de Tabla y Descripción |
|---|---|
| `SUBJECT_ID` / `HADM_ID` | Identificadores únicos del paciente y de su ingreso hospitalario respectivo. |
| `LOS` | **[Variable Objetivo]** Días totales de estancia en cuidados intensivos (*Length of Stay*). |
| `ICD9_CODE` | Diagnóstico clínico codificado del paciente (hay miles de códigos distintos). |
| `ADMITTIME` | Fecha y hora exacta de admisión para calcular duración con series de tiempo. |

---

## 3. NHANES / NAMCS 2021-2023 (Encuesta Nacional de Salud - CDC)
* **¿Qué es?**: Diversos módulos clínicos de la Encuesta Nacional de Examen de Salud y Nutrición de EE. UU. (NHANES). Se divide en exámenes por dominio: `BMX` (Antropometría/Masa Corporal), `BPX` (Presión arterial), `BAX` (Equilibrio) y `LUX` (Ultrasonido Hepático).
* **¿Para qué se podría usar?**: 
  - **Uso principal**: Sistemas de clasificación de riesgo cardiovascular temprano o modelos de regresión para estimar rigidez hepática usando indicadores indirectos de todo el cuerpo.
  - **Retos técnicos**: Permite perfeccionar las uniones de bases de datos biomédicas con llaves únicas (la columna `SEQN`). Permite aprender a gestionar archivos en el formato SAS `.xpt` y trabajar con datos en los que todas las encuestas ya se han convertido a formato numérico desde origen.
* **Cuántas columnas tiene**: BMX tiene **22 columnas**, BAX tiene **45**, BPXO **12**, y LUX **13**. Una vez unidos, aportan alrededor de 92 dimensiones de estudio clínico.
* **Cuántos ejemplos tiene**: El módulo corporal (BMX) tiene **8,860 pacientes**. Tras aplicar una fusión (`INNER JOIN`) para dejar sólo aquellos que hicieron las 4 pruebas, resultan aproximadamente en ~4,000+ pacientes con datos completos en distintos módulos.
* **¿Qué representa exactamente un ejemplo (fila)?**: Tras la sincronización de tablas, cada fila equivale a **un paciente real encuestado y medido físicamente**. Un ejemplo unificado sería: *"Paciente masculino (SEQN=10023), que tiene una presión arterial de 120mmHg, un IMC de 28, completó la prueba de equilibrio satisfactoriamente y su hígado presenta una rigidez de 5.2 kPa"*.

* **Explicación de sus columnas principales**:  
  Los módulos encuestados arrojan resultados codificados biomédicamente en un archivo numérico SAS `.xpt`. Las más importantes para IA son emparejadas con esta llave:

| Columna Principal | Módulo de Origen y Descripción |
|---|---|
| `SEQN` | **[Llave Primaria]** Identificador del paciente. Obligatorio para realizar cruces de tablas. |
| `LUXSMED` | Módulo *LUX*: Mediana de la rigidez hepática (Elastografía). Puede ser la **variable objetivo**. |
| `BPXOSY1` | Módulo *BPX*: Medición principal de Presión Arterial Sistólica. |
| `BMXBMI` | Módulo *BMX*: Índice de Masa Corporal (IMC) del paciente. |
| `BAXSTAT` | Módulo *BAX*: Estado en el que quedó la prueba de equilibrio (Completada o No). |

---

## 4. Bike Sharing Demand (Demanda en Alquiler de Bicicletas)
* **¿Qué es?**: Registro histórico del flujo de bicicletas de alquiler del programa de Washington D.C., combinando recuentos de uso con indicadores meteorológicos (temperatura, lluvia, humedad, viento).
* **¿Para qué se podría usar?**: 
  - **Uso principal**: Análisis y regresión de series temporales (Time Series Forecasting) para proyectar y prevenir cuellos de botella mediante la predicción logística. 
  - **Retos técnicos**: Es la herramienta perfecta para enseñar cómo transformar variables de "Fecha" (datetime) en entidades ciclícas usando el Seno y el Coseno o extraer la hora para que la IA entienda el "ciclo diario" de alta demanda.
* **Cuántas columnas tiene**: 12 variables (componentes del clima y fecha) + 1 objetivo, dando unas 13 columnas.
* **Cuántos ejemplos tiene**: Alrededor de 10,886 registros en el grupo de entrenamiento, con intervalos por cada hora estructurada.
* **¿Qué representa exactamente un ejemplo (fila)?**: Cada fila **no es una persona ni un objeto**, sino **una hora exacta del universo temporal**. Ejemplo literal: *"El 15 de marzo de 2011, entre las 08:00 AM y 09:00 AM, bajo un clima parcialmente nublado a 15°C y con vientos de 12 km/h, la ciudad experimentó un total de 342 bicicletas alquiladas simultáneamente"*.

* **Explicación de sus columnas principales**:  
  Se divide equitativamente entre aspectos de estacionalidad térmica y estampas de marca del reloj:

| Columna Principal | Descripción de la Variable |
|---|---|
| `count` | **[Variable Objetivo]** Cantidad neta de bicicletas alquiladas en esa hora en particular. |
| `datetime` | Fecha y hora exacta. (Debe ser descompuesta numéricamente en hora, día y mes para el modelo). |
| `temp` / `humidity` | Temperatura climática y humedad relativa estructurada. |
| `windspeed` | Medición y velocidad del viento. |
| `season` / `weather` | Estación del año enumerada y tipo de clima presente. |

---

## 5. Adult Income Dataset (Datos del Censo - Ingresos)
* **¿Qué es?**: Un clásico inamovible extraído de la base del censo de 1994 de EE.UU. Engloba información demográfica (edad, educación, origen, género, ocupación).
* **¿Para qué se podría usar?**: 
  - **Uso principal**: Clasificación binaria (predecir si la persona gana `<=50K` o `>50K` anuales).
  - **Retos técnicos**: Analíticas de justicia de Modelos (Fairness IA). Enseña de manera precisa a procesar valores nulos camuflados, puesto que algunos datos faltantes ingresaron como un signo de interrogación `?` intencionalmente por privacidad ciudadana y es mejor abordarlos como categoría `Unknown` en lugar de imputarlos de forma agresiva.
* **Cuántas columnas tiene**: 15 columnas en total (14 predictoras, 1 variable objetivo).
* **Cuántos ejemplos tiene**: Un volumen robusto de **32,561 personas encuestadas**.
* **¿Qué representa exactamente un ejemplo (fila)?**: Cada fila corresponde a **un individuo encuestado** en el censo del 94. Un registro típico es: *"Hombre de 42 años, de origen afroamericano, casado, con título de Maestría, que trabaja en un gobierno local, y su ingreso anual reportado sobrepasa el límite (>50K)"*.

* **Explicación de sus columnas principales**:  
  Reúne una radiografía socieconómica del ciudadano promedio a través de metadatos de categoría:

| Columna Principal | Descripción de la Variable |
|---|---|
| `income` | **[Variable Objetivo binaria]** Indica `<=50K` (Gana $50 mil o menos) o `>50K` (Gana más de $50 mil). |
| `workclass` | Sector de origen de sus fondos monetarios o trabajo (Privado, Autónomo, Gobierno, o `?`). |
| `education` | Nivel educativo máximo culminado. |
| `occupation` | Ocupación laboral principal desarrollada por el encuestado. |
| `marital-status` | Condición subyacente y familiar (Estado civil). |

---

## 6. Credit Approval Dataset (CRX - Aprobación de Créditos)
* **¿Qué es?**: Datos reales de solicitudes de tarjetas de crédito japonesas cedidos a investigadores bajo el anonimato. Las columnas y su información (tanto categórica como numérica) han sido ocultadas con nombres como (A1, A2, A3... A15).
* **¿Para qué se podría usar?**: 
  - **Uso principal**: Clasificación (algoritmos bancarios para aprobación/rechazo automático).
  - **Retos técnicos**: Forzar a realizar un trabajo "a ciegas". Entrena la capacidad del científico de datos para ejecutar *Duck-Typing* y limpiar variables mezcladas sin contexto humano guiándose por la estadística descriptiva (distribución, mediana y modas) para estandarizar en un rango controlable (e.g., con `MinMaxScaler`).
* **Cuántas columnas tiene**: 16 columnas enmascaradas.
* **Cuántos ejemplos tiene**: 690 aplicaciones de tarjetas, que conforman un conjunto de datos pequeño pero sensible al ruido metodológico.
* **¿Qué representa exactamente un ejemplo (fila)?**: Cada fila es **una solicitud de tarjeta de crédito**. Al estar la data cifrada, no podemos leer el nombre ni el salario real, sino abstracciones como: *"Sujeto con variable categórica A1='b', variable A2=30.83, variable A3=15.2 (ej: años trabajando), con el estado de aprobación crediticia final siendo '-' (solicitud rechazada)"*.

* **Explicación de sus columnas principales**:  
  Todo aspecto identificable ha sido ocultado deliberadamente para preservar la identidad del cliente (anonimización):

| Columna Principal | Descripción de la Variable |
|---|---|
| `A16` | **[Variable Objetivo]** Dictamina si la tarjeta de crédito se aprueba (`+`) o se rechaza (`-`). |
| `A1` a `A15` | Predictores totalmente anónimos. Detrás de ellos se ocultan variables abstractas como Historial Crediticio, Ingresos, Antigüedad o Sexo, que no podemos nombrar de manera certificada. Mezclan formato numérico con categórico. |

---

## 7. Statlog Australian Credit Approval 
* **¿Qué es?**: Hermano del Dataset CRX. Trata sobre lo mismo (decisiones crediticias abstractas), sin embargo, está "extremadamente pulido", de manera sistemática los valores de clases ya se estandarizaron localmente a escalas numerales en las entrañas del dataset.
* **¿Para qué se podría usar?**: 
  - **Uso principal**: Clasificación binaria con algoritmos tipo SVM o Redes Neuronales (MLP) ligeras.
  - **Retos técnicos**: Es muy valioso para demostrar metodologías de castigo o recortes (*Clipping* con Z-Score) para "domar" outliers y picos estadísticos excesivos reduciéndolos a la desviación estándar aceptada, evitando prescindir de datos como en los procedimientos tradicionales de descarte (drop).
* **Cuántas columnas tiene**: 15 columnas enteramente numéricas (float64/float32 preprocesadas).
* **Cuántos ejemplos tiene**: 690 individuos.
* **¿Qué representa exactamente un ejemplo (fila)?**: Al igual que el CRX, cada fila dictamina el destino de **un formulario bancario**. Ejemplo: *"Solicitud 45A, donde el predictor flotante A2=22.08, A3=11.46 (posiblemente deudas procesadas por un MinMaxScaler), y finalmente la Clase resultante es '1' (Tarjeta Aprobada)"*.

* **Explicación de sus columnas principales**:  
  Similar al CRX, pero todas sus columnas que eran originalmente texto/categoría fueron traducidas a números estandarizados desde la fuente:

| Columna Principal | Descripción de la Variable |
|---|---|
| `Class` | **[Variable Objetivo]** Aprobación crediticia o denegación (En formato ya binarizado de `0` o `1`). |
| `A1` a `A14` | Valores predictores cifrados flotantes. Es notorio estadísticamente que los features `A2`, `A3` y `A7` poseen la mayor desviación y varianza (outliers) en las aplicaciones bancarias cotidianas. |

---

## 8. Breast Cancer Wisconsin (Diagnostic Clínico)
* **¿Qué es?**: Data capturada midiendo visualmente las varianzas de núcleos celulares al someter un bulto mamario a una aspiración de aguja fina.
* **¿Para qué se podría usar?**: 
  - **Uso principal**: Clasificación binaria vital (Diferenciar entre patología "Maligna" vs "Benigna").
  - **Retos técnicos**: Al poseer rasgos anatómicos tan conectados entre sí (el perímetro, el área y el radio de una célula correlacionan perfectamente), es un proyecto de oro puro para poner en marcha algoritmos no supervisados de Reducción de la Dimensionalidad (como Análisis de Componentes Principales, o **PCA** en Python) y consolidación biológica clínica.
* **Cuántas columnas tiene**: Varía por la versión. Posee la matriz clásica de **11 columnas** y las matriciales computadas de 32 o hasta 35 columnas (wdbc y wpbc).
* **Cuántos ejemplos tiene**: Típicamente **699 pacientes** (la versión extendida cuenta con 569 de casos pulcramente divididos para algoritmos médicos diagnósticos).
* **¿Qué representa exactamente un ejemplo (fila)?**: Cada fila es **un estudio de microscopio sobre un tejido de masa mamaria** extraído de una paciente mujer viva. El registro se interpreta así: *"Biopsia #851042 reporta células con 14.2 μm de radio, un perímetro de 90.1 μm, textura uniforme visual (20.3) y mínima deformidad; concluyendo un tejido tipo 'B' (Benigno)"*.

* **Explicación de sus columnas principales**:  
  Los features corresponden a elementos analizados artificialmente a través de microscopio en formaciones nucleares de las glándulas mamarias.

| Columna Principal | Descripción de la Variable |
|---|---|
| `Diagnosis` | **[Variable Objetivo]** Categoriza el tumor directamente entre Benigno (`B`) y Maligno (`M`). |
| `Radius_mean` | Magnitud promedio del alcance y radio de los núcleos malignos de células halladas. |
| `Texture_mean` | Variación en los patrones de luz y sombras (escala de grises) al evaluar la célula fotográficamente. |
| `Perimeter_mean` | Longitud perimetral del núcleo de la masa anormal detectada. |
| `Bare_Nuclei` | Proporción visual de los núcleos desnudos localizados en la toma. *Importante:* Contiene errores incrustados con signos `?`. |

---

## 9. Meningitis Clinical Dataset (con agujeros forzados)
* **¿Qué es?**: Resultados cruciales bioquímicos procedentes de la punción de líquido cefalorraquídeo, género y edad, orientados al diagnóstico diferencial de meningitis.
* **¿Para qué se podría usar?**: 
  - **Uso principal**: Clasificación multiclase en urgencias médicas para dirimir entre el origen: Bacterial, Viral o Fúngico.
  - **Retos técnicos**: Contiene lagunas informáticas introducidas bajo un patrón de estudio de "condiciones críticas de emergencia", donde recolectarlo todo es imposible en la vida real. Es supremo para poner a prueba algoritmos de `KNNImputer`, ya que la inferencia de valores vacíos aquí no debe ocurrir cruzando la población general, sino utilizando a los "5 vecinos diabéticos y febriles semejantes".
* **Cuántas columnas tiene**: 14 columnas biomédicas y demográficas.
* **Cuántos ejemplos tiene**: **1,200 observaciones** muy compactas y relevantes clínicamente.
* **¿Qué representa exactamente un ejemplo (fila)?**: Cada fila es **un paciente sospechoso llegado de urgencia**. Ejemplo clínico inmersivo: *"Niño varón de 8 años en coma que, al hacerle punción lumbar exploratoria, arroja Leucocitos disparados (600), Glucosa desplomada (15 mg/dL) y Proteínas elevadas; su etiología final se codifica como Meningitis **Bacteriana**"*.

* **Explicación de sus columnas principales**:  
  Características que describen tanto la demografía del cuadro médico, como las mediciones críticas (que a falta de poder medir en urgencia extrema, generan lagunas que la IA debe imputar comparando pacientes):

| Columna Principal | Descripción de la Variable |
|---|---|
| `Etiología` | **[Variable Objetivo Multiclase]** Discerne en 3 clases el origen del brote: Meningitis Viral, Bacteriana o Fúngica. |
| `Leucocitos` | Conteo intensivo de glóbulos blancos. Vital porque en cepas bacterianas explota considerablemente. |
| `Glucosa` | Niveles en el LCR; suelen estar en decaimiento en infecciones bacterianas, pero normales en las víricas. |
| `Proteínas` | Concentración detectada. Su pico y alzada advierte severidad inmunológica. |
| `Edad` y `Género` | Marcadores demográficos de base. |

---
*Fin de la guía descriptiva. Esta documentación puede usarse para mapear el ciclo de ingeniería de características a implementar por cada sector del script de Jupyter de IA a futuro.*
