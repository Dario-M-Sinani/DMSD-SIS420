# 🛠️ Explicación Técnica y Detallada del Script (`preprocesamiento_total.py`)

Este documento detalla la arquitectura algorítmica detrás del script maestro que consolida de forma automática la limpieza de los 9 datasets del parcial, listando de forma minuciosa las transformaciones precisas realizadas por dataset, y aclarando la estructura estadística obtenida.

---

## 1. Tratamiento a Nivel de Código por Dataset

El script fue programado de manera modular y resiliente. Cada flujo está encapsulado en bloques individuales `try-except` lo cual previene que el programa principal falle ante la potencial pérdida o alteración física de un `.csv`.
A continuación se explica línea por línea la justificación y transformación técnica aplicada en cada función:

### 📊 Dataset 1: Ames Housing (`procesar_ames`)
1. **Imputación Inteligente de Nulos (LotFrontage):** En lugar de imputar por la media global y distorsionar los datos, el script agrupa las casas por vecindario (`df.groupby('Neighborhood')`) y le asigna la **mediana topográfica** del vecindario específico de cada casa con `transform(lambda x: x.fillna(x.median()))`.
2. **One-Hot Encoding (OHE):** Las redes neuronales no entienden la palabra "CollgCr". Se aplicó la función paramétrica `pd.get_dummies` aislando la columna `Neighborhood`, lo que convierte un solo atributo de texto en múltiples variables binarias (0 y 1). Se aplicó `drop_first=True` para esquivar la multicolinealidad matemática (Dummy Variable Trap) de los modelos lineales.

### 📊 Dataset 2: MIMIC-III Demo (`procesar_mimic`)
1. **Feature Engineering - Temporal a Decimal:** Un modelo estadístico es ciego a las fechas tradicionales. Se estandarizaron los vectores de ingreso y egreso con `pd.to_datetime()`, restándolos para generar una nueva característica predictiva invaluable: `LOS_dias` (Length of Stay). Para precisión estricta, la diferencia se calculó usando mili-segundos y se dividió para calcular una permanencia decimal exacta `(24 * 3600)`.
2. **Supresión de Cardinalidad Severa:** De haber encontrado diagnósticos (CIE-9 / ICD-9) como '410.12', el script los truncó por string dinámico al grupo raíz de patología `str[:3]` ('410' -> Enfermedades del corazón), fusionando enfermedades ultra-raras con enfermedades matrices.

### 📊 Dataset 3: NHANES 2021-2023 (`procesar_nhanes`)
1. **Unión Serial Multi-archivo (`Merge`):** Aplicó iteración de lectura para decodificar los 4 módulos `.XPT` independientes. Llevó a cabo 3 fusiones internas paramétricas `how='inner'` enganchando los archivos utilizando estrictamente al identificador `SEQN`. El modelo ahora cruza presión sistólica contra grasa hepática sobre la misma traza celular del mismo paciente sin falsos emparejamientos.
2. **Purgado Vital (Dropna):** Cortó registros huérfanos que imposibilitaban la supervivencia de un modelo. Eliminó de manera tajante registros biológicos que carecían del atributo IMC y la presión central sistólica (`dropna(subset=['BMXBMI', 'BPXOSY1'])`). 

### 📊 Dataset 4: Bike Sharing (`procesar_bike_sharing`)
1. **Descomposición de Ciclos:** Como las rentas de bicis responden al ciclo del sol y calendario, se extrajo la correlación cíclica dividiendo explícitamente y creando columnas discretas con factores influyentes como `df['datetime'].dt.hour`.
2. **Log-Transform Poblacional:** Como cualquier producto de alta demanda en horas pico, el resultado es dramáticamente asimétrico a la derecha. El script aplicó la fórmula NumPy de Suavizado Integral: `np.log1p(df['count'])` [Log(1+x)], achatando los valores absurdos atípicos para forzar a la Variable Predictora Base a buscar una figura de campana perfecta (Normalidad Gausiana).

### 📊 Dataset 5: Adult Income (`procesar_adult_income`)
1. **Manejo Ortográfico de Importación:** Los archivos crudos del UCI Repository suelen fallar en Pandas porque unen comas con espacios irregulares. Se neutralizó esto usando `sep=r',\s+'`.
2. **Subversión Sensata de Nulos (Imputación Semántica):** En lugar de tratar al carácter sospechoso (`?`) como un ruido a borrar, o imputarle con una Moda incierta; se transformó textualmente al estado `"Unknown"`. Los economistas y algoritmos interpretan que "ocultar" la ocupación en un censo ya es de por sí una altísima variable de predicción de la pobreza (Target: >50K).
3. **Binarización:** La celda de beneficio mayor se tradujo matemáticamente al entero `1` con una Función Lambda vectorizada.

### 📊 Dataset 6: Credit Approval - CRX (`procesar_crx`)
1. **Duck-Typing Seguro (`try-except .astype`):** Un archivo con muchísima perturbación ofuscada. Como los valores bancarios son secretos, hay nulos interconectados. El código ejecuta una fuerza bruta sobre todo el DF intentando transformarlos de forma estricta a Flotantes Decimales.
2. **Bifurcación Algorítmica Imputacional (Media vs Moda):** Empleó la biblioteca experta interna de API de Pandas (`is_numeric_dtype`). Las columnas que pasaron el Duck-Typing y fueron leídas como número, ganaron una Imputación robusta a `Mediana`, rellenando sus huecos y siendo recodificados artificialmente entre 0.0 y 1.0 geométrico con `MinMaxScaler`. Las columnas que no pasaron (Alfabéticas categorizadas), se agruparon sustituyendo huecos con la categoría popular y mayoritaria `.mode()[0]`.

### 📊 Dataset 7: Statlog Australian (`procesar_australian`)
1. **Extirpación Quirúrgica en Tails Limites (Clipping):** Los algoritmos perceptrón y SVM se paralizan al procesar extremos insólitos infinitos que distorsionan pesos. Si la columna australiana no es dicotómica (`nunique() > 2`), el algoritmo mide matemáticamente el límite ínfimo (1%) y supremo (99%) y aplasta los picos que los excedan congelándolos en ese límite seguro (`.clip(lower, upper)`).
2. **Estandarización Z-Score:** Se empleó `StandardScaler`. Las variantes económicas dispares perdieron el efecto de escala. El conjunto ahora gravita alrededor de la Media 0 e impone una Varianza de 1 exacto.

### 📊 Dataset 8: Breast Cancer Wisconsin (`procesar_breast_cancer`)
1. **Aislamiento Sensitivo Nuclear:** Obliga al comando infernar `.dropna()` exclusivamente en la métrica fotográfica oncológica de `Bare_Nuclei`. Fabricar valores promedios irreales inventando núcleos invisibles de un paciente, para no borrar 16 filas inservibles, es éticamente letal. Por ello, la regla Listwise-drop predominó.
2. **Neutralización del Ruido ID:** Se retiró contundentemente vía `.drop` a la métrica identificadora `ID`, evadiendo falsedades de regresión en el modelaje de Inteligencia Artificial que pueda relacionar un dígito secuencial al azar como correlación de un tumor maligno.

### 📊 Dataset 9: Meningitis Missing 12 (`procesar_meningitis`)
1. **KNN-Imputer Biológico:** Se segregó la tabla aislando variables puramente clínicas cuantitativas (`select_dtypes`). Posteriormente, para suplir el espacio hueco del paciente (ej. falta examen de leucocitos), el script invoca al modelo avanzado `KNNImputer(n_neighbors=5)`. Este calcula la distancia euclidiana entre dicho paciente y los **5 perfiles que más se le asemejan biológicamente en la sala**, transfiriendo e integrando las proporciones promedios calculadas a su expediente de forma mucho más orgánica para el metabolismo general de este mal etiológico.

---

## 2. 🧩 El enigma del Dataset 3 (NHANES): ¿Por qué solo veo números?

Cuando abras el archivo resultante `nhanes_ready.csv` notarás de inmediato que absolutamente todas las columnas (incluso tu género, el estado moral al responder la entrevista biomédica, la respuesta de equilibrio, o el estatus de las presiones de brazalete) parecen ser puros números decimales ciegos.

**Explicación Fundamental:**
El pipeline de Data Engineering no eliminó las letanía, ni hay daño o pérdida de información estructural en tu csv de origen NHANES derivado de este script; el motivo de la ceguera semántica recae en la fisiología de su formato en **Ecosistemas Avanzados SAS (`.XPT`)**.

El *Statistical Analysis System* (SAS) del Gobierno Federal de los EE. UU. cruza perfiles a niveles nacionales colosales con una eficiencia de memoria bruta. Las bases de SAS de la gran parte de dependencias de la Organización Mundial de la OMS o la CDC **jamás exportan en cadenas de memoria alfabética (`"White/Black"`, `"Masculino"`)**. Todo registro cualitativo y cuantitativo se convierte irremediablemente durante la recolección, usando bases de diccionarios subyacentes, a banderas binarias numéricas continuas/discretas ("1", "2").

* **Evidencia Práctica 1 (Prueba Postural):** Visualizarás categóricas camufladas como `BAXSTAT` (Chequeo neurológico del equilibrio). En lugar de narrar de forma rústica un largo mensaje como "Prueba completada" o "Paciente no ejecutó prueba", SAS la encriptó nativamente en un número flotante de 64-bits que marca `1.0` si la ejecutó exitosamente, `2.0` si fue parcial, o `3.0` si no se realizó.
* **Evidencia Práctica 2 (Elastografía Hepática):** En lugar de decir "Ausencia de medición" o "Medido con éxito", el módulo de hígado (`LUX_L`) procesado directamente expulsa la métrica continua numérica validada de Fibrosis en KiloPascales (`LUXSMED`). Las variables categóricas o de evaluación siempre tendrán de igual modo la forma `1.0` (Positivo/Completo) y `2.0` (Negativo/Fallo).

### La Ventaja Escondida (The Blessing in Disguise)
En las competencias de inteligencia artificial de este parcial, la estructura originaria codificada numéricamente de las columnas en tu archivo .CSV NHANES te confiere una **Inmunidad contra Errores de Categoría de Texto**. El trabajo tedioso, abismal y complejo del **Label Encoding** ya se realizó directamente en los despachos de la CDC por nosotros. Como programador Machine Learning Data-Scientist en esta prueba, tu posterior algoritmo estadístico podrá inyectarse nativamente a todos los atributos sin colapsos de texto que arruinen los tensores predictivos de tus arquitecturas.
