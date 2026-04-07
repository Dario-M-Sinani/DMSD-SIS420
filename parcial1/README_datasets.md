# 📊 Documentación de Datasets — Fase 1: ETL y Preparación para ML

Este repositorio documenta la preparación de **9 datasets** para modelado predictivo. Cada uno fue sometido a análisis de tipos de datos, limpieza de valores nulos/anómalos y estandarización antes de pasar a la fase de modelado.

El script principal (`preprocesamiento_total.py`) está diseñado de forma **modular y resiliente**: cada dataset vive en su propio bloque `try-except`, lo que significa que si un archivo `.csv` falta o falla, el resto del pipeline continúa ejecutándose sin interrupciones.

---

## Índice

1. [Ames Housing Dataset](#1-ames-housing-dataset)
2. [MIMIC-III Clinical Database](#2-mimic-iii-clinical-database)
3. [NHANES 2021–2023](#3-nhanes-20212023--fusión-de-exámenes-físicos)
4. [Bike Sharing Demand](#4-bike-sharing-demand)
5. [Adult Income Dataset](#5-adult-income-dataset-census)
6. [Credit Approval Dataset (CRX)](#6-credit-approval-dataset-crx)
7. [Statlog Australian Credit](#7-statlog-australian-credit)
8. [Breast Cancer Wisconsin](#8-breast-cancer-wisconsin-diagnostic)
9. [Meningitis Clinical Dataset](#9-meningitis-clinical-dataset)
10. [Resumen general](#resumen-general-de-estrategias)

---

## ⚠️ Aclaración importante — Dataset 3 (NHANES vs NAMCS)

> El script actualmente en el repositorio corresponde a **NAMCS 2024** (National Ambulatory Medical Care Survey), pero el requerimiento apunta a **NHANES** (National Health and Nutrition Examination Survey). **Son datasets completamente distintos.**

| | NAMCS | NHANES |
|---|---|---|
| **¿Qué mide?** | Visitas médicas ambulatorias | Estado de salud y nutrición del paciente |
| **Tipo de datos** | Diagnósticos y medicamentos prescritos por visita | Exámenes físicos y resultados de laboratorio por persona |
| **Formato** | Un registro por visita médica | Múltiples módulos `.XPT` (DEMO, EXAM, LAB, Q) |
| **Llave de cruce** | No aplica entre módulos | `SEQN` — identificador único por paciente, compartido en todos los módulos |

**Cómo usar los archivos correctos de NHANES:**
1. Descargar los archivos `.XPT` de un **mismo ciclo** (ej. 2017–2018 o 2021–2023). Mezclar ciclos distintos genera datos inconsistentes.
2. Como mínimo necesitarás `DEMO_J.XPT` (demografía) y un archivo de laboratorio como `GLU_J.XPT` (glucosa).
3. Unir los módulos con `INNER JOIN` o `LEFT JOIN` usando la columna `SEQN` como llave.

---

## 1. Ames Housing Dataset

**Fuente:** Dean De Cock / Kaggle  
**Tipo de problema:** Regresión  
**Variable objetivo:** `SalePrice` — precio final de venta de la vivienda

### ¿Qué contiene este dataset?

Características físicas y contextuales de viviendas vendidas en Ames, Iowa (EE.UU.). El objetivo es construir un modelo que estime el precio de venta a partir de esas características.

### Variables principales

| Variable | Tipo | Descripción |
|---|---|---|
| `SalePrice` | Numérica continua | **Variable objetivo** |
| `Neighborhood` | Categórica nominal | Barrio donde se ubica la vivienda |
| `OverallQual` | Categórica ordinal (1–10) | Calidad general de materiales y acabados |
| `LotFrontage` | Numérica continua | Pies lineales de calle que conectan con la propiedad |
| `GrLivArea` | Numérica continua | Superficie habitable sobre nivel del suelo (ft²) |
| `GarageType` | Categórica nominal | Tipo de garaje (puede ser `NA` si no existe) |

### Limpieza de datos

**1. Imputación de `LotFrontage`**

Este campo tiene bastantes nulos. La estrategia no es imputar con la mediana global del dataset, sino con la **mediana del barrio (`Neighborhood`)** al que pertenece la propiedad. La razón: propiedades en el mismo barrio suelen tener frentes de calle similares, ya que los lotes fueron diseñados con estándares parecidos.

```python
df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'] \
                      .transform(lambda x: x.fillna(x.median()))
```

**2. Variables con `NA` que significan "no existe"**

En este dataset, muchos `NA` no son datos faltantes — son ausencia real del rasgo. Ejemplos:
- `GarageType = NA` → la propiedad **no tiene garaje**
- `PoolQC = NA` → la propiedad **no tiene piscina**
- `Fence = NA` → la propiedad **no tiene cerca**

En estos casos, imputar con la moda o la media generaría un sesgo falso (como si la propiedad tuviera un garaje promedio). La solución correcta es imputar con `"None"` (o `0` en variables numéricas asociadas).

```python
cols_none = ['GarageType', 'GarageFinish', 'BsmtQual', 'PoolQC', 'Fence']
df[cols_none] = df[cols_none].fillna('None')
```

**3. Eliminación de outliers extremos**

Se eliminan viviendas con más de 4.000 ft² (`GrLivArea > 4000`) que tienen un precio anómalamente bajo. Estas propiedades son casos atípicos que distorsionan el modelo, no representan el comportamiento del mercado general.

### Estrategia de preparación ML

- **Variables nominales** (ej. `Neighborhood`): aplicar *One-Hot Encoding* con `drop_first=True` para evitar la *Dummy Variable Trap* (multicolinealidad perfecta entre columnas binarias).
- **Variables ordinales** (ej. `OverallQual`): mantener como numéricas. Ya son una escala ordenada del 1 al 10, por lo que el modelo puede interpretar su magnitud directamente.
- **Variables numéricas con outliers** (ej. `LotArea`, `GrLivArea`): usar `RobustScaler` en lugar de `StandardScaler`. El `RobustScaler` escala usando la mediana y el rango intercuartílico (IQR), lo que lo hace resistente a los valores extremos que sí existen en este dataset.

```python
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
df[['LotArea', 'GrLivArea']] = scaler.fit_transform(df[['LotArea', 'GrLivArea']])
```

> **Por qué `RobustScaler` y no `StandardScaler`:** `StandardScaler` usa la media y desviación estándar, que son sensibles a outliers. Una propiedad de 10.000 ft² jalará la media hacia arriba y distorsionará la escala de todas las demás. `RobustScaler` ignora esos extremos al basarse en percentiles.

---

## 2. MIMIC-III Clinical Database

**Fuente:** PhysioNet / MIT — Demo v1.4  
**Tipo de problema:** Clasificación / Regresión  
**Variable objetivo:** Mortalidad intrahospitalaria o `LOS` (días en UCI)

### ¿Qué contiene este dataset?

Registros clínicos reales (anonimizados) de pacientes en unidades de cuidados intensivos del MIT. Contiene notas de admisión, signos vitales, resultados de laboratorio y diagnósticos codificados en ICD-9.

### Variables principales

| Variable | Tipo | Descripción |
|---|---|---|
| `SUBJECT_ID` / `HADM_ID` | ID | Identificadores de paciente y de admisión hospitalaria |
| `LOS` | Numérica | Días totales en UCI (*Length of Stay*) |
| `ICD9_CODE` | Categórica (alta cardinalidad) | Código de diagnóstico clínico — miles de valores únicos |
| `ADMITTIME` | Fecha/hora | Momento de ingreso al hospital |

### Limpieza de datos

**1. Conversión de fechas a duración (`LOS_dias`)**

Las fechas de ingreso y egreso se convierten a `datetime` y se restan para obtener una variable numérica útil:

```python
df['ADMITTIME'] = pd.to_datetime(df['ADMITTIME'])
df['DISCHTIME'] = pd.to_datetime(df['DISCHTIME'])
df['LOS_dias'] = (df['DISCHTIME'] - df['ADMITTIME']).dt.days
```

Esto convierte un par de timestamps en una característica predictiva directa: cuántos días estuvo internado el paciente.

**2. Imputación de signos vitales faltantes con Forward Fill**

En datos clínicos longitudinales (una fila por medición por paciente), el valor más reciente conocido de un signo vital es una mejor estimación que la mediana global. *Forward Fill* arrastra el último valor válido hacia adelante en el tiempo para el mismo paciente:

```python
df = df.sort_values(['SUBJECT_ID', 'CHARTTIME'])
df[vitals] = df.groupby('SUBJECT_ID')[vitals].ffill()
```

Si no existe ningún valor previo (el paciente no tiene mediciones anteriores), se imputa con la mediana del conjunto.

**3. Reducción de cardinalidad en `ICD9_CODE`**

`ICD9_CODE` puede tener miles de valores únicos (ej. `410.11`, `410.12`, `410.9`...). Usarlos tal cual dispararía la dimensionalidad. La solución es truncar al prefijo de 3 caracteres para agrupar diagnósticos relacionados:

```python
df['ICD9_CAT'] = df['ICD9_CODE'].str[:3]
# '410.11' -> '410' (Infarto agudo de miocardio)
# '486'    -> '486' (Neumonía)
```

Luego se aplica *Target Encoding* u OHE sobre estas categorías agrupadas.

### Estrategia de preparación ML

- **`ICD9_CAT`**: Target Encoding (cada categoría se reemplaza por la media del target en ese grupo) o OHE si la cardinalidad resultante es manejable.
- **Variables clínicas numéricas**: `StandardScaler` — sus unidades son heterogéneas (presión en mmHg, glucosa en mg/dL, temperatura en °C) y necesitan estar en la misma escala.

> **Por qué agrupar ICD-9 y no usar todos los códigos:** Un modelo con miles de columnas binarias para diagnósticos raros no generaliza — memoriza. Agrupar en ~20 categorías clínicas grandes previene la *maldición de la dimensionalidad*.

---

## 3. NHANES 2021–2023 — Fusión de exámenes físicos

**Fuente:** CDC — Ciclo 2021–2023, Versión L  
**Tipo de problema:** Regresión / Clasificación  
**Variable objetivo:** Riesgo cardiovascular o fibrosis hepática

### ¿Qué contiene este dataset?

No es un único archivo — es la **unión de 4 módulos de exámenes físicos** distribuidos por el CDC en formato SAS (`.XPT`). Cada módulo cubre un dominio clínico diferente, y todos comparten la columna `SEQN` como identificador único de paciente.

### ¿Por qué los datos vienen en formato SAS (`.XPT`) y por qué todo es numérico?

El CDC distribuye los datos en formato **SAS Transport (`.XPT`)**. SAS es un software estadístico que, para ahorrar memoria a escala poblacional, nunca almacena cadenas de texto. En cambio, factoriza las categorías en diccionarios internos y las guarda como números flotantes:

- `"Prueba completada"` → `1.0`
- `"Prueba parcial"` → `2.0`
- `"No realizada"` → `3.0`

Esto significa que al abrir los archivos `.XPT` en Python, **todas las columnas ya son numéricas** — incluso las que conceptualmente son categóricas. Esto es una ventaja para ML: nos saltamos el paso de Label Encoding manual.

### Módulos del dataset

| Archivo | Variable clave | ¿Qué mide? | Tipo |
|---|---|---|---|
| `BAX_L.xpt` | `BAXSTAT` | Estado de la prueba de equilibrio | Numérica (1.0/2.0/3.0) |
| `BPXO_L.xpt` | `BPXOSY1` | Presión arterial sistólica — 1ra lectura (mmHg) | Numérica continua |
| `BMX_L.xpt` | `BMXBMI` | Índice de Masa Corporal (kg/m²) | Numérica continua |
| `LUX_L.xpt` | `LUXSMED` | Rigidez hepática mediana (kPa) — elastografía | Numérica continua |

### Cómo unir los 4 módulos en Python

Todos los archivos comparten `SEQN` como llave. Se encadenan con `merge()`:

```python
import pandas as pd

bax = pd.read_sas('BAX_L.xpt')
bpx = pd.read_sas('BPXO_L.xpt')
bmx = pd.read_sas('BMX_L.xpt')
lux = pd.read_sas('LUX_L.xpt')

df = bax.merge(bpx, on='SEQN', how='inner') \
        .merge(bmx, on='SEQN', how='inner') \
        .merge(lux, on='SEQN', how='inner')
```

> **`inner` vs `left`:** Usar `inner` conserva solo pacientes que tienen datos en los **4 módulos**. Usar `left` en el último merge (con `LUX_L`) mantiene a todos los pacientes aunque no tengan elastografía hepática — útil si `LUXSMED` es solo predictor secundario.

### Limpieza de datos

**1. Nulos en `LUX_L` (elastografía hepática)**

Este examen tiene un porcentaje alto de nulos por razones médicas: pacientes embarazadas, con marcapasos u otros implantes, o quienes rechazan el procedimiento no son elegibles. No son datos perdidos al azar — tienen una causa estructural.

- Si `LUXSMED` es la **variable objetivo**: eliminar filas con nulos (`dropna`).
- Si `LUXSMED` es un **predictor secundario**: imputar con `KNNImputer` (ver dataset 9 para detalle de la técnica).

**2. Eliminación de nulos basales**

Se eliminan filas que carezcan del IMC o de la presión sistólica, ya que son los predictores fundamentales del modelo:

```python
df = df.dropna(subset=['BMXBMI', 'BPXOSY1'])
```

**3. Binarización de `BAXSTAT`**

`BAXSTAT` tiene 3 valores (1.0 = completa, 2.0 = parcial, 3.0 = no realizada). Se binariza en "¿Se completó la prueba o no?":

```python
df['BAX_completa'] = (df['BAXSTAT'] == 1.0).astype(int)
# 1.0 -> 1 (completada)
# 2.0 o 3.0 -> 0 (no completada)
```

### Estrategia de preparación ML

- `StandardScaler` en todas las métricas físicas numéricas (`BPXOSY1`, `BMXBMI`, `LUXSMED`).
- No se aplican ponderaciones de survey del CDC (el objetivo es ML predictivo, no inferencia estadística poblacional representativa).

> **Por qué unir los 4 módulos:** Analizar cada archivo por separado pierde las correlaciones clínicas clave. La obesidad (`BMXBMI` alto) exacerba la presión arterial (`BPXOSY1` alto), lo que propicia hígado graso no alcohólico (`LUXSMED` alto), que a su vez puede afectar el sistema nervioso y alterar el equilibrio (`BAXSTAT`). El modelo necesita ver estas relaciones simultáneamente.

---

## 4. Bike Sharing Demand

**Fuente:** Capital Bikeshare / Kaggle  
**Tipo de problema:** Regresión de series temporales  
**Variable objetivo:** `count` — bicicletas rentadas en una hora determinada

### ¿Qué contiene este dataset?

Registros horarios del sistema de bicicletas compartidas de Washington D.C., con condiciones climáticas y estacionales. El objetivo es predecir cuántas bicicletas se rentarán en cada franja horaria.

### Variables principales

| Variable | Tipo | Descripción |
|---|---|---|
| `datetime` | Timestamp | Marca de tiempo horaria (`2011-01-01 00:00:00`) |
| `count` | Entera | **Variable objetivo** — bicicletas rentadas |
| `temp` | Numérica | Temperatura normalizada |
| `humidity` | Numérica | Humedad relativa |
| `windspeed` | Numérica | Velocidad del viento normalizada |
| `season` | Categórica (1–4) | Estación del año |
| `weather` | Categórica (1–4) | Condición climática |

### Limpieza de datos

**1. Brechas en la serie temporal**

Si hay horas faltantes en la secuencia (ej. salta de las 9am a las 11am), se interpola linealmente para variables continuas como temperatura:

```python
df = df.set_index('datetime').asfreq('H')
df[['temp', 'humidity', 'windspeed']] = df[['temp', 'humidity', 'windspeed']].interpolate(method='linear')
```

**2. Descomposición de `datetime`**

Un modelo de ML no puede leer `"2011-01-01 09:00:00"` como dato útil. Se descompone en columnas numéricas separadas:

```python
df['hora']  = df['datetime'].dt.hour
df['dia']   = df['datetime'].dt.dayofweek   # 0 = lunes, 6 = domingo
df['mes']   = df['datetime'].dt.month
df['anio']  = df['datetime'].dt.year
```

**3. Codificación cíclica de la hora**

La hora del día es cíclica: las 23:00 y las 00:00 son casi contiguas, pero numéricamente `23` y `0` parecen estar en extremos opuestos. Para que el modelo capture esta continuidad, se transforma con seno y coseno:

```python
import numpy as np
df['hora_sin'] = np.sin(2 * np.pi * df['hora'] / 24)
df['hora_cos'] = np.cos(2 * np.pi * df['hora'] / 24)
```

Esto convierte la hora en coordenadas de un círculo — las 23h y las 0h quedan cerca en el espacio geométrico.

**4. Log-transform del objetivo**

La distribución de `count` es muy asimétrica a la derecha: la mayoría de horas tienen pocos rentals, pero las horas pico disparan el conteo. Aplicar `log1p` "aplana" esa asimetría:

```python
df['count_log'] = np.log1p(df['count'])
# log1p = log(1 + x), evita log(0) cuando count = 0
```

Al predecir, se revierte con `np.expm1()`.

> **Por qué descomponer fechas:** Los algoritmos de ML (salvo RNNs o Transformers temporales) no tienen noción de secuencia. Exponer `hora`, `mes` y componentes cíclicos explícitamente le da al modelo la posibilidad de aprender que "los lunes a las 8am hay mucha demanda" sin que tenga que inferirlo de un timestamp crudo.

---

## 5. Adult Income Dataset (Census)

**Fuente:** UCI Machine Learning Repository  
**Tipo de problema:** Clasificación binaria  
**Variable objetivo:** `income` — ¿el ingreso anual supera los $50.000?

### ¿Qué contiene este dataset?

Datos del censo de EE.UU. de 1994. Cada fila es una persona con características demográficas y laborales. El objetivo es predecir si su ingreso supera o no los $50.000/año.

### Variables principales

| Variable | Tipo | Descripción |
|---|---|---|
| `income` | Binaria | **Variable objetivo:** `<=50K` → `0`, `>50K` → `1` |
| `workclass` | Categórica | Tipo de empleo (Private, Self-emp, Gov, etc.) |
| `education` | Categórica ordinal | Nivel educativo |
| `education-num` | Numérica | Representación numérica del nivel educativo |
| `marital-status` | Categórica | Estado civil |
| `occupation` | Categórica | Ocupación laboral |

### Limpieza de datos

**1. El problema de los archivos UCI con espacios**

Los archivos crudos del UCI ML Repository usan `", "` (coma seguida de espacio) como separador. Si se importan con `sep=','` estándar, los strings quedan con un espacio al frente (`" Private"` en lugar de `"Private"`), lo que rompe cualquier comparación o encoding posterior. La solución es usar una expresión regular al importar:

```python
df = pd.read_csv('adult.csv', sep=r',\s+', engine='python',
                 header=None, names=col_names)
```

**2. Nulos encubiertos con `'?'`**

Las columnas `workclass` y `occupation` usan el token `'?'` para indicar que el dato no fue declarado. No son errores técnicos — son personas que optaron por no revelar su situación laboral. La diferencia importa:

```python
# ❌ Mal: imputar con la moda borra el patrón
df['workclass'] = df['workclass'].replace('?', df['workclass'].mode()[0])

# ✅ Bien: convertir a categoría explícita
df['workclass']  = df['workclass'].replace('?', 'Unknown')
df['occupation'] = df['occupation'].replace('?', 'Unknown')
```

**3. Codificación del target**

```python
df['income'] = df['income'].map({'<=50K': 0, '>50K': 1})
```

### Estrategia de preparación ML

- `marital-status`, `relationship`, `race`: One-Hot Encoding (sin orden implícito).
- `education`: se puede usar directamente `education-num` (ya es ordinal con escala 1–16).
- `workclass`, `occupation`: OHE incluyendo `"Unknown"` como categoría válida.

> **Por qué no imputar el `'?'` con la moda:** Hay evidencia de que las personas con ingresos altos tienen mayor incentivo para no declarar su ocupación (privacidad). Colapsar ese `'?'` a la moda destruiría esa señal predictiva. `"Unknown"` la preserva como categoría propia con poder informativo real.

---

## 6. Credit Approval Dataset (CRX)

**Fuente:** UCI Machine Learning Repository  
**Tipo de problema:** Clasificación binaria  
**Variable objetivo:** `A16` — ¿se aprueba la solicitud de tarjeta de crédito?

### ¿Qué contiene este dataset?

Solicitudes de tarjetas de crédito con todas las variables anonimizadas (renombradas `A1`–`A16`) para proteger la confidencialidad. La mezcla de tipos de datos (numéricas y categóricas anonimizadas) requiere identificarlos por inferencia.

### Variables principales

| Variable | Tipo aparente | Descripción |
|---|---|---|
| `A1` | Categórica | Variable anónima (ej. b/a) |
| `A2`–`A15` | Mixtas | Predictores anónimos — algunos numéricos, algunos categóricos |
| `A16` | Binaria | **Variable objetivo:** `+` → `1` (aprobado), `-` → `0` (denegado) |

### Limpieza de datos

**1. Identificar tipos de columnas por Duck-Typing**

Como los nombres están anonimizados, no se sabe a priori qué columnas son numéricas. Se intenta convertir cada columna a `float` — si lanza error, es categórica:

```python
for col in df.columns:
    try:
        df[col] = df[col].astype(float)
    except (ValueError, TypeError):
        pass  # es categórica, se deja como está
```

**2. Imputación diferenciada según tipo**

- **Numéricas con `'?'`**: imputar con la **mediana** (resistente a outliers como ingresos muy altos).
- **Categóricas con `'?'`**: imputar con la **moda** (el valor más frecuente).

```python
for col in df.select_dtypes(include='number').columns:
    df[col] = df[col].fillna(df[col].median())

for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].fillna(df[col].mode()[0])
```

**3. Codificación del target**

```python
df['A16'] = df['A16'].map({'+': 1, '-': 0})
```

### Estrategia de preparación ML

- `MinMaxScaler` en todas las variables numéricas continuas.
- OHE o Label Encoding en las columnas categóricas restantes (ej. `A1`).

> **Por qué `MinMaxScaler` y no `StandardScaler`:** Las unidades de negocio son desconocidas por el anonimato — no sabemos si `A3` está en miles de dólares o en años de historial crediticio. `MinMaxScaler` lleva todo al rango [0, 1] sin asumir ninguna distribución. Esto beneficia especialmente a SVM y KNN, que son sensibles a la escala relativa entre variables.

---

## 7. Statlog Australian Credit

**Fuente:** UCI Machine Learning Repository  
**Tipo de problema:** Clasificación binaria  
**Variable objetivo:** Columna clase (`0` / `1`)

### ¿Qué contiene este dataset?

Similar al CRX, evalúa solicitudes de crédito, pero a diferencia de este, **ya viene preprocesado desde la fuente**: todas las variables categóricas fueron recodificadas como numéricas, y no hay valores faltantes declarados.

### Variables principales

| Variable | Tipo | Descripción |
|---|---|---|
| `A1`–`A14` | Numéricas | Predictores (categóricas recodificadas + numéricas originales) |
| Clase | Binaria | `0` = denegado, `1` = aprobado |

### Limpieza de datos

**1. Verificación de nulos**

El dataset está declarado como limpio, pero se verifica antes de avanzar:

```python
assert df.isnull().sum().sum() == 0, "Hay nulos inesperados"
```

**2. Clipping de outliers con Z-Score**

En lugar de eliminar filas con valores extremos (lo que reduciría el dataset), se "recortan" los valores fuera de ±3 desviaciones estándar. Esto atenúa el impacto de los extremos sin perder observaciones:

```python
from scipy import stats
import numpy as np

for col in ['A2', 'A3', 'A7']:
    z = np.abs(stats.zscore(df[col]))
    lower = df[col][z < 3].min()
    upper = df[col][z < 3].max()
    df[col] = df[col].clip(lower=lower, upper=upper)
```

**Por qué específicamente A2, A3 y A7:** Son las columnas con mayor dispersión en el dataset (alta desviación estándar), lo que indica que contienen los outliers más problemáticos para algoritmos basados en gradiente o distancia.

**3. Conversión a Float32**

```python
df = df.astype('float32')
```

Reduce el consumo de memoria a la mitad frente a `float64`, sin pérdida de precisión relevante para ML.

### Estrategia de preparación ML

- `StandardScaler` directo — todas las variables ya son numéricas continuas.
- El escalado lleva cada columna a media 0 y varianza 1, lo que acelera la convergencia de algoritmos como SVM o regresión logística.

> **Por qué clipping en lugar de eliminar filas:** Preservar todas las observaciones es preferible cuando el dataset no es enorme. El clipping "domestica" los extremos sin borrar la información de esa fila. Eliminar filas por outliers en una sola columna descartaría datos válidos en todas las demás columnas de ese registro.

---

## 8. Breast Cancer Wisconsin (Diagnostic)

**Fuente:** Dr. William H. Wolberg / UCI ML Repository  
**Tipo de problema:** Clasificación binaria  
**Variable objetivo:** `Diagnosis` — ¿el tumor es maligno o benigno?

### ¿Qué contiene este dataset?

Medidas de núcleos celulares extraídas de imágenes digitalizadas de aspirados de aguja fina (FNA) de masas mamarias. Cada fila es una paciente; cada columna describe una característica geométrica o textural de los núcleos observados.

### Variables principales

| Variable | Tipo | Descripción |
|---|---|---|
| `ID` | Identificador | Número de caso — **no aporta información predictiva** |
| `Diagnosis` | Binaria | **Variable objetivo:** `M` → `1` (maligno), `B` → `0` (benigno) |
| `Radius_mean` | Numérica | Radio promedio de los núcleos celulares |
| `Texture_mean` | Numérica | Desviación estándar de intensidades de escala de grises |
| `Perimeter_mean` | Numérica | Perímetro promedio del núcleo |
| `Bare_Nuclei` | Numérica | Proporción de núcleos sin citoplasma (~16 nulos como `'?'`) |

### Limpieza de datos

**1. Eliminar la columna `ID`**

El número de caso del paciente no tiene ningún poder predictivo — es un identificador administrativo. Si se deja, el modelo podría "memorizar" IDs y generar correlaciones espurias (sobreajuste artificial):

```python
df = df.drop(columns=['ID'])
```

**2. Manejo de nulos en `Bare_Nuclei`**

Aproximadamente 16 filas tienen `'?'` en esta columna. Se eliminan directamente (*listwise deletion*):

```python
df['Bare_Nuclei'] = pd.to_numeric(df['Bare_Nuclei'], errors='coerce')
df = df.dropna(subset=['Bare_Nuclei'])
```

16 filas representan menos del 3% del dataset total (~699 registros). La pérdida es mínima.

**3. Codificación del target**

```python
df['Diagnosis'] = df['Diagnosis'].map({'M': 1, 'B': 0})
```

### Estrategia de preparación ML

- `StandardScaler` en todas las variables numéricas.
- La multicolinealidad entre variables morfológicas es muy alta (el radio, el perímetro y el área están matemáticamente relacionados). Considerar **PCA** en fases posteriores para reducir redundancia.

> **Por qué eliminación y no imputación en `Bare_Nuclei`:** Imponer un valor artificial (ej. la media = 3.5 núcleos desnudos) a una biopsia clínica real podría crear un perfil celular que no existe en la naturaleza. En diagnóstico médico, la calidad del dato prima sobre la cantidad de filas. Perder 16 observaciones es un costo completamente aceptable frente al riesgo de entrenar con datos fabricados.

---

## 9. Meningitis Clinical Dataset

**Fuente:** Archivo médico clínico  
**Tipo de problema:** Clasificación  
**Variable objetivo:** Etiología de la meningitis (viral / bacteriana)

### ¿Qué contiene este dataset?

Resultados de análisis de fluido cefalorraquídeo (LCR) y variables demográficas de pacientes con meningitis. El diagnóstico diferencial entre meningitis viral y bacteriana es clínicamente crítico: la bacteriana requiere antibióticos urgentes; la viral, manejo sintomático.

### Variables principales

| Variable | Tipo | Descripción |
|---|---|---|
| `Leucocitos` | Numérica | Conteo de células blancas en LCR — muy elevado en meningitis bacteriana |
| `Proteínas` | Numérica | Concentración en LCR — muy elevada en meningitis bacteriana |
| `Glucosa` | Numérica | Nivel en LCR — bajo en meningitis bacteriana (las bacterias la consumen) |
| `Edad` | Numérica | Edad del paciente |
| `Género` | Categórica | Género del paciente |
| `Etiología` | Categórica | **Variable objetivo** — viral / bacteriana / fúngica |

### ¿Qué significa "Missing 12"?

Este dataset tiene lagunas **por diseño deliberado** ("Missing 12"): en condiciones clínicas reales, no siempre es posible recopilar todos los datos. Un paciente puede llegar inconsciente, o la punción lumbar puede estar contraindicada. Estos datos no faltan por error — faltan por la realidad del entorno médico. Esto se denomina técnicamente *Missing Not At Random* (MNAR).

### Limpieza de datos

**Imputación con `KNNImputer`**

La imputación por media o mediana global sería incorrecta aquí. Ejemplo concreto:

- La glucosa normal en LCR es ~60 mg/dL.
- En meningitis bacteriana puede caer a 10–20 mg/dL.
- Imputar con la media global (~50 mg/dL) a un paciente bacteriano subestimaría gravemente la severidad de su perfil.

`KNNImputer` resuelve esto buscando los 5 pacientes más similares (en todas las demás variables disponibles) y usando sus valores para imputar el dato faltante:

```python
from sklearn.impute import KNNImputer

lab_cols = ['leucocitos', 'proteinas', 'glucosa']
imputer = KNNImputer(n_neighbors=5)
df[lab_cols] = imputer.fit_transform(df[lab_cols])
```

**¿Cómo elige KNN los "vecinos más similares"?**

Calcula la distancia euclidiana entre el paciente con datos faltantes y todos los demás, usando únicamente las columnas que *sí* tienen datos. Los 5 más cercanos son los "vecinos", y el valor imputado es el promedio de ese campo en esos 5 pacientes. Si el paciente faltante es bacteriano severo, sus vecinos también lo serán, y la imputación reflejará ese perfil real.

### Estrategia de preparación ML

- `KNNImputer(n_neighbors=5)` como estrategia de imputación principal.
- Label Encoding para la variable objetivo (viral → 0, bacteriana → 1, fúngica → 2).
- OHE para síntomas categóricos observados (ej. rigidez de nuca, fotofobia).

> **Por qué KNNImputer y no media/mediana:** Un paciente con meningitis bacteriana severa tiene un perfil bioquímico completamente distinto a uno viral. Promediar con toda la sala mezclaría perfiles clínicamente opuestos, destruyendo las señales más importantes del dataset. KNN preserva la covariación biológica del individuo al imputar usando pacientes con características similares.

---

## Resumen general de estrategias

| # | Dataset | Problema | Nulos: estrategia | Scaler | Encoding principal |
|---|---|---|---|---|---|
| 1 | Ames Housing | Regresión | Mediana por grupo / `"None"` | `RobustScaler` | OHE (nominales) |
| 2 | MIMIC-III | Clasificación | Forward Fill + mediana | `StandardScaler` | Target Encoding (ICD-9 agrupado) |
| 3 | NHANES 2021–23 | Clasificación | `dropna` basales / KNN secundarios | `StandardScaler` | Binarización (`BAXSTAT`) |
| 4 | Bike Sharing | Regresión temporal | Interpolación lineal | `StandardScaler` | Sin/Cos cíclico + descomposición fecha |
| 5 | Adult Income | Clasificación | `"Unknown"` explícito | `StandardScaler` | OHE + Label (target) |
| 6 | Credit CRX | Clasificación | Mediana (numéricas) / moda (categóricas) | `MinMaxScaler` | Label Encoding + OHE |
| 7 | Statlog Credit | Clasificación | Sin nulos — clipping Z-Score ±3σ | `StandardScaler` | No requerido (ya numérico) |
| 8 | Breast Cancer | Clasificación | Listwise deletion (<3% filas) | `StandardScaler` | Binario M→1 / B→0 |
| 9 | Meningitis | Clasificación | `KNNImputer(n_neighbors=5)` | `StandardScaler` | Label + OHE (síntomas) |

---

## Guía rápida: ¿Qué scaler usar y cuándo?

| Scaler | Cuándo usarlo | ¿Sensible a outliers? |
|---|---|---|
| `StandardScaler` | Datos aproximadamente normales, sin outliers extremos | Sí — usa media y desviación estándar |
| `RobustScaler` | Datos con outliers conocidos y significativos | No — usa mediana e IQR |
| `MinMaxScaler` | Unidades desconocidas o rango necesariamente acotado [0,1] | Sí — un outlier extremo comprime todo lo demás |

---

## Guía rápida: ¿Qué hacer con los nulos?

| Situación | Estrategia recomendada |
|---|---|
| `NA` significa "no tiene ese rasgo" (ej. sin garaje) | Imputar con `"None"` o `0` |
| Nulo en serie temporal con datos previos del mismo sujeto | Forward Fill |
| Nulo en variable numérica, baja proporción de missing | Mediana (más robusta que la media ante outliers) |
| Nulo en variable categórica, baja proporción | Moda (valor más frecuente) |
| Nulo con causa conocida no aleatoria (MNAR) | `KNNImputer` — preserva relaciones entre variables |
| Nulo en variable clínica crítica y pocas filas afectadas (<3–5%) | Listwise deletion — calidad sobre cantidad |
| El dato faltante **es** la variable objetivo | Eliminar fila — nunca imputar el target |
