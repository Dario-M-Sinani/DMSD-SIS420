import json

def expand_notebook():
    file_path = 'metricas_clasificacion_pytorch.ipynb'
    with open(file_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    cells = nb['cells']

    # We will iterate through cells and expand the markdown explanations where appropriate.
    for cell in cells:
        if cell['cell_type'] == 'markdown':
            source = "".join(cell['source'])
            
            # Expand 1. Importaciones
            if '## 1. Importaciones' in source:
                cell['source'] = [
                    "## 1. Importaciones y Configuración\n",
                    "\n",
                    "A continuación importaremos las librerías necesarias. Destacamos:\n",
                    "- `pandas` y `numpy` para carga y manipulación de datos en memoria.\n",
                    "- `torch`, `torch.nn` y `torch.optim` para la creación del "
                    "Perceptrón y el cálculo del gradiente en PyTorch.\n",
                    "- `sklearn` que proveerá funciones para normalizar, dividir conjuntos "
                    "y calcular todas nuestras métricas importantes de validación clínica."
                ]
            
            # Expand 3. Preprocesamiento
            elif '## 3. Preprocesamiento' in source:
                cell['source'] = [
                    "## 3. Preprocesamiento de Datos Clinícos\n",
                    "\n",
                    "En esta etapa crítica prepararemos la data médica:\n",
                    "1. **Imputación**: Rellenaremos nulos numéricos (mediana) y categóricos (moda) para no perder pacientes.\n",
                    "2. **Binarización Objetivo**: Convertiremos `High Risk` en la clase Positiva (1) y demás en (0).\n",
                    "3. **Train/Test Split**: Guardaremos 20% para la evaluación imparcial final.\n",
                    "4. **Escalado**: Normalizamos las variables con `StandardScaler`. Esto es vital "
                    "puesto que PyTorch requiere entradas con medias similares para que el algoritmo optimizador pueda converger y encontrar el mínimo global eficientemente."
                ]

            elif '## 4. Definición del Modelo' in source:
                cell['source'] = [
                    "## 4. Definición del Modelo (Perceptrón Lineal en PyTorch)\n",
                    "\n",
                    "Implementaremos un **Perceptrón Básico Lineal**. "
                    "Recibiremos nuestras variables médicas de entrada y lo conectaremos "
                    "hacia un solo valor de salida. Finalmente, procesaremos esa salida "
                    "a través de una función de activación matemática tipo **Sigmoide**, la cual "
                    "transformará nuestra señal a una predicción que estará forzada entre el 0.0 y el 1.0 (interpretada lógicamente como una probabilidad)."
                ]
            
            elif '## 5. Entrenamiento del Modelo' in source:
                cell['source'] = [
                    "## 5. Entrenamiento Supervisado del Modelo\n",
                    "\n",
                    "Entrenar un modelo de PyTorch implica 4 grandes fases en cada vuelta `epoch`:\n",
                    "1. Hacer la pasada (`forward pass`) del lote y calcular qué resultado arroja.\n",
                    "2. Medir el error comparándolo con la realidad médica usando `BCELoss` (Binary Cross Entropy).\n",
                    "3. Calcular hacia dónde mover las variables usando la propagación hacia atrás `loss.backward()`.\n",
                    "4. Actualizar físicamente el peso de las neuronas usando el Optimizador con `optimizer.step()`."
                ]

            elif '### 6.2 Accuracy' in source:
                cell['source'] = [
                    "### 6.2 Accuracy (Exactitud Global)\n",
                    "\n",
                    "La *accuracy* cuenta el total de pacientes que atinamos de manera correcta sobre el total de la base poblacional:\n",
                    "\n",
                    "$$\\text{accuracy} = \\frac{\\text{correctos}}{\\text{total}}$$\n",
                    "\n",
                    "> ⚠️ **Cuidado con el sesgo en Medicina:** Típicamente los pacientes sanos son mayoría. Si tuviéramos 90% sanos y un modelo inútil o roto predijera siempre 'Sano', tendríamos mágicamente un \"90% de exactitud\", y simultáneamente estaríamos dejando morir a todos los enfermos graves. "
                    "Usar SOLO Accuracy causará problemas severos. Por ello pasaremos a analizar matrices y ROC."
                ]

            elif '### 6.4 Precision y Recall' in source:
                cell['source'] = [
                    "### 6.4 Precision y Recall (Sensibilidad Médica)\n",
                    "\n",
                    "$$\\text{precision} = \\frac{TP}{TP+FP} \\qquad \\text{recall} = \\frac{TP}{TP+FN}$$\n",
                    "\n",
                    "En nuestro escenario médico el dictamen se interpreta como:\n",
                    "- **Precision**: De las veces que avisamos con sirenas de urgencia que el paciente es severo, la precisión indica qué porcentaje realmente lo era. Un número bajo implica muchas \"falsas alarmas\" en sala de urgencias.\n",
                    "- **Recall (Sensibilidad)**: De todos los pacientes que mundialmente corrían real peligro de muerte o severidad por la meningitis, ¿a cuántos logró atinar nuestro modelo?. **Para escenarios médicos, maximizar el RECALL y evitar los 'Falsos Negativos' es la prioridad #1**."
                ]

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print("Notebook successfully updated.")

if __name__ == '__main__':
    expand_notebook()
