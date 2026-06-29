import json
import os

nb_file = "EMS_Predict_Fase1.ipynb"

with open(nb_file, 'r', encoding='utf-8') as f:
    nb = json.load(f)

def create_markdown_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\\n" if not line.endswith("\\n") else line for line in source.split("\\n")]
    }

def create_code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\\n" if not line.endswith("\\n") else line for line in source.split("\\n")]
    }

# 1. Explicación de la Arquitectura
md_text1 = """## 10. Arquitectura de Despliegue: Del Modelo Reactivo a la Fusión Espacial Proactiva

### ¿Para qué se hizo? (El Objetivo)
El objetivo fundamental de **EMS-Predict** no es solo tener un número estadístico, sino construir una herramienta táctica. Evolucionamos la atención de emergencias de un modelo *reactivo* a uno **proactivo**. En lugar de esperar a que ocurra un accidente para despachar una ambulancia, la inteligencia artificial anticipa **cuándo y dónde** ocurrirán los picos de demanda. Esto permite a las centrales de comando **pre-posicionar ambulancias**, reduciendo los tiempos de respuesta y salvando vidas.

### El Problema del Sesgo Espacial (El "Muro Rojo")
Durante las primeras iteraciones, el modelo recibía la variable geográfica (`Zone_ID`) directamente. Esto generaba un sesgo masivo donde la red aprendía que los números de zona más altos (los bordes del mapa) tenían intrínsecamente más demanda debido a los datos originales de EE.UU.

### ¿Cómo se resolvió? (Hibridación y Fusión de Datos)
Desarrollamos una arquitectura híbrida de dos capas:

1. **El Cerebro Neuronal (PyTorch MLP):** Entrenado en este notebook con 5 variables (Mes, Día, Hora, Clima, Festividad). Su única misión es predecir la **Demanda Global** de la ciudad, abstrayéndose de la geografía y enfocándose en la temporalidad/eventos.
2. **Enmascaramiento Espacial (Distribución Gaussiana):** Durante la inferencia táctica (en la aplicación Streamlit), tomamos la predicción global y la esparcimos sobre Sucre usando *Campanas de Gauss 2D*. Las ecuaciones matemáticas están ancladas a las coordenadas reales de los *Hotspots* (Centro Histórico, Zona Comercial, Ruta Oscar Crespo)."""

# 2. Visualización de Curva
md_text2 = """## 11. Visualización de la Curva de Demanda Temporal (24 Horas)

A continuación, programaremos la función que el Dashboard utiliza en tiempo real para generar la curva predictiva para las 24 horas de un escenario específico."""

code_text1 = """import matplotlib.pyplot as plt

def plot_hourly_curve(dia_num, fest_val, weather_val, mes_actual=6):
    hourly_risks = []
    # Usamos el modelo previamente instanciado 'loaded_model'
    for h in range(24):
        X = np.array([[mes_actual, dia_num, h, fest_val, weather_val]], dtype=np.float32)
        X_scaled = (X - mean_vals) / std_vals
        X_tensor = torch.tensor(X_scaled)
        
        with torch.no_grad():
            pred = loaded_model(X_tensor).numpy().flatten()[0]
            hourly_risks.append(max(0, pred))
            
    plt.figure(figsize=(10, 5))
    plt.plot(range(24), hourly_risks, marker='o', color='crimson', linewidth=2)
    plt.title("Proyección de Riesgo (Curva de 24 Horas)")
    plt.xlabel("Hora del Día")
    plt.ylabel("Riesgo Predicho (Demanda Global)")
    plt.grid(alpha=0.3)
    plt.xticks(range(0, 24))
    plt.show()

# Probamos con Día Sábado(5), Carnaval(4) y Clima Despejado(0.1)
plot_hourly_curve(dia_num=5, fest_val=4, weather_val=0.1)"""

# 3. Distribución Espacial Gaussiana
md_text3 = """## 12. Simulación de Fusión de Datos y Enmascaramiento Espacial

El siguiente bloque de código recrea de forma gráfica cómo se distribuye la demanda global sobre un mapa usando funciones Gaussianas ancladas a Hotspots reales de Sucre, eliminando por completo el sesgo en los perímetros."""

code_text2 = """# Definición de Hotspots en Sucre
centro_historico = (-19.043, -65.259)
zona_comercial = (-19.030, -65.250)

# Crear grilla espacial sintética simulando la topografía de la ciudad
grid_size = 50
center_lat, center_lng = -19.04, -65.26
lats = np.linspace(center_lat - 0.05, center_lat + 0.05, grid_size)
lngs = np.linspace(center_lng - 0.05, center_lng + 0.05, grid_size)
grid_lats, grid_lngs = np.meshgrid(lats, lngs)

# Distribución normal bidimensional
def gaussian_2d(lat, lng, target_lat, target_lng, sigma=0.015):
    return np.exp(-((lat - target_lat)**2 + (lng - target_lng)**2) / (2 * sigma**2))

# Generar máscaras espaciales
gauss_ch = gaussian_2d(grid_lats, grid_lngs, centro_historico[0], centro_historico[1])
gauss_zc = gaussian_2d(grid_lats, grid_lngs, zona_comercial[0], zona_comercial[1])

# Fusión: Para Carnaval, el riesgo se dispara en el Centro Histórico
spatial_surface = np.random.uniform(0.1, 0.3, size=(grid_size, grid_size)) # Ruido base orgánico
spatial_surface += gauss_ch * 3.0 + gauss_zc * 0.5 # Impacto del Carnaval

# Aplicar decaimiento en los bordes (Táctica Radar)
dist_to_center = np.sqrt((grid_lats - center_lat)**2 + (grid_lngs - center_lng)**2)
edge_penalty = 1.0 - (dist_to_center / dist_to_center.max())**2
spatial_surface *= np.clip(edge_penalty, 0.0, 1.0)

# Predicción Global (Demanda base, ej: 15 incidentes)
demanda_global = 15.0
mapa_riesgo_final = demanda_global * spatial_surface

plt.figure(figsize=(9, 7))
plt.contourf(grid_lngs, grid_lats, mapa_riesgo_final, cmap='hot', levels=20)
plt.colorbar(label='Nivel de Riesgo (Despliegue)')
plt.scatter([centro_historico[1]], [centro_historico[0]], color='cyan', marker='*', s=200, label='Centro Histórico')
plt.scatter([zona_comercial[1]], [zona_comercial[0]], color='lime', marker='*', s=200, label='Zona Comercial')
plt.title("Enmascaramiento Espacial Táctico (Mapa de Calor de Carnaval)")
plt.xlabel("Longitud")
plt.ylabel("Latitud")
plt.legend()
plt.show()"""

nb['cells'].extend([
    create_markdown_cell(md_text1),
    create_markdown_cell(md_text2),
    create_code_cell(code_text1),
    create_markdown_cell(md_text3),
    create_code_cell(code_text2)
])

with open(nb_file, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook actualizado con éxito usando json nativo.")
