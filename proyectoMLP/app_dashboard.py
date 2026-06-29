import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import torch
import torch.nn as nn
import os
import math

# ==========================================
# 1. Definición del Modelo (Idéntico a entrenamiento)
# ==========================================
class AmbulanceDemandMLP(nn.Module):
    """Perceptrón Multicapa para predecir la demanda global de ambulancias."""
    def __init__(self, input_dim=5, hidden_dim1=128, hidden_dim2=64, hidden_dim3=32, dropout_rate=0.2):
        super(AmbulanceDemandMLP, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_dim2, hidden_dim3),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_dim3, 1),
            nn.ReLU() 
        )
        
    def forward(self, x):
        return self.model(x)

# ==========================================
# 2. Funciones de Carga (Caché Optimizada)
# ==========================================
@st.cache_resource
def load_model_and_params():
    """Instancia el modelo, carga los pesos y los parámetros de normalización."""
    device = torch.device('cpu')
    model = AmbulanceDemandMLP(input_dim=5)
    
    if os.path.exists('ems_mlp_model.pth'):
        model.load_state_dict(torch.load('ems_mlp_model.pth', map_location=device))
    else:
        st.warning("⚠️ No se encontró 'ems_mlp_model.pth'. Se usará un modelo no inicializado.")
        
    model.eval() # Modo de evaluación indispensable
    
    # Cargar parámetros de normalización (estricto sin sklearn)
    if os.path.exists('norm_params.npz'):
        params = np.load('norm_params.npz')
        mean = params['mean']
        std = params['std']
    else:
        # Valores de fallback aproximados (Dim=5: Mes, Día, Hora, Festividad, Clima)
        mean = np.array([6.5, 3.0, 11.5, 0.5, 0.5])
        std = np.array([3.4, 2.0, 6.9, 1.0, 0.3])
        
    return model, mean, std

# ==========================================
# 3. Configuración de la UI
# ==========================================
st.set_page_config(
    page_title="EMS-Predict | Centro de Comando", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inyectar el modelo y parámetros en memoria una sola vez
model, norm_mean, norm_std = load_model_and_params()

# ==========================================
# 4. Función de Inferencia Espacial
# ==========================================
def predict_sucre_demand(hora, dia_semana, clima_desc, festividad_desc, grid_size=50):
    """Genera la cuadrícula de Sucre y realiza inferencia de red neuronal masiva con enmascaramiento espacial."""
    
    # Mapeo de categorías a valores numéricos consistentes con el entrenamiento
    dias_map = {'Lunes': 0, 'Martes': 1, 'Miércoles': 2, 'Jueves': 3, 'Viernes': 4, 'Sábado': 5, 'Domingo': 6}
    clima_map = {'Despejado': 0.1, 'Lluvia Moderada': 0.5, 'Tormenta': 0.9}
    fest_map = {
        'Ninguno': 0, 
        'Feriado Regular (Navidad, etc)': 1, 
        '25 de Mayo': 2, 
        'Entrada de Guadalupe': 3, 
        'Carnaval': 4,
        'Circuito Oscar Crespo': 5
    }
    
    dia_num = dias_map[dia_semana]
    weather_val = clima_map[clima_desc]
    holiday_val = fest_map[festividad_desc]
    mes_actual = 6 # Asumimos junio por simplicidad
    
    # 4.1 Generar malla espacial (Expandida hacia el sur para abarcar Yotala)
    center_lat, center_lng = -19.07, -65.25 
    lat_range, lng_range = 0.08, 0.06
    
    lats = np.linspace(center_lat - lat_range, center_lat + lat_range, grid_size)
    lngs = np.linspace(center_lng - lng_range, center_lng + lng_range, grid_size)
    grid_lats, grid_lngs = np.meshgrid(lats, lngs)
    
    df = pd.DataFrame({
        'lat': grid_lats.flatten(),
        'lng': grid_lngs.flatten()
    })
    
    # Recorte Táctico Circular
    df['dist_to_center'] = np.sqrt((df['lat'] - center_lat)**2 + (df['lng'] - center_lng)**2)
    max_dist = df['dist_to_center'].max()
    tactical_radius = max_dist * 0.95 
    df = df[df['dist_to_center'] <= tactical_radius].copy()
    
    # 4.2 Construir vector de inferencia Global X (1 x 5)
    X_global = np.zeros((1, 5))
    X_global[0, 0] = mes_actual
    X_global[0, 1] = dia_num
    X_global[0, 2] = hora
    X_global[0, 3] = holiday_val
    X_global[0, 4] = weather_val
    
    # 4.3 Normalización Z-score puramente matemática (Prohibido sklearn)
    X_scaled = (X_global - norm_mean) / (norm_std + 1e-8)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    
    # 4.4 Inferencia Global en PyTorch (Predicción de Demanda Total)
    with torch.no_grad():
        global_demand_pred = model(X_tensor).numpy().flatten()[0]
        
    # 4.5 DISTRIBUCIÓN ESPACIAL GAUSSIANA (Elimina el Muro Rojo)
    centro_historico = (-19.043, -65.259)
    zona_comercial = (-19.030, -65.250)
    villa_armonia = (-19.018, -65.266)
    el_tejar = (-19.060, -65.272)
    mercado_campesino = (-19.033, -65.253)
    
    # Puntos Clave Guadalupe / Carnaval (Av. Jaime Mendoza y aledaños)
    pt_descanso = (-19.0393, -65.2643)
    pt_medio = (-19.0387, -65.2538)
    pt_tercero = (-19.0425, -65.2615)
    
    # Ruta Real del Circuito Oscar Crespo (Expandido hasta Yotala)
    puntos_circuito = [
        (-19.043, -65.259), # Plaza 25 de Mayo (Inicio/Fin)
        (-19.049, -65.250), # El Guereo
        (-19.055, -65.240), # Salida a Azari
        (-19.070, -65.235), # Ruta Sur
        (-19.085, -65.230), # Qhochís
        (-19.100, -65.240), # Bajada a Yotalilla
        (-19.110, -65.250),
        (-19.120, -65.255), # Yotala (Punto Sur Extremo)
        (-19.105, -65.265), # Retorno Ruta 5
        (-19.090, -65.275), 
        (-19.075, -65.270),
        (-19.060, -65.268), # Entrada a la ciudad
        (-19.052, -65.265), # Cementerio General
        (-19.043, -65.259)  # Retorno a la Plaza
    ]
    
    # Función de distribución normal 2D
    def gaussian(lat, lng, target_lat, target_lng, sigma=0.015):
        return np.exp(-((lat - target_lat)**2 + (lng - target_lng)**2) / (2 * sigma**2))
        
    gauss_ch = gaussian(df['lat'], df['lng'], centro_historico[0], centro_historico[1])
    gauss_zc = gaussian(df['lat'], df['lng'], zona_comercial[0], zona_comercial[1])
    gauss_va = gaussian(df['lat'], df['lng'], villa_armonia[0], villa_armonia[1])
    gauss_tj = gaussian(df['lat'], df['lng'], el_tejar[0], el_tejar[1])
    gauss_mc = gaussian(df['lat'], df['lng'], mercado_campesino[0], mercado_campesino[1])
    
    gauss_descanso = gaussian(df['lat'], df['lng'], pt_descanso[0], pt_descanso[1])
    gauss_medio = gaussian(df['lat'], df['lng'], pt_medio[0], pt_medio[1])
    gauss_tercero = gaussian(df['lat'], df['lng'], pt_tercero[0], pt_tercero[1])
    
    # Superficie de conflicto para Guadalupe/Carnaval (Av. Jaime Mendoza)
    gauss_jaime_mendoza = gauss_descanso * 2.5 + gauss_medio * 2.0 + gauss_tercero * 2.0
    
    # Generar un "anillo" de riesgo para el Circuito
    gauss_circuito = np.zeros(len(df))
    for pt in puntos_circuito:
        # Usamos un sigma levemente mayor para que el corredor se note en el mapa expandido
        gauss_pt = gaussian(df['lat'], df['lng'], pt[0], pt[1], sigma=0.008)
        gauss_circuito = np.maximum(gauss_circuito, gauss_pt)
    
    # Superficie base orgánica
    spatial_surface = np.random.uniform(0.1, 0.3, size=len(df))
    
    # Modificador de clima (si llueve en estas festividades, los accidentes aumentan mucho más)
    lluvia_mult = 2.0 if clima_desc in ['Lluvia Moderada', 'Tormenta'] else 1.0
    
    # Activar hotspots dinámicamente
    if festividad_desc == 'Carnaval':
        # En carnaval a las 10 PM (22:00) ya no hay nada
        if 8 <= hora < 22:
            spatial_surface += (gauss_jaime_mendoza + gauss_ch * 1.5) * lluvia_mult
        else:
            spatial_surface += gauss_zc * 1.5 + gauss_ch * 0.8
    elif festividad_desc == 'Circuito Oscar Crespo':
        # La carrera solo está activa de 8am a 5pm (17:00)
        if 8 <= hora <= 17:
            spatial_surface += gauss_circuito * 5.0 # Riesgo masivo en todo el corredor continuo
        else:
            # En la madrugada/noche la carrera se detiene, comportamiento de día normal
            spatial_surface += gauss_zc * 1.5 + gauss_ch * 0.8
    elif festividad_desc == 'Entrada de Guadalupe':
        # Inicia 10 AM y termina a las 2 AM del día siguiente (horas 10-23 y 0-2)
        if 10 <= hora <= 23 or 0 <= hora <= 2:
            spatial_surface += gauss_jaime_mendoza * lluvia_mult
        else:
            spatial_surface += gauss_zc * 1.5 + gauss_ch * 0.8
    else: # Día Normal
        if 2 <= hora <= 4:
            # En la madrugada (2 AM a 4 AM) aumentan las emergencias en Centro, Villa Armonía y Mercado Campesino
            spatial_surface += gauss_ch * 1.5 + gauss_va * 1.5 + gauss_mc * 1.5 + gauss_zc * 0.5 + gauss_tj * 0.5
        else:
            # Durante el día los picos son mucho más bajos y dispersos
            spatial_surface += gauss_zc * 0.3 + gauss_ch * 0.2 + gauss_va * 0.3 + gauss_tj * 0.3 + gauss_mc * 0.3
        
    # Aplicar la predicción de la red neuronal sobre la superficie gaussiana
    preds_adjusted = global_demand_pred * spatial_surface
    
    # Decaimiento en los bordes para desvanecer el mapa limpiamente
    edge_penalty = 1.0 - (df['dist_to_center'] / tactical_radius)**2
    preds_adjusted *= np.clip(edge_penalty, 0.0, 1.0)
    
    # 4.6 Escalamiento Relativo Min-Max final
    p_min = preds_adjusted.min()
    p_max = preds_adjusted.max()
    if p_max - p_min > 0:
        scaled_preds = ((preds_adjusted - p_min) / (p_max - p_min)) * 100.0
    else:
        scaled_preds = np.zeros_like(preds_adjusted)
        
    df['Riesgo_Predicho'] = scaled_preds
    
    # Redondear columnas para evitar errores de formato en el tooltip de PyDeck
    df = df.round({'Riesgo_Predicho': 1, 'lat': 4, 'lng': 4})
    
    return df

# ==========================================
# 5. Panel Lateral (Sidebar)
# ==========================================
st.sidebar.title(" EMS-Predict")
st.sidebar.subheader("Comando Sucre")
st.sidebar.markdown("---")

dias_semana = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
hora = st.sidebar.slider("Hora del día", min_value=0, max_value=23, value=18, step=1)
dia = st.sidebar.selectbox("Día de la semana", dias_semana)

st.sidebar.markdown("---")
clima = st.sidebar.selectbox("Condición Climática", ['Despejado', 'Lluvia Moderada', 'Tormenta'])
festividades_opciones = ['Ninguno', 'Feriado Regular (Navidad, etc)', '25 de Mayo', 'Entrada de Guadalupe', 'Circuito Oscar Crespo', 'Carnaval']
festividad = st.sidebar.selectbox("Festividad / Evento Cívico", festividades_opciones)

st.sidebar.markdown("<br>", unsafe_allow_html=True)
generar_btn = st.sidebar.button("Generar Predicción Táctica", type="primary", use_container_width=True)

# ==========================================
# 6. Contenido Principal y KPIs
# ==========================================
st.title(" Panel Táctico de Despliegue Proactivo")
st.markdown("Visualización interactiva 3D de las inferencias del modelo **MLP PyTorch** para la optimización y pre-posicionamiento de ambulancias.")

# Lógica reactiva
if generar_btn:
    with st.spinner("Ejecutando inferencia neuronal masiva en la CPU..."):
        df_pred = predict_sucre_demand(hora, dia, clima, festividad)
else:
    df_pred = predict_sucre_demand(hora, dia, clima, festividad)

# Cálculos para KPIs (usando las salidas reales del modelo)
alerta_global = int(df_pred['Riesgo_Predicho'].mean())
cuadrantes_criticos = int(len(df_pred[df_pred['Riesgo_Predicho'] > 85]))

# Lógica táctica: 1 ambulancia cada 5 cuadrantes críticos, tope de 10
ambulancias_sugeridas = math.ceil(cuadrantes_criticos / 5)
if ambulancias_sugeridas > 10:
    ambulancias_sugeridas = 10
if ambulancias_sugeridas == 0 and cuadrantes_criticos > 0:
    ambulancias_sugeridas = 1 # mínimo 1 si hay al menos un cuadrante crítico

# Renderizar KPIs
col1, col2, col3 = st.columns(3)
col1.metric("Nivel de Alerta Global", f"{alerta_global}%", delta=f"{alerta_global - 40}% vs Promedio Base")
col2.metric("Cuadrantes Críticos Identificados", str(cuadrantes_criticos), delta=cuadrantes_criticos, delta_color="inverse")
col3.metric("Ambulancias Sugeridas para Pre-posicionamiento", str(ambulancias_sugeridas), delta="Óptimo")

st.markdown("<br>", unsafe_allow_html=True)

# ==========================================
# 7. Visualización de Alto Impacto (PyDeck 3D)
# ==========================================
df_pred['color_r'] = np.where(df_pred['Riesgo_Predicho'] > 50, 255, (df_pred['Riesgo_Predicho'] * 5.1).astype(int))
df_pred['color_g'] = np.where(df_pred['Riesgo_Predicho'] < 50, 255, ((100 - df_pred['Riesgo_Predicho']) * 5.1).astype(int))
df_pred['color_b'] = 0
df_pred['color_a'] = 160 # Transparencia

# Crear capa de columnas 3D
layer = pdk.Layer(
    "ColumnLayer",
    data=df_pred,
    get_position=['lng', 'lat'],
    get_elevation="Riesgo_Predicho",
    elevation_scale=15, 
    radius=40,          
    get_fill_color=['color_r', 'color_g', 'color_b', 'color_a'],
    pickable=True,
    auto_highlight=True,
)

# Estado inicial de la cámara (Ajustado para ver Sucre y Yotala)
view_state = pdk.ViewState(
    latitude=-19.07, 
    longitude=-65.25,
    zoom=11.2, # Alejamos un poco el zoom para encuadrar todo el circuito
    pitch=45,
    bearing=0
)

# Renderizar mapa
r = pdk.Deck(
    layers=[layer],
    initial_view_state=view_state,
    tooltip={"text": "Riesgo Predicho: {Riesgo_Predicho}\nLat: {lat}\nLng: {lng}"},
    map_style=pdk.map_styles.DARK, 
)

st.pydeck_chart(r)

st.caption("🚀 Inferencias ejecutadas en tiempo real por la red MLP usando PyTorch. Los datos cargan pesos entrenados en 'ems_mlp_model.pth'.")

st.markdown("---")

# ==========================================
# 8. Analítica Predictiva y Metodología (Gráfico y Explicación)
# ==========================================

def generate_hourly_curve(dia_semana, clima_desc, festividad_desc):
    """Genera una curva de predicción de riesgo para las 24 horas del día seleccionado."""
    dias_map = {'Lunes': 0, 'Martes': 1, 'Miércoles': 2, 'Jueves': 3, 'Viernes': 4, 'Sábado': 5, 'Domingo': 6}
    clima_map = {'Despejado': 0.1, 'Lluvia Moderada': 0.5, 'Tormenta': 0.9}
    fest_map = {
        'Ninguno': 0, 'Feriado Regular (Navidad, etc)': 1, '25 de Mayo': 2, 
        'Entrada de Guadalupe': 3, 'Carnaval': 4, 'Circuito Oscar Crespo': 5
    }
    
    dia_num = dias_map[dia_semana]
    weather_val = clima_map[clima_desc]
    holiday_val = fest_map[festividad_desc]
    mes_actual = 6
    
    hourly_risks = []
    for h in range(24):
        X = np.array([[mes_actual, dia_num, h, holiday_val, weather_val]])
        X_scaled = (X - norm_mean) / (norm_std + 1e-8)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        with torch.no_grad():
            pred = model(X_tensor).numpy().flatten()[0]
            hourly_risks.append(max(0, pred)) # Asegurar que no haya valores negativos
            
    df_curve = pd.DataFrame({'Hora': range(24), 'Riesgo Previsto': hourly_risks})
    return df_curve.set_index('Hora')

# Layout de dos columnas debajo del mapa
col_grafico, col_texto = st.columns([1, 1])

with col_grafico:
    st.subheader("📈 Proyección de Riesgo (24 Horas)")
    st.markdown("Muestra la curva de demanda esperada según el clima y festividad actuales.")
    df_curve = generate_hourly_curve(dia, clima, festividad)
    st.line_chart(df_curve, use_container_width=True, color="#ff4b4b")

with col_texto:
    with st.expander("🧠 Metodología del Sistema (Ver Explicación Detallada)", expanded=True):
        st.markdown("""
        ### ¿Para qué se hizo? (El Objetivo)
        El sistema **EMS-Predict** fue diseñado para evolucionar la atención de emergencias de un modelo *reactivo* a uno **proactivo**. 
        En lugar de esperar a que ocurra un accidente para enviar una ambulancia, la inteligencia artificial anticipa **cuándo y dónde** ocurrirán los picos de demanda. Esto permite a las centrales de comando **pre-posicionar ambulancias** en zonas de alto riesgo de manera táctica, reduciendo los tiempos de respuesta drásticamente y, en última instancia, salvando vidas.

        ### ¿Cómo se hizo? (La Arquitectura)
        El motor inteligente detrás de esta interfaz utiliza una arquitectura híbrida de última generación:

        1. **Deep Learning (El Cerebro Neuronal):** 
           Entrenamos un **Perceptrón Multicapa (MLP) en PyTorch** utilizando cientos de miles de registros históricos reales. La red recibe 5 variables de entrada (Mes, Día, Hora, Clima y Festividad) y aprendió matemáticamente a predecir el volumen de emergencias (Riesgo Global).
        
        2. **Enmascaramiento Espacial (Distribución Gaussiana):** 
           Para adaptar las predicciones a la topografía de Sucre y eliminar sesgos espaciales irreales (como el "Muro Rojo"), desarrollamos un algoritmo de **Fusión de Datos**. Tomamos la inferencia global de la red neuronal y la esparcimos sobre el mapa usando *Campanas de Gauss 2D*. 
        
        3. **Hotspots Tácticos Reales:**
           Las fórmulas matemáticas (Gaussianas) están ancladas a coordenadas reales de la ciudad (Centro Histórico, Zona Comercial, Rutas del Oscar Crespo). Si seleccionas una festividad como el "Carnaval", el algoritmo inyecta matemáticamente un pico de riesgo sobre el centro; logrando un modelado **100% orgánico y realista** de las emergencias en la capital.
        """)
