const API_PREDICT_URL = `${import.meta.env.VITE_PREDICTOR_URL}/api`;

export interface PredictionPayload {
  nombres_productos: string[];
  factor_comercial_1: number;
  factor_eventos_2: number;
  factor_estacional_3: number;
}

export interface SuccessfulPrediction {
  producto: string;
  fecha_prediccion: string;
  ventas_estimadas: number;
  rango_minimo: number;
  rango_maximo: number;
}

export interface PredictionError {
  producto: string;
  error: string;
}

export interface PredictionResponse {
  predicciones_exitosas: SuccessfulPrediction[];
  errores: PredictionError[];
}

export const predictSales = async (payload: PredictionPayload): Promise<PredictionResponse> => {
  const response = await fetch(`${API_PREDICT_URL}/predecir/`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.detail || "Error al realizar la predicción");
  }

  return response.json();
};
// En tu archivo predictionService.ts

// ... (el resto de interfaces y 'predictSales' quedan igual) ...

// --- ESTA ES LA INTERFAZ QUE DEBES ACTUALIZAR ---
export interface NotifyPayload {
  product_code: string;
  product_name: string;
  stock_actual: number;
  punto_pedido: number;
  cantidad_a_pedir: number;
  tiempo_llegada: number;
  ventas_estimadas: number; // ya la tenías
  destinatario_correo: string;
}

export const notifyStock = async (payload: NotifyPayload): Promise<{ message: string }> => {
  const response = await fetch(`${API_PREDICT_URL}/notificar/`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.error || "Error al enviar la notificación.");
  }

  return response.json();
};