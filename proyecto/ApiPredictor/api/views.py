from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import PrediccionInputSerializer
from django.core.mail import send_mail, EmailMultiAlternatives
from django.conf import settings

import pandas as pd
import pickle
import os
from prophet import Prophet # Necesario para cargar el modelo

# --- CONFIGURACIÓN ---
CARPETA_CON_MODELOS = 'modelos_entrenados'

# --- FUNCIÓN Y VISTA DE PREDICCIÓN (EL CÓDIGO QUE FALTABA) ---

def obtener_prediccion_individual(producto_nombre, data):
    """
    Carga un modelo y predice las ventas para un solo producto.
    Devuelve un diccionario con la predicción o un diccionario de error.
    """
    nombre_archivo = f"modelo_{producto_nombre}.pkl"
    ruta_modelo = os.path.join(CARPETA_CON_MODELOS, nombre_archivo)

    try:
        with open(ruta_modelo, 'rb') as f:
            modelo = pickle.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"El modelo predictivo para el producto '{producto_nombre}' no ha sido entrenado aún.")
    except Exception:
        raise Exception(f"Error interno al cargar el modelo predictivo para '{producto_nombre}'.")

    hoy = pd.Timestamp('today')
    primer_dia_mes_actual = hoy.to_period('M').to_timestamp()
    proximo_mes = (primer_dia_mes_actual + pd.DateOffset(months=1))

    df_futuro = pd.DataFrame({
        'ds': [proximo_mes],
        'Factor_Comercial_1': [data.get('factor_comercial_1', 0)],
        'Factor_Eventos_2': [data.get('factor_eventos_2', 0)],
        'Factor_Estacional_3': [data.get('factor_estacional_3', 0)]
    })

    try:
        forecast = modelo.predict(df_futuro).iloc[0]
        return {
            "producto": producto_nombre,
            "fecha_prediccion": str(forecast['ds'].date()),
            "ventas_estimadas": round(forecast['yhat'], 2),
            "rango_minimo": round(forecast['yhat_lower'], 2),
            "rango_maximo": round(forecast['yhat_upper'], 2)
        }
    except Exception:
        return {"producto": producto_nombre, "error": "Error al realizar la predicción."}


class PredecirVentaView(APIView):
    """
    Vista de la API para recibir una lista de productos y devolver sus predicciones.
    """
    def post(self, request, *args, **kwargs):
        serializer = PrediccionInputSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        validated_data = serializer.validated_data
        nombres_productos = validated_data['nombres_productos']
        
        try:
            resultados = [obtener_prediccion_individual(nombre, validated_data) for nombre in nombres_productos]
        except FileNotFoundError as e:
            return Response({"error": str(e)}, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
        predicciones_exitosas = [res for res in resultados if "error" not in res]
        errores = [res for res in resultados if "error" in res]

        respuesta_final = {
            "predicciones_exitosas": predicciones_exitosas,
            "errores": errores
        }

        return Response(respuesta_final, status=status.HTTP_200_OK)


# --- VISTA DE NOTIFICACIÓN (LA QUE YA TENÍAS) ---

class NotifyStockView(APIView):
    """
    Vista de la API para enviar una notificación por correo electrónico
    recomendando un pedido basado en el Punto de Pedido.
    """
    def post(self, request, *args, **kwargs):
        # --- 1. Recibimos el nuevo payload del frontend ---
        product_code = request.data.get('product_code')
        product_name = request.data.get('product_name')
        stock_actual = request.data.get('stock_actual')
        punto_pedido = request.data.get('punto_pedido')
        cantidad_a_pedir = request.data.get('cantidad_a_pedir')
        tiempo_llegada = request.data.get('tiempo_llegada')
        ventas_estimadas = request.data.get('ventas_estimadas')
        destinatario_correo = request.data.get('destinatario_correo')

        if not all([product_code, product_name, stock_actual, punto_pedido, cantidad_a_pedir, tiempo_llegada, ventas_estimadas, destinatario_correo]):
            return Response({"error": "Faltan datos clave para la notificación."}, status=status.HTTP_400_BAD_REQUEST)

        subject = f"Recomendación de Pedido: {product_name} ({product_code})"
        
        # --- 2. Nueva Plantilla de Texto Plano ---
        message = f"""
Estimado/a,

El Sistema de Gestión de Inventario ha detectado que es necesario realizar un pedido para el siguiente producto:

----------------------------------------------------
Producto: {product_name} ({product_code})
Stock Actual: {stock_actual} unidades
Punto de Pedido: {punto_pedido} unidades
----------------------------------------------------

El Stock Actual ({stock_actual}) ha alcanzado o está por debajo del Punto de Pedido ({punto_pedido}).

Se recomienda iniciar un proceso de compra para evitar un quiebre de stock.

Detalles Adicionales:
- Cantidad Mínima a Pedir: {cantidad_a_pedir} unidades
- Tiempo de Llegada Estimado: {tiempo_llegada} días
- Ventas Estimadas (Próx. Mes): {ventas_estimadas:.2f} unidades

Atentamente,
Sistema de Gestión de Inventario
        """

        # --- 3. Nueva Plantilla HTML ---
        html_message = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
  .container {{ max-width: 600px; margin: 0 auto; padding: 20px; border: 1px solid #ddd; border-radius: 5px; background-color: #f9f9f9; }}
  .header {{ background-color: #0056b3; color: white; padding: 10px 20px; text-align: center; border-radius: 5px 5px 0 0; }}
  .content {{ padding: 20px; }}
  .product-details {{ background-color: #e9ecef; padding: 15px; border-left: 5px solid #007bff; margin-bottom: 20px; }}
  .product-details p {{ margin: 5px 0; }}
  .footer {{ text-align: center; font-size: 0.8em; color: #777; margin-top: 20px; }}
  .alert-text {{ color: #fd7e14; font-weight: bold; }} /* Color naranja para advertencia */
</style>
</head>
<body>
  <div class="container">
    <div class="header">
      <h2>Recomendación de Pedido</h2>
    </div>
    <div class="content">
      <p>Estimado/a,</p>
      <p>El Sistema de Gestión de Inventario ha detectado que es necesario realizar un pedido para el siguiente producto:</p>
      
      <div class="product-details">
        <p><strong>Producto:</strong> {product_name} ({product_code})</p>
        <p><strong>Stock Actual:</strong> {stock_actual} unidades</p>
        <p><strong>Punto de Pedido:</strong> {punto_pedido} unidades</p>
      </div>

      <p class="alert-text">
        El Stock Actual ({stock_actual}) ha alcanzado o está por debajo del Punto de Pedido ({punto_pedido}).
      </d>

      <p>Se recomienda iniciar un proceso de compra para este producto a la brevedad.</p>
      
      <div class="product-details" style="border-left-color: #6c757d; background-color: #f8f9fa;">
        <p><strong>Detalles para el Pedido:</strong></p>
        <p><strong>Cantidad Mínima a Pedir:</strong> {cantidad_a_pedir} unidades</p>
        <p><strong>Tiempo de Llegada Estimado:</strong> {tiempo_llegada} días</p>
        <p><strong>Ventas Estimadas (Próx. Mes):</strong> {ventas_estimadas:.2f} unidades</p>
      </div>

      <p>Atentamente,</p>
      <p><strong>Sistema de Gestión de Inventario</strong></p>
    </div>
    <div class="footer">
      <p>Este es un mensaje automático, por favor no responda a este correo.</p>
    </div>
  </div>
</body>
</html>
        """

        from_email = settings.DEFAULT_FROM_EMAIL
        recipient_list = [destinatario_correo]

        try:
            msg = EmailMultiAlternatives(subject, message, from_email, recipient_list)
            msg.attach_alternative(html_message, "text/html")
            msg.send(fail_silently=False)
            return Response({"message": "Correo de notificación enviado con éxito."}, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({"error": f"Error al enviar el correo: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)