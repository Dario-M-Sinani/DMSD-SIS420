from django.urls import path
from .views import PredecirVentaView, NotifyStockView

urlpatterns = [
    # Rutas para el endpoint de predicción (con y sin barra para evitar 404/301 en POST)
    path('predecir/', PredecirVentaView.as_view(), name='predecir-venta'),
    path('predecir', PredecirVentaView.as_view(), name='predecir-venta-no-slash'),
    
    # Rutas para el endpoint de notificaciones
    path('notificar/', NotifyStockView.as_view(), name='notificar-stock'),
    path('notificar', NotifyStockView.as_view(), name='notificar-stock-no-slash'),
]
