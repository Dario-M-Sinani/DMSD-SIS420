import csv
from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from .models import Producto
from .serializers import ProductoSerializer

class ProductoViewSet(viewsets.ModelViewSet):
    """
    API endpoint that allows users to be viewed or edited.
    """
    queryset = Producto.objects.all().order_by('-id')
    serializer_class = ProductoSerializer
    
    @action(detail=False, methods=['post'], url_path='cargar-csv', parser_classes=[MultiPartParser, FormParser])
    def cargar_csv(self, request):
        if 'file' not in request.FILES:
            return Response({"error": "No se envió ningún archivo."}, status=status.HTTP_400_BAD_REQUEST)
        
        csv_file = request.FILES['file']
        
        if not csv_file.name.endswith('.csv'):
            return Response({"error": "El archivo debe tener formato .csv"}, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            decoded_file = csv_file.read().decode('utf-8').splitlines()
            reader = csv.DictReader(decoded_file)
            
            productos_creados = 0
            for row in reader:
                codigo = row.get('codigo')
                if not codigo:
                    continue
                
                # Update or create the product based on 'codigo'
                Producto.objects.update_or_create(
                    codigo=codigo,
                    defaults={
                        'nombre': row.get('nombre', ''),
                        'stock': int(row.get('stock_actual', 0)),
                        'precio_venta': float(row.get('precio_venta', 0.0)),
                        'tiempo_llegada': int(row.get('tiempo_llegada', 0)),
                        'cantidad_minima_compra': int(row.get('cantidad_minima', 0)),
                        'costo': float(row.get('precio_venta', 0.0)) * 0.7,
                        'pais': "Bolivia"
                    }
                )
                productos_creados += 1
                
            return Response({"message": f"Se cargaron {productos_creados} productos exitosamente."}, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({"error": f"Error al procesar el archivo: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)