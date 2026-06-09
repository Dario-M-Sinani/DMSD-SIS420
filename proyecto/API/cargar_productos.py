import os
import django

# 1. Configurar el entorno de Django para poder usar los modelos fuera del servidor
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'api_project.settings') # Ajusta 'api_project' si tu carpeta principal se llama diferente
django.setup()

from productos.models import Producto

def cargar_catalogo_apicola():
    print("Iniciando la carga del catálogo apícola en la base de datos...")
    
    # Asegurar que esté vacío antes de cargar
    Producto.objects.all().delete()
    
    # El diccionario con los datos exactos que coinciden con la IA
    productos_agro = [
        {'codigo': 'SINCROCIP', 'nombre': 'Cera Estampada de Abeja (Pack x50)', 'stock': 550, 'precio': 45.00, 'lead_time': 15, 'min_compra': 10},
        {'codigo': 'SINCROFORTE', 'nombre': 'Suplemento Proteico Colmenas 20kg', 'stock': 250, 'precio': 65.00, 'lead_time': 7, 'min_compra': 5},
        {'codigo': 'RESOLUTOR', 'nombre': 'Tratamiento Orgánico Varroa', 'stock': 110, 'precio': 120.00, 'lead_time': 5, 'min_compra': 2},
        {'codigo': 'OUROVIT-500', 'nombre': 'Nutriente Foliar Estimulante 5L', 'stock': 250, 'precio': 85.00, 'lead_time': 10, 'min_compra': 4},
        {'codigo': 'CIPROLAC', 'nombre': 'Sanitizante de Material Apícola', 'stock': 500, 'precio': 35.00, 'lead_time': 4, 'min_compra': 5},
        {'codigo': 'SUPERHION-5L', 'nombre': 'Control Ecológico de Plagas (5L)', 'stock': 500, 'precio': 150.00, 'lead_time': 12, 'min_compra': 2},
        {'codigo': 'IMPACTO-25', 'nombre': 'Repelente de Ácaros Concentrado', 'stock': 750, 'precio': 18.00, 'lead_time': 5, 'min_compra': 10},
        {'codigo': 'IMPACTO-100', 'nombre': 'Repelente de Ácaros Concentrado (Medio)', 'stock': 520, 'precio': 40.00, 'lead_time': 5, 'min_compra': 5},
        {'codigo': 'DOXIFIN', 'nombre': 'Antibiótico Apícola Loque', 'stock': 420, 'precio': 55.00, 'lead_time': 8, 'min_compra': 3},
        {'codigo': 'OUROVAC-AFT', 'nombre': 'Envases de Vidrio para Miel (Caja x100)', 'stock': 1500, 'precio': 25.00, 'lead_time': 20, 'min_compra': 5},
        {'codigo': 'SUPERHION-1L', 'nombre': 'Control Ecológico de Plagas (1L)', 'stock': 635, 'precio': 38.00, 'lead_time': 6, 'min_compra': 5},
        {'codigo': 'OUROVIT-250', 'nombre': 'Nutriente Foliar Estimulante 2.5L', 'stock': 740, 'precio': 48.00, 'lead_time': 10, 'min_compra': 5},
        {'codigo': 'OUROTETRA', 'nombre': 'Sanitizador de Suelos Orgánico', 'stock': 1400, 'precio': 95.00, 'lead_time': 14, 'min_compra': 2},
        {'codigo': 'IMPACTO-1000', 'nombre': 'Repelente de Ácaros Industrial (1L)', 'stock': 550, 'precio': 220.00, 'lead_time': 15, 'min_compra': 10}
    ]

    for prod in productos_agro:
        Producto.objects.create(
            codigo=prod['codigo'],
            nombre=prod['nombre'],
            stock=prod['stock'],
            costo=prod['precio'] * 0.7, # Costo estimado del 70%
            precio_venta=prod['precio'],
            pais="Bolivia",
            tiempo_llegada=prod['lead_time'],
            cantidad_minima_compra=prod['min_compra']
        )
        print(f"  -> {prod['nombre']} guardado correctamente.")

    print("\n¡Catálogo apícola cargado exitosamente en SQLite!")

if __name__ == '__main__':
    cargar_catalogo_apicola()