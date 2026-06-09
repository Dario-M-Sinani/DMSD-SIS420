from django.db import models

class Producto(models.Model):
    codigo = models.CharField(max_length=50, unique=True)
    nombre = models.CharField(max_length=100)
    descripcion = models.TextField(blank=True, null=True)
    stock = models.PositiveIntegerField()
    costo = models.DecimalField(max_digits=10, decimal_places=2)
    precio_venta = models.DecimalField(max_digits=10, decimal_places=2)
    pais = models.CharField(max_length=50)
    tiempo_llegada = models.CharField(max_length=50)  # e.g., "5-7 dias"
    cantidad_minima_compra = models.PositiveIntegerField()

    def __str__(self):
        return self.nombre