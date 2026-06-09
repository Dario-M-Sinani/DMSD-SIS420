from django.contrib import admin
from .models import Producto

@admin.register(Producto)
class ProductoAdmin(admin.ModelAdmin):
    list_display = ('codigo', 'nombre', 'stock', 'precio_venta', 'pais')
    list_filter = ('pais', 'stock')
    search_fields = ('codigo', 'nombre')
    
    def get_readonly_fields(self, request, obj=None):
        """Make `codigo` editable when creating an object, but readonly when editing."""
        if obj:  # editing an existing object
            return ('codigo',)
        return ()
