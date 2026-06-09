from rest_framework import serializers

class PrediccionInputSerializer(serializers.Serializer):
    """
    Serializador para validar los datos de entrada de la predicción.
    Acepta una lista de nombres de producto.
    """
    nombres_productos = serializers.ListField(
        child=serializers.CharField(max_length=100),
        allow_empty=False
    )
    factor_comercial_1 = serializers.IntegerField(min_value=0, max_value=1)
    factor_eventos_2 = serializers.IntegerField(min_value=0, max_value=1)
    factor_estacional_3 = serializers.IntegerField(min_value=0, max_value=1)