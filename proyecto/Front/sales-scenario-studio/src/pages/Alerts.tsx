import React, { useEffect, useState } from 'react';

interface Producto {
  id: number;
  nombre: string;
  descripcion: string;
  precio: number;
  stock: number;
}

const Alerts: React.FC = () => {
  const [productos, setProductos] = useState<Producto[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [sliderValues, setSliderValues] = useState<{ [key: number]: number }>({});

  useEffect(() => {
    const fetchProductos = async () => {
      try {
        const response = await fetch(`${import.meta.env.VITE_API_URL}/api/productos/`);
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        setProductos(data);
        // Inicializar los valores del slider a 0 para cada producto
        const initialSliderValues: { [key: number]: number } = {};
        data.forEach((producto: Producto) => {
          initialSliderValues[producto.id] = 0;
        });
        setSliderValues(initialSliderValues);
      } catch (e: any) {
        setError(e.message);
      } finally {
        setLoading(false);
      }
    };

    fetchProductos();
  }, []);

  const handleSliderChange = (productId: number, value: number) => {
    setSliderValues((prevValues) => ({
      ...prevValues,
      [productId]: value,
    }));
  };

  if (loading) {
    return <div>Cargando productos...</div>;
  }

  if (error) {
    return <div>Error: {error}</div>;
  }

  return (
    <div className="p-6 bg-gray-50 min-h-screen">
      <h1 className="text-3xl font-extrabold text-gray-800 mb-6 text-center">Alertas de Productos</h1>
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
        {productos.map((producto) => (
          <div key={producto.id} className="bg-white border border-gray-200 rounded-xl shadow-lg p-6 flex flex-col justify-between transform transition duration-300 hover:scale-105 hover:shadow-xl">
            <div>
              <h2 className="text-2xl font-bold text-gray-900 mb-2">{producto.nombre}</h2>
              <p className="text-gray-700 text-sm mb-3">{producto.descripcion}</p>
              <p className="text-xl font-extrabold text-indigo-600 mb-1">Precio: ${producto.precio ? producto.precio.toFixed(2) : 'N/A'}</p>
              <p className="text-md text-gray-600">Stock actual: <span className="font-semibold">{producto.stock}</span></p>
            </div>
            <div className="mt-5 pt-4 border-t border-gray-100">
              <label htmlFor={`slider-${producto.id}`} className="block text-sm font-medium text-gray-700 mb-2">
                Cantidad bajo stock: <span className="font-semibold text-indigo-600">{sliderValues[producto.id]}</span>
              </label>
              <input
                type="range"
                id={`slider-${producto.id}`}
                min="0"
                max={producto.stock}
                value={sliderValues[producto.id]}
                onChange={(e) => handleSliderChange(producto.id, parseInt(e.target.value))}
                className="w-full h-2 bg-indigo-200 rounded-lg appearance-none cursor-pointer accent-indigo-600"
              />
              <div className="flex justify-between text-xs text-gray-500 mt-1">
                <span>0</span>
                <span>{producto.stock}</span>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default Alerts;