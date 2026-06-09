const API_BASE_URL = `${import.meta.env.VITE_API_URL}/api`;

export interface Product {
  id?: number;
  codigo: string;
  nombre: string;
  descripcion: string;
  stock: number;
  costo: string;
  precio_venta: string;
  pais: string;
  tiempo_llegada: number;
  cantidad_minima_compra: number;
}

export const getProducts = async (): Promise<Product[]> => {
  const response = await fetch(`${API_BASE_URL}/productos/`);
  if (!response.ok) {
    throw new Error("Error al obtener los productos");
  }
  return response.json();
};

export const createProduct = async (product: Omit<Product, 'id'>): Promise<Product> => {
  const response = await fetch(`${API_BASE_URL}/productos/`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(product),
  });
  if (!response.ok) {
    throw new Error("Error al crear el producto");
  }
  return response.json();
};

export const updateProduct = async (id: number, product: Partial<Product>): Promise<Product> => {
  const response = await fetch(`${API_BASE_URL}/productos/${id}/`, {
    method: "PUT",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(product),
  });
  if (!response.ok) {
    throw new Error("Error al actualizar el producto");
  }
  return response.json();
};

export const deleteProduct = async (id: number): Promise<void> => {
  const response = await fetch(`${API_BASE_URL}/productos/${id}/`, {
    method: "DELETE",
  });
  if (!response.ok) {
    throw new Error("Error al eliminar el producto");
  }
};
