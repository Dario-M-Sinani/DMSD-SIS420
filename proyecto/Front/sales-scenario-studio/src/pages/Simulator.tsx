// src/pages/Simulator.tsx (CÓDIGO FINAL Y CORREGIDO)

import { useState, useMemo } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { getProducts, Product } from "@/services/productService";
import { predictSales, PredictionPayload, PredictionResponse } from "@/services/predictionService";
import SimulationForm from "@/components/Simulator/SimulationForm";
import ResultDisplay from "@/components/Simulator/ResultChart"; // Importa tu componente de resultados
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Loader2, BrainCircuit } from "lucide-react";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { saveSimulationResult } from "@/lib/utils";

// Esta interfaz se exporta para que ResultChart.tsx pueda usarla
export interface ReorderInfo {
  codigo: string;
  nombre: string;
  stock_actual: number;
  ventas_estimadas: number;
  rango_maximo: number;
  tiempo_llegada: number;
  punto_pedido: number;
  necesita_pedido: boolean;
  cantidad_a_pedir: number;
}


export default function Simulator() {
  const [selectedProducts, setSelectedProducts] = useState<Product[]>([]);
  const [predictionResult, setPredictionResult] = useState<PredictionResponse | null>(null);
  const [searchTerm, setSearchTerm] = useState("");
  const [showForm, setShowForm] = useState(false);
  const [sortBy, setSortBy] = useState("none"); 

  const { data: allProducts, isLoading: isLoadingProducts } = useQuery<Product[]>({
    queryKey: ["products"],
    queryFn: getProducts,
  });

  const sortedAndFilteredProducts = useMemo(() => {
    let productsToDisplay = allProducts || [];
    if (searchTerm) {
      productsToDisplay = productsToDisplay.filter(p =>
        p.nombre.toLowerCase().includes(searchTerm.toLowerCase()) ||
        p.codigo.toLowerCase().includes(searchTerm.toLowerCase())
      );
    }
    switch (sortBy) {
      case "stock-asc":
        productsToDisplay.sort((a, b) => a.stock - b.stock);
        break;
      case "stock-desc":
        productsToDisplay.sort((a, b) => b.stock - a.stock);
        break;
      case "name-asc":
        productsToDisplay.sort((a, b) => a.nombre.localeCompare(b.nombre));
        break;
      case "name-desc":
        productsToDisplay.sort((a, b) => b.nombre.localeCompare(a.nombre));
        break;
      default:
        productsToDisplay.sort((a, b) => a.nombre.localeCompare(b.nombre));
        break;
    }
    return productsToDisplay;
  }, [allProducts, searchTerm, sortBy]);

  const predictMutation = useMutation<PredictionResponse, Error, Omit<PredictionPayload, 'nombres_productos'>>({
    mutationFn: (options) => {
      const payload: PredictionPayload = {
        nombres_productos: selectedProducts.map(p => p.codigo),
        ...options,
      };
      return predictSales(payload);
    },
    onSuccess: (data) => {
      setPredictionResult(data);
      saveSimulationResult(data, selectedProducts, reorderData);
    },
  });

  // LÓGICA DE REABASTECIMIENTO (CORREGIDA - A NIVEL SUPERIOR)
  const reorderData = useMemo(() => {
    const data: ReorderInfo[] = [];

    if (predictionResult && predictMutation.isSuccess) {
      for (const prediction of predictionResult.predicciones_exitosas) {
                  // Encuentra el producto completo para obtener 'tiempo_llegada' y 'stock'
                  const product = allProducts?.find(p => p.codigo === prediction.producto);
        
                  if (product) {
                    // tiempo_llegada ya es un number gracias a la interfaz Product
                    const tiempoLlegadaNum = product.tiempo_llegada;
                    
                    // 1. Demanda Diaria Máxima
                    const maxDailyDemand = (prediction.rango_maximo || prediction.ventas_estimadas) / 30;
                    // 2. Punto de Pedido
                    const reorderPoint = Math.ceil(maxDailyDemand * tiempoLlegadaNum);
                    // 3. Decisión
                    const needsReorder = product.stock <= reorderPoint;          
          data.push({
            codigo: product.codigo,
            nombre: product.nombre,
            stock_actual: product.stock,
            ventas_estimadas: prediction.ventas_estimadas,
            rango_maximo: prediction.rango_maximo,
            tiempo_llegada: tiempoLlegadaNum,
            punto_pedido: reorderPoint,
            necesita_pedido: needsReorder,
            cantidad_a_pedir: needsReorder ? product.cantidad_minima_compra : 0,
          });
        }
      }
    }
    return data;
  }, [predictionResult, allProducts, predictMutation.isSuccess]); // Usamos allProducts aquí


  // --- Funciones 'handle' (sin cambios) ---
  const handleSelectProduct = (product: Product, checked: boolean) => {
    setSelectedProducts(prev =>
      checked ? [...prev, product] : prev.filter(p => p.id !== product.id)
    );
  };
  const handleSelectAll = (checked: boolean) => {
    setSelectedProducts(checked ? [...sortedAndFilteredProducts] : []);
  };
  const handlePredictAll = () => {
    setSelectedProducts(allProducts || []);
    setShowForm(true);
  };
  const handlePredictSelected = () => {
    if (selectedProducts.length > 0) {
      setShowForm(true);
    }
  };
  const handleNewSimulation = () => {
    setSelectedProducts([]);
    setPredictionResult(null);
    predictMutation.reset();
    setShowForm(false);
  };
  // --- Fin de funciones 'handle' ---


  // RENDER: Vista de Resultados
  if (predictMutation.isSuccess && predictionResult) {
    return (
      <div className="p-4 space-y-4">
        <div className="flex justify-between items-center">
          <h1 className="text-2xl font-bold">Resultados de la Simulación</h1>
          <Button onClick={handleNewSimulation}>Nueva Simulación</Button>
        </div>
        
        {/* ¡AQUÍ ESTÁ LA CORRECCIÓN! */}
        <ResultDisplay 
          result={predictionResult} 
          products={selectedProducts} 
          reorderData={reorderData}
        />
      </div>
    );
  }
  
  // RENDER: Formulario de Simulación
  if (showForm) {
     return (
      <div className="p-4">
        <Card className="max-w-3xl mx-auto">
          <CardHeader>
            <CardTitle>Configure Escenario y Prediga</CardTitle>
            <p className="text-sm text-muted-foreground">
              Se realizará la predicción para {selectedProducts.length} producto(s) seleccionado(s).
            </p>
          </CardHeader>
          <CardContent className="space-y-6">
            <SimulationForm
              onSubmit={(options) => predictMutation.mutate(options)}
              isSubmitting={predictMutation.isPending}
            />
            {predictMutation.isPending && (
              <div className="flex items-center justify-center p-8"><Loader2 className="h-8 w-8 animate-spin" /><p className="ml-4">Realizando predicción...</p></div>
            )}
            {predictMutation.isError && (
              <Alert variant="destructive"><AlertTitle>Error en la Predicción</AlertTitle><AlertDescription>{predictMutation.error.message || "Ocurrió un error desconocido."}</AlertDescription></Alert>
            )}
            <Button variant="outline" onClick={() => setShowForm(false)}>Volver a la selección</Button>
          </CardContent>
        </Card>
      </div>
    );
  }

  // RENDER: Vista Principal (Tabla de Selección)
  return (
    <div className="p-4 space-y-6">
      {/* Panel Contextual */}
      <div className="rounded-xl border bg-slate-50/50 text-card-foreground shadow-sm p-6 relative overflow-hidden transition-all hover:shadow-md">
        <div className="absolute top-0 left-0 w-1.5 h-full bg-primary/80"></div>
        <div className="flex items-start space-x-4">
          <div className="p-3 bg-primary/10 rounded-xl mt-0.5">
            <BrainCircuit className="w-6 h-6 text-primary" />
          </div>
          <div>
            <h2 className="text-xl font-semibold tracking-tight text-foreground">Estudio de Escenarios de Ventas Predictivo</h2>
            <p className="text-muted-foreground mt-2 leading-relaxed text-sm max-w-4xl">
              Esta herramienta utiliza Inteligencia Artificial (Meta Prophet) para proyectar la demanda del próximo mes cruzando datos históricos con factores de mercado, calculando automáticamente el Punto de Pedido crítico para optimizar tu cadena de suministro.
            </p>
          </div>
        </div>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Simulador de Ventas</CardTitle>
          <p className="text-sm text-muted-foreground">Seleccione los productos para los que desea generar una predicción de ventas.</p>
          <div className="flex justify-between items-center pt-4 gap-2">
            <Input
              placeholder="Buscar productos..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="max-w-sm"
            />
            <Select value={sortBy} onValueChange={setSortBy}>
              <SelectTrigger className="w-[180px]">
                <SelectValue placeholder="Ordenar por" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="none">Ninguno</SelectItem>
                <SelectItem value="name-asc">Nombre (A-Z)</SelectItem>
                <SelectItem value="name-desc">Nombre (Z-A)</SelectItem>
                <SelectItem value="stock-asc">Stock (Menor a Mayor)</SelectItem>
                <SelectItem value="stock-desc">Stock (Mayor a Menor)</SelectItem>
              </SelectContent>
            </Select>
            <div className="flex gap-2">
              <Button onClick={handlePredictAll} variant="outline">Predecir Todo</Button>
              <Button onClick={handlePredictSelected} disabled={selectedProducts.length === 0}>Predecir Selección ({selectedProducts.length})</Button>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <div className="rounded-md border">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead className="w-[50px]">
                    <Checkbox
                      checked={sortedAndFilteredProducts.length > 0 && selectedProducts.length === sortedAndFilteredProducts.length}
                      onCheckedChange={(checked) => handleSelectAll(!!checked)}
                      aria-label="Seleccionar todo"
                    />
                  </TableHead>
                  <TableHead>Código</TableHead>
                  <TableHead>Nombre</TableHead>
                  <TableHead className="text-right">Stock</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {isLoadingProducts ? (
                  <TableRow><TableCell colSpan={4} className="h-24 text-center">Cargando productos...</TableCell></TableRow>
                ) : sortedAndFilteredProducts.length > 0 ? (
                  sortedAndFilteredProducts.map((product) => (
                    <TableRow key={product.id} data-state={selectedProducts.some(p => p.id === product.id) && "selected"}>
                      <TableCell>
                        <Checkbox
                          checked={selectedProducts.some(p => p.id === product.id)}
                          onCheckedChange={(checked) => handleSelectProduct(product, !!checked)}
                          aria-label={`Seleccionar ${product.nombre}`}
                        />
                      </TableCell>
                      <TableCell className="font-medium">{product.codigo}</TableCell>
                      <TableCell>{product.nombre}</TableCell>
                      <TableCell className="text-right">{product.stock}</TableCell>
                    </TableRow>
                  ))
                ) : (
                  <TableRow><TableCell colSpan={4} className="h-24 text-center">No se encontraron productos.</TableCell></TableRow>
                )}
              </TableBody>
            </Table>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}