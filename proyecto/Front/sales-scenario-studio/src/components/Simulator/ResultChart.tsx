// src/components/Simulator/ResultChart.tsx (CÓDIGO COMPLETO Y FINAL)

import { useState, useMemo, useRef } from "react";
// Importamos la nueva interfaz desde Simulator
import { ReorderInfo } from "@/pages/Simulator"; 
import { PredictionResponse, SuccessfulPrediction, notifyStock, NotifyPayload } from "@/services/predictionService";
import { Product } from "@/services/productService";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Bar, BarChart, CartesianGrid, Legend, Rectangle, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { AlertCircle, AlertTriangle, ChevronDown, ChevronRight, Expand, Mail, FileText, Loader2, ShoppingCart } from "lucide-react";
import HistoricalChart from "./HistoricalChart";
import { Fragment } from "react/jsx-runtime";
import { Badge } from "@/components/ui/badge";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { useToast } from "@/hooks/use-toast";
import { toast as sonnerToast } from "sonner";
import html2pdf from 'html2pdf.js';

interface ResultDisplayProps {
  result: PredictionResponse;
  products: Product[];
  reorderData?: ReorderInfo[]; // Make reorderData optional
  isReportView?: boolean;
}

export default function ResultDisplay({ result, products, reorderData = [], isReportView = false }: ResultDisplayProps) {
  const { predicciones_exitosas, errores } = result;
  const [expandedProduct, setExpandedProduct] = useState<string | null>(null);
  const [expandedChart, setExpandedChart] = useState<{ type: 'bar'; title: string } | null>(null);
  const [isSendingEmail, setIsSendingEmail] = useState(false);
  const { toast } = useToast();
  const expandedChartRef = useRef<HTMLDivElement>(null);

  // Creamos un Map con la nueva información de reabastecimiento para buscarla fácilmente
  const reorderMap = useMemo(() => {
    return Object.fromEntries(reorderData.map(r => [r.codigo, r]));
  }, [reorderData]);

  // Tus datos de gráficos (sin cambios)
  const chartData = predicciones_exitosas.map(p => ({
    name: p.producto,
    "Ventas Estimadas": p.ventas_estimadas,
    "Rango Mínimo": p.rango_minimo,
    "Rango Máximo": p.rango_maximo,
  }));

  // Tu función de clic en fila (sin cambios)
  const handleRowClick = (productCode: string) => {
    setExpandedProduct(prev => (prev === productCode ? null : productCode));
  };

  // --- FUNCIÓN DE NOTIFICAR ACTUALIZADA ---
  // Ahora acepta el objeto 'ReorderInfo' completo
  const handleNotifyClick = async (reorderData: ReorderInfo) => {
    setIsSendingEmail(true);
    try {
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      // Creamos el nuevo payload con todos los datos que el backend espera
      const payload: NotifyPayload = {
        product_code: reorderData.codigo,
        product_name: reorderData.nombre,
        stock_actual: reorderData.stock_actual,
        punto_pedido: reorderData.punto_pedido,
        cantidad_a_pedir: reorderData.cantidad_a_pedir,
        tiempo_llegada: reorderData.tiempo_llegada,
        ventas_estimadas: reorderData.ventas_estimadas,
        destinatario_correo: "admin@empresa.com"
      };
      
      const response = await notifyStock(payload); // Llamamos al servicio con el payload
      sonnerToast.success(response.message || "Correo de notificación enviado con éxito.");
    } catch (error: any) {
      sonnerToast.error(error.message || "Error al enviar el correo de notificación.");
    } finally {
      setIsSendingEmail(false);
    }
  };

  // Tu función de PDF (sin cambios)
  const handleGeneratePdf = (filenamePrefix: string) => {
    if (expandedChartRef.current) {
      sonnerToast.info("Generando PDF...", { duration: 3000 });
      html2pdf().set({
        margin: [10, 10, 10, 10],
        filename: `${filenamePrefix}_${new Date().toISOString().slice(0,10)}.pdf`,
        image: { type: 'jpeg', quality: 0.98 },
        html2canvas: { scale: 2, logging: true, dpi: 192, letterRendering: true },
        jsPDF: { unit: 'mm', format: 'a4', orientation: 'portrait' },
        pagebreak: { mode: ['avoid-all', 'css', 'legacy'] }
      }).from(expandedChartRef.current).save();
      sonnerToast.success("PDF generado con éxito.");
    } else {
      sonnerToast.error("No se pudo encontrar el contenido para generar el PDF.");
    }
  };

  // Tu contenido de gráficos (sin cambios)
  const barChartContent = (
    <ResponsiveContainer width="100%" height={500}>
      <BarChart data={chartData}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="name" />
        <YAxis />
        <Tooltip content={({ active, payload, label }) => {
          if (active && payload && payload.length) {
            const data = payload[0].payload;
            return (
              <div className="p-2 border bg-background rounded-lg shadow">
                <p className="font-bold">{label}</p>
                <p>Ventas Estimadas: {data['Ventas Estimadas'].toFixed(2)}</p>
                <p>Rango Mínimo: {data['Rango Mínimo'].toFixed(2)}</p>
                <p>Rango Máximo: {data['Rango Máximo'].toFixed(2)}</p>
              </div>
            );
          }
          return null;
        }} />
        <Legend />
        <Bar dataKey="Ventas Estimadas" fill="#8884d8" />
        <Bar dataKey="Rango Mínimo" fill="#82ca9d" />
        <Bar dataKey="Rango Máximo" fill="#ffc658" />
      </BarChart>
    </ResponsiveContainer>
  );

  // Contar productos para pedir (opcional, para el resumen)
  const productsToReorder = reorderData.filter(item => item.necesita_pedido).length;

  return (
    <div className="space-y-4">
      {/* Tus Diálogos (sin cambios) */}
      <Dialog open={!!expandedChart} onOpenChange={() => setExpandedChart(null)}>
        <DialogContent className="w-[90vw] max-w-[90vw] h-[90vh] flex flex-col">
          <DialogHeader className="flex flex-row items-center justify-between">
            <DialogTitle>{expandedChart?.title}</DialogTitle>
            <Button onClick={() => handleGeneratePdf(expandedChart?.title.replace(/\s/g, '_') || 'Reporte_Grafico')} variant="outline" size="sm">
              <FileText className="mr-2 h-4 w-4" />
              Generar PDF
            </Button>
          </DialogHeader>
          <div ref={expandedChartRef} className="flex-grow h-full">
            {expandedChart?.type === 'bar' && barChartContent}
            {expandedChart?.type === 'pie' && pieChartContent}
          </div>
        </DialogContent>
      </Dialog>
      <Dialog open={isSendingEmail}>
        <DialogContent className="w-auto p-8 flex flex-col items-center justify-center">
          <Loader2 className="h-12 w-12 animate-spin text-primary" />
          <p className="mt-4 text-lg font-semibold">Enviando correo de notificación...</p>
          <p className="text-sm text-muted-foreground">Por favor, espera.</p>
        </DialogContent>
      </Dialog>

      <div className="space-y-4">
        
        {/* --- NUEVA TABLA DE ALERTAS (Reemplaza la tuya) --- */}
        {/* Esta tarjeta ahora va primero y muestra la lógica de Punto de Pedido */}
        {!isReportView && (
          <Card className={`overflow-hidden transition-all duration-300 ${productsToReorder > 0 ? "border-red-200 bg-gradient-to-br from-red-50 to-white shadow-md" : ""}`}>
            <CardHeader className={productsToReorder > 0 ? "text-red-900 pb-4 border-b border-red-100" : ""}>
              <CardTitle className="flex items-center text-xl">
                {productsToReorder > 0 ? (
                  <AlertTriangle className="mr-3 h-6 w-6 text-red-600" />
                ) : (
                  <ShoppingCart className="mr-3 h-5 w-5" />
                )}
                Alertas de Reabastecimiento
              </CardTitle>
              <p className={`text-sm mt-1 ${productsToReorder > 0 ? "text-red-700/80 font-medium" : "text-muted-foreground"}`}>
                Basado en la predicción de ventas y los tiempos de llegada, 
                {productsToReorder === 0 
                  ? " no se detectan alertas de stock crítico." 
                  : ` se recomienda gestionar el pedido de ${productsToReorder} producto(s) para evitar quiebres de inventario.`
                }
              </p>
            </CardHeader>
            <CardContent className={productsToReorder > 0 ? "pt-4" : ""}>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead className="w-[50px]"></TableHead>
                    <TableHead>Producto</TableHead>
                    <TableHead className="text-right">Stock Actual</TableHead>
                    <TableHead className="text-right">Punto de Pedido</TableHead>
                    <TableHead className="text-right">Vtas. Estimadas</TableHead>
                    <TableHead className="text-right">Stock Final</TableHead>
                    <TableHead>Estado</TableHead>
                    <TableHead className="text-right">Pedir (Cant. Mín)</TableHead>
                    <TableHead className="text-center">Acciones</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {predicciones_exitosas.map((p: SuccessfulPrediction) => {
                    
                    // Obtenemos la info de reabastecimiento que calculamos en Simulator.tsx
                    const reorderInfo = reorderMap[p.producto];
                    
                    // Usamos los valores de reorderInfo o valores por defecto
                    const stockActual = reorderInfo ? reorderInfo.stock_actual : 0;
                    const stockFinal = stockActual - p.ventas_estimadas;
                    const necesitaCompra = reorderInfo ? reorderInfo.necesita_pedido : false;
                    const puntoDePedido = reorderInfo ? reorderInfo.punto_pedido : '---';
                    const cantidadAPedir = reorderInfo ? reorderInfo.cantidad_a_pedir : 0;

                    return (
                      <Fragment key={p.producto}>
                        <TableRow data-state={expandedProduct === p.producto && 'selected'}>
                          <TableCell>
                            <Button variant="ghost" size="icon" onClick={() => handleRowClick(p.producto)}>
                              {expandedProduct === p.producto ? <ChevronDown className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
                            </Button>
                          </TableCell>
                          <TableCell className="font-medium">
                            <div>{reorderInfo?.nombre || p.producto}</div>
                            <div className="text-xs text-muted-foreground">{p.producto}</div>
                          </TableCell>
                          <TableCell className={`text-right font-bold ${necesitaCompra ? 'text-red-600' : 'text-green-600'}`}>
                            {stockActual}
                          </TableCell>
                          <TableCell className="text-right font-bold">{puntoDePedido}</TableCell>
                          <TableCell className="text-right">{p.ventas_estimadas.toFixed(2)}</TableCell>
                          <TableCell className="text-right">{stockFinal.toFixed(2)}</TableCell>
                          <TableCell>
                            <Badge variant={necesitaCompra ? 'destructive' : 'default'}>
                              {necesitaCompra ? 'Pedir' : 'Suficiente'}
                            </Badge>
                          </TableCell>
                          <TableCell className="text-right font-bold">
                            {cantidadAPedir > 0 ? cantidadAPedir : '---'}
                          </TableCell>
                          <TableCell className="text-center">
                            {necesitaCompra && (
                              <Button 
                                variant="outline" 
                                size="sm" 
                                // --- ¡AQUÍ SE PASA EL OBJETO COMPLETO! ---
                                onClick={() => handleNotifyClick(reorderInfo)}
                              >
                                <Mail className="mr-2 h-4 w-4" />
                                Notificar
                              </Button>
                            )}
                          </TableCell>
                        </TableRow>
                        {/* Tu lógica de expansión (sin cambios) */}
                        {expandedProduct === p.producto && (
                          <TableRow>
                            <TableCell colSpan={9}>
                              <HistoricalChart 
                                productCode={p.producto} 
                                prediction={{ 
                                  date: p.fecha_prediccion, 
                                  estimatedSales: p.ventas_estimadas,
                                  minRange: p.rango_minimo,
                                  maxRange: p.rango_maximo,
                                }} 
                              />
                            </TableCell>
                          </TableRow>
                        )}
                      </Fragment>
                    )
                  })}
                </TableBody>
              </Table>
            </CardContent>
          </Card>
        )}

        <Card>
          <CardHeader>
            <div className="flex justify-between items-center">
              <CardTitle>Comparación de Escenarios de Predicción por Producto</CardTitle>
            </div>
            <p className="text-sm text-muted-foreground">Visualización de ventas estimadas, rango mínimo y rango máximo por producto.</p>
          </CardHeader>
          <CardContent>
            <div className="grid gap-4 md:grid-cols-1">
              <Card>
                <CardHeader className="flex flex-row items-center justify-between">
                  <CardTitle>Gráfico de Barras: Escenarios por Producto</CardTitle>
                  <Button variant="ghost" size="icon" onClick={() => setExpandedChart({ type: 'bar', title: `Gráfico de Barras: Escenarios por Producto` })}>
                    <Expand className="h-4 w-4" />
                  </Button>
                </CardHeader>
                <CardContent>{barChartContent}</CardContent>
              </Card>
            </div>
          </CardContent>
        </Card>

        {/* Tu manejo de errores (sin cambios) */}
        {!isReportView && errores && errores.length > 0 && (
          <Alert variant="destructive">
            <AlertCircle className="h-4 w-4" />
            <AlertTitle>Errores en la Predicción</AlertTitle>
            <AlertDescription>
              <ul className="list-disc pl-5">
                {errores.map((e, index) => (<li key={index}><strong>{e.producto}:</strong> {e.error}</li>))}
              </ul>
            </AlertDescription>
          </Alert>
        )}
      </div>
    </div>
  );
}