import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Trash2, ArrowLeft } from "lucide-react";
import { toast } from "sonner";
import { loadSimulationResults, clearSimulationResults, SavedSimulation } from "@/lib/utils";
import ResultDisplay from "@/components/Simulator/ResultChart";
import { format } from "date-fns";
import { es } from "date-fns/locale";

export default function ReportsPage() {
  const [simulations, setSimulations] = useState<SavedSimulation[]>([]);
  const [viewingSimulation, setViewingSimulation] = useState<SavedSimulation | null>(null);

  useEffect(() => {
    setSimulations(loadSimulationResults());
  }, []);

  const handleClearReports = () => {
    clearSimulationResults();
    setSimulations([]);
    setViewingSimulation(null);
    toast.success("Todos los reportes han sido eliminados.");
  };

  const handleViewDetails = (simulation: SavedSimulation) => {
    setViewingSimulation(simulation);
  };

  if (viewingSimulation) {
    return (
      <div className="min-h-screen bg-background p-8">
        <div className="mx-auto max-w-7xl">
          <div className="mb-8 flex items-center justify-between">
            <Button onClick={() => setViewingSimulation(null)} className="gap-2">
              <ArrowLeft className="h-5 w-5" />
              Volver a la lista de reportes
            </Button>
            <h1 className="text-2xl font-bold text-foreground">Detalles de Simulación</h1>
            <div></div> {/* Placeholder for spacing */}
          </div>
          <ResultDisplay result={viewingSimulation.result} products={viewingSimulation.selectedProducts} reorderData={viewingSimulation.reorderData} isReportView={true} />
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background p-8">
      <div className="mx-auto max-w-7xl">
        <div className="mb-8 flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-foreground">Reportes de Simulaciones Guardadas</h1>
            <p className="mt-2 text-muted-foreground">
              Historial de todas las simulaciones de ventas generadas.
            </p>
          </div>
          <Button onClick={handleClearReports} variant="destructive" className="gap-2" disabled={simulations.length === 0}>
            <Trash2 className="h-5 w-5" />
            Limpiar Todos los Reportes
          </Button>
        </div>

        {simulations.length === 0 ? (
          <Card>
            <CardContent className="p-6 text-center text-muted-foreground">
              No hay simulaciones guardadas aún. Genera una en la página de Simulador.
            </CardContent>
          </Card>
        ) : (
          <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
            {simulations.map((sim) => (
              <Card key={sim.id}>
                <CardHeader>
                  <CardTitle>Simulación {format(new Date(sim.timestamp), "dd/MM/yyyy HH:mm", { locale: es })}</CardTitle>
                  <CardDescription>
                    Productos: {sim.selectedProducts.length}
                  </CardDescription>
                </CardHeader>
                <CardContent className="flex justify-between items-center">
                  <p className="text-sm text-muted-foreground">
                    {sim.selectedProducts.map(p => p.nombre).join(', ').substring(0, 50)}...
                  </p>
                  <Button onClick={() => handleViewDetails(sim)} size="sm">
                    Ver Detalles
                  </Button>
                </CardContent>
              </Card>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
