import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Label } from "@/components/ui/label";
import { PredictionPayload } from "@/services/predictionService";
import { Loader2 } from "lucide-react";

interface SimulationFormProps {
  onSubmit: (options: Omit<PredictionPayload, 'nombres_productos'>) => void;
  isSubmitting: boolean;
}

export default function SimulationForm({ onSubmit, isSubmitting }: SimulationFormProps) {
  const [factorComercial, setFactorComercial] = useState(false);
  const [factorEventos, setFactorEventos] = useState(false);
  const [factorEstacional, setFactorEstacional] = useState(false);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSubmit({
      factor_comercial_1: factorComercial ? 1 : 0,
      factor_eventos_2: factorEventos ? 1 : 0,
      factor_estacional_3: factorEstacional ? 1 : 0,
    });
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-6 p-6 border rounded-xl bg-card shadow-sm">
      <div className="grid gap-4 md:grid-cols-3">
        {/* Factor Comercial */}
        <label
          htmlFor="factor_comercial"
          className={`flex flex-col items-start p-5 border rounded-xl cursor-pointer transition-all hover:border-primary/50 hover:shadow-sm ${factorComercial ? 'border-primary bg-primary/5' : 'border-border'}`}
        >
          <div className="flex items-center space-x-3 mb-3">
            <Checkbox id="factor_comercial" checked={factorComercial} onCheckedChange={(checked) => setFactorComercial(!!checked)} />
            <span className="font-semibold text-sm">Factor Comercial</span>
          </div>
          <p className="text-xs text-muted-foreground leading-relaxed">
            Simula campañas de preventa o acuerdos de distribución al por mayor.
          </p>
        </label>

        {/* Factor Eventos */}
        <label
          htmlFor="factor_eventos"
          className={`flex flex-col items-start p-5 border rounded-xl cursor-pointer transition-all hover:border-primary/50 hover:shadow-sm ${factorEventos ? 'border-primary bg-primary/5' : 'border-border'}`}
        >
          <div className="flex items-center space-x-3 mb-3">
            <Checkbox id="factor_eventos" checked={factorEventos} onCheckedChange={(checked) => setFactorEventos(!!checked)} />
            <span className="font-semibold text-sm">Factor Eventos</span>
          </div>
          <p className="text-xs text-muted-foreground leading-relaxed">
            Simula el impacto de ferias agropecuarias o congresos sectoriales.
          </p>
        </label>

        {/* Factor Estacional */}
        <label
          htmlFor="factor_estacional"
          className={`flex flex-col items-start p-5 border rounded-xl cursor-pointer transition-all hover:border-primary/50 hover:shadow-sm ${factorEstacional ? 'border-primary bg-primary/5' : 'border-border'}`}
        >
          <div className="flex items-center space-x-3 mb-3">
            <Checkbox id="factor_estacional" checked={factorEstacional} onCheckedChange={(checked) => setFactorEstacional(!!checked)} />
            <span className="font-semibold text-sm">Factor Estacional</span>
          </div>
          <p className="text-xs text-muted-foreground leading-relaxed">
            Monitorea los ciclos naturales (temporadas de floración alta, cosecha de miel o regímenes de lluvia).
          </p>
        </label>
      </div>

      <Button type="submit" disabled={isSubmitting} className="w-full mt-6 py-6 text-base rounded-lg transition-all">
        {isSubmitting ? <Loader2 className="mr-2 h-5 w-5 animate-spin" /> : null}
        {isSubmitting ? "Calculando Proyección..." : "Ejecutar Predicción"}
      </Button>
    </form>
  );
}
