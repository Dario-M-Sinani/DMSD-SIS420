import React, { useState } from 'react';
import { useHistoricalData } from "@/hooks/useHistoricalData";
import { Skeleton } from "@/components/ui/skeleton";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import styles from './HistoricalChart.module.css';

interface HistoricalChartProps {
  productCode: string;
  prediction: {
    date: string;
    estimatedSales: number;
    minRange: number;
    maxRange: number;
  } | null;
}

const formatDate = (timestamp: number) => {
  const date = new Date(timestamp);
  const month = date.toLocaleString('es-ES', { month: 'short' }).replace('.', '');
  const year = date.getFullYear().toString().slice(-2);
  return `${month}-${year}`;
};

export default function HistoricalChart({ productCode, prediction }: HistoricalChartProps) {
  const { getHistoryForProduct, isLoading, error } = useHistoricalData();
  const [tooltip, setTooltip] = useState<{ x: number; y: number; value: string; visible: boolean } | null>(null);

  if (isLoading) {
    return <Skeleton className="h-[250px] w-full" />;
  }

  if (error) {
    return (
      <Alert variant="destructive">
        <AlertTitle>Error</AlertTitle>
        <AlertDescription>No se pudieron cargar los datos históricos: {error}</AlertDescription>
      </Alert>
    );
  }

  const fullHistoricalData = getHistoryForProduct(productCode);

  if (!fullHistoricalData || fullHistoricalData.length === 0) {
    return <p>No hay datos históricos disponibles para {productCode}.</p>;
  }

  const historical = fullHistoricalData.slice(-6).map(p => ({
    month: formatDate(p.date),
    sales: p.sales
  }));

  const predictionDataPoints = [];
  if (prediction) {
    const predictionMonth = formatDate(new Date(prediction.date).getTime());
    predictionDataPoints.push({
      month: predictionMonth,
      estimatedSales: prediction.estimatedSales,
      minRange: prediction.minRange,
      maxRange: prediction.maxRange,
    });
  }

  const allData = [...historical, ...predictionDataPoints];
  const maxSales = Math.max(
    ...allData.map(d => d.sales || 0), // Historical sales
    ...predictionDataPoints.map(d => d.maxRange || 0), // Prediction max range
    0
  );
  const chartHeight = 200;
  const chartWidth = 800;
  const padding = 40;

  const xStep = allData.length > 1 ? (chartWidth - padding * 2) / (allData.length - 1) : 0;
  const yScale = maxSales > 0 ? (chartHeight - padding * 2) / maxSales : 0;

  const pointData = allData.map((d, i) => ({
    ...d,
    x: padding + i * xStep,
    ySales: chartHeight - padding - (d.sales || 0) * yScale,
    yEstimated: chartHeight - padding - (d.estimatedSales || 0) * yScale,
    yMin: chartHeight - padding - (d.minRange || 0) * yScale,
    yMax: chartHeight - padding - (d.maxRange || 0) * yScale,
    isPrediction: i >= historical.length,
  }));

  const historicalPoints = pointData.filter(p => !p.isPrediction);
  const predictionPoints = pointData.filter(p => p.isPrediction);

  return (
    <div className={styles.resultCard}>
      <svg viewBox={`0 0 ${chartWidth} ${chartHeight}`} className={styles.svg} onMouseLeave={() => setTooltip(null)}>
        <defs>
          <filter id="shadow" x="-50%" y="-50%" width="200%" height="200%">
            <feDropShadow dx="0" dy="1" stdDeviation="1" floodColor="#000000" floodOpacity="0.2" />
          </filter>
        </defs>
      
        {/* Grid lines */}
        {[0, 0.25, 0.5, 0.75, 1].map((ratio) => (
          <line key={ratio} x1={padding} y1={chartHeight - padding - maxSales * yScale * ratio} x2={chartWidth - padding} y2={chartHeight - padding - maxSales * yScale * ratio} stroke="#e5e7eb" strokeWidth="1" />
        ))}

        {/* Historical line */}
        {historicalPoints.length > 0 && (
          <polyline points={historicalPoints.map(p => `${p.x},${p.ySales}`).join(' ')} fill="none" stroke="#0d9488" strokeWidth="2" />
        )}

        {/* Prediction lines */}
        {predictionPoints.length > 0 && historicalPoints.length > 0 && (
          <>
            {/* Estimated Sales */}
            <polyline points={[`${historicalPoints[historicalPoints.length - 1].x},${historicalPoints[historicalPoints.length - 1].ySales}`, ...predictionPoints.map(p => `${p.x},${p.yEstimated}`)].join(' ')} fill="none" stroke="#f59e0b" strokeWidth="2" strokeDasharray="5,5" />
            {/* Min Range */}
            <polyline points={[`${historicalPoints[historicalPoints.length - 1].x},${historicalPoints[historicalPoints.length - 1].ySales}`, ...predictionPoints.map(p => `${p.x},${p.yMin}`)].join(' ')} fill="none" stroke="#9ca3af" strokeWidth="1" strokeDasharray="2,2" />
            {/* Max Range */}
            <polyline points={[`${historicalPoints[historicalPoints.length - 1].x},${historicalPoints[historicalPoints.length - 1].ySales}`, ...predictionPoints.map(p => `${p.x},${p.yMax}`)].join(' ')} fill="none" stroke="#ef4444" strokeWidth="1" strokeDasharray="2,2" />
          </>
        )}

        {/* Points and hover areas */}
        {pointData.map((p, i) => (
          <React.Fragment key={`hover-area-group-${i}`}>
            {/* Hover area for historical point */}
            {!p.isPrediction && (
              <circle
                cx={p.x}
                cy={p.ySales}
                r="8"
                fill="transparent"
                onMouseEnter={() => setTooltip({
                  x: p.x,
                  y: p.ySales,
                  value: `Ventas: ${p.sales?.toFixed(0)}`,
                  visible: true
                })}
              />
            )}
            {/* Hover areas for prediction points */}
            {p.isPrediction && (
              <>
                <circle
                  cx={p.x}
                  cy={p.yEstimated}
                  r="8"
                  fill="transparent"
                  onMouseEnter={() => setTooltip({
                    x: p.x,
                    y: p.yEstimated,
                    value: `Est: ${p.estimatedSales?.toFixed(0)}`,
                    visible: true
                  })}
                />
                <circle
                  cx={p.x}
                  cy={p.yMin}
                  r="8"
                  fill="transparent"
                  onMouseEnter={() => setTooltip({
                    x: p.x,
                    y: p.yMin,
                    value: `Min: ${p.minRange?.toFixed(0)}`,
                    visible: true
                  })}
                />
                <circle
                  cx={p.x}
                  cy={p.yMax}
                  r="8"
                  fill="transparent"
                  onMouseEnter={() => setTooltip({
                    x: p.x,
                    y: p.yMax,
                    value: `Max: ${p.maxRange?.toFixed(0)}`,
                    visible: true
                  })}
                />
              </>
            )}
          </React.Fragment>
        ))}
        {pointData.map((p, i) => (
          <React.Fragment key={`visible-point-group-${i}`}>
            <circle cx={p.x} cy={p.ySales} r="4" fill={p.isPrediction ? "transparent" : "#0d9488"} style={{ pointerEvents: 'none' }} />
            {p.isPrediction && (
              <>
                <circle cx={p.x} cy={p.yEstimated} r="4" fill="#f59e0b" style={{ pointerEvents: 'none' }} />
                <circle cx={p.x} cy={p.yMin} r="4" fill="#9ca3af" style={{ pointerEvents: 'none' }} />
                <circle cx={p.x} cy={p.yMax} r="4" fill="#ef4444" style={{ pointerEvents: 'none' }} />
              </>
            )}
          </React.Fragment>
        ))}

        {/* X-axis labels */}
        {allData.map((d, i) => {
          if (i % 2 === 0 || i === allData.length - 1) {
            return (<text key={i} x={padding + i * xStep} y={chartHeight - 10} textAnchor="middle" fontSize="10" fill="#6b7280">{d.month}</text>);
          }
          return null;
        })}

        {/* Y-axis labels */}
        {[0, 0.25, 0.5, 0.75, 1].map((ratio) => (
          <text key={ratio} x={padding - 10} y={chartHeight - padding - maxSales * yScale * ratio + 4} textAnchor="end" fontSize="10" fill="#6b7280">{Math.round(maxSales * ratio)}</text>
        ))}

        {/* Tooltip */}
        {tooltip?.visible && (
          <g transform={`translate(${tooltip.x}, ${tooltip.y - 15})`} style={{ pointerEvents: 'none' }}>
            <rect x="-22" y="-22" width="44" height="24" rx="4" fill="white" stroke="#e5e7eb" filter="url(#shadow)" />
            <text x="0" y="-7" textAnchor="middle" fill="black" fontSize="12" fontWeight="bold">{tooltip.value}</text>
          </g>
        )}
      </svg>

      <div className={styles.legend}>
        <div className={styles.legendItem}><div className={`${styles.legendColor} ${styles.legendHistorical}`}></div><span>Histórico</span></div>
        <div className={styles.legendItem}><div className={`${styles.legendColor} ${styles.legendPrediction}`}></div><span>Predicción Estimada</span></div>
        <div className={styles.legendItem}><div className={`${styles.legendColor} ${styles.legendMinRange}`}></div><span>Rango Mínimo</span></div>
        <div className={styles.legendItem}><div className={`${styles.legendColor} ${styles.legendMaxRange}`}></div><span>Rango Máximo</span></div>
      </div>
    </div>
  );
};
