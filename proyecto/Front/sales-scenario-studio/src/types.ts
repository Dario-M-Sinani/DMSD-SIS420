// src/pages/Simulador/types.ts

export interface Product {
  code: string;
  name: string;
  country: string;
  leadTime: number;
  minQuantity: number;
  pricesensitivity: 'Alta' | 'Media' | 'Baja';
  stock: number;
}

export interface HistoricalData {
  month: string;
  sales: number;
}

export interface PredictionData {
  month: string;
  sales: number;
}

export interface SimulationParams {
  months: 1 | 2 | 3 | 6;
  hasRain: boolean;
  hasPromotion: boolean;
  hasFair: boolean;
}

export interface SimulationResult {
  productCode: string;
  historical: HistoricalData[];
  predictions: PredictionData[];
  stock: number;
  stockSufficient: boolean;
  shortfall?: number;
}