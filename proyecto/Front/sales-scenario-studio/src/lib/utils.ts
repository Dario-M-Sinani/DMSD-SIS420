import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";
import { PredictionResponse } from "@/services/predictionService";
import { ReorderInfo } from "@/pages/Simulator"; // Import ReorderInfo

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export interface SavedSimulation {
  id: string;
  timestamp: number;
  result: PredictionResponse;
  selectedProducts: { id?: number; codigo: string; nombre: string; stock: number; }[];
  reorderData?: ReorderInfo[]; // Make reorderData optional
}

const LOCAL_STORAGE_KEY = "savedSimulations";

export function loadSimulationResults(): SavedSimulation[] {
  if (typeof window === "undefined") {
    return [];
  }
  const stored = localStorage.getItem(LOCAL_STORAGE_KEY);
  return stored ? JSON.parse(stored) : [];
}

export function saveSimulationResult(
  result: PredictionResponse,
  selectedProducts: { id?: number; codigo: string; nombre: string; stock: number; }[],
  reorderData: ReorderInfo[] // Add reorderData parameter
): void {
  if (typeof window === "undefined") {
    return;
  }
  const simulations = loadSimulationResults();
  const newSimulation: SavedSimulation = {
    id: crypto.randomUUID(),
    timestamp: Date.now(),
    result: result,
    selectedProducts: selectedProducts.map(p => ({ id: p.id, codigo: p.codigo, nombre: p.nombre, stock: p.stock })),
    reorderData: reorderData, // Save reorderData
  };
  simulations.push(newSimulation);
  localStorage.setItem(LOCAL_STORAGE_KEY, JSON.stringify(simulations));
}

export function clearSimulationResults(): void {
  if (typeof window === "undefined") {
    return;
  }
  localStorage.removeItem(LOCAL_STORAGE_KEY);
}
