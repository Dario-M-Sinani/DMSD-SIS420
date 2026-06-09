// src/pages/Simulador/mockData.ts

// src/pages/Simulador/mockData.ts
import type { Product, HistoricalData } from '@/types';

export const mockProducts: Product[] = [
  {
    code: 'OUR1000',
    name: 'Producto Alpha',
    country: 'Brasil',
    leadTime: 60,
    minQuantity: 50,
    pricesensitivity: 'Alta',
    stock: 120
  },
  {
    code: 'OUR2000',
    name: 'Producto Beta',
    country: 'Argentina',
    leadTime: 45,
    minQuantity: 30,
    pricesensitivity: 'Media',
    stock: 80
  },
  {
    code: 'OUR3000',
    name: 'Producto Gamma',
    country: 'Chile',
    leadTime: 30,
    minQuantity: 40,
    pricesensitivity: 'Baja',
    stock: 200
  },
  {
    code: 'OUR4000',
    name: 'Producto Delta',
    country: 'Perú',
    leadTime: 50,
    minQuantity: 25,
    pricesensitivity: 'Alta',
    stock: 45
  }
];

export const mockHistoricalData: Record<string, HistoricalData[]> = {
  'OUR1000': [
    { month: 'jul-24', sales: 78 },
    { month: 'ago-24', sales: 35 },
    { month: 'sept-24', sales: 39 },
    { month: 'oct-24', sales: 132 },
    { month: 'nov-24', sales: 122 },
    { month: 'dic-24', sales: 213 },
    { month: 'ene-25', sales: 51 },
    { month: 'feb-25', sales: 23 },
    { month: 'mar-25', sales: 144 },
    { month: 'abr-25', sales: 101 },
    { month: 'may-25', sales: 282 },
    { month: 'jun-25', sales: 320 },
    { month: 'jul-25', sales: 50 },
    { month: 'ago-25', sales: 219 },
    { month: 'sept-25', sales: 179 },
    { month: 'oct-25', sales: 443 }
  ],
  'OUR2000': [
    { month: 'jul-24', sales: 32 },
    { month: 'ago-24', sales: 53 },
    { month: 'sept-24', sales: 81 },
    { month: 'oct-24', sales: 88 },
    { month: 'nov-24', sales: 114 },
    { month: 'dic-24', sales: 59 },
    { month: 'ene-25', sales: 33 },
    { month: 'feb-25', sales: 105 },
    { month: 'mar-25', sales: 142 },
    { month: 'abr-25', sales: 31 },
    { month: 'may-25', sales: 318 },
    { month: 'jun-25', sales: 111 },
    { month: 'jul-25', sales: 101 },
    { month: 'ago-25', sales: 65 },
    { month: 'sept-25', sales: 70 },
    { month: 'oct-25', sales: 173 }
  ],
  'OUR3000': [
    { month: 'jul-24', sales: 45 },
    { month: 'ago-24', sales: 67 },
    { month: 'sept-24', sales: 89 },
    { month: 'oct-24', sales: 112 },
    { month: 'nov-24', sales: 98 },
    { month: 'dic-24', sales: 156 },
    { month: 'ene-25', sales: 78 },
    { month: 'feb-25', sales: 92 },
    { month: 'mar-25', sales: 134 },
    { month: 'abr-25', sales: 121 },
    { month: 'may-25', sales: 167 },
    { month: 'jun-25', sales: 189 },
    { month: 'jul-25', sales: 145 },
    { month: 'ago-25', sales: 156 },
    { month: 'sept-25', sales: 178 },
    { month: 'oct-25', sales: 145 }
  ],
  'OUR4000': [
    { month: 'jul-24', sales: 89 },
    { month: 'ago-24', sales: 95 },
    { month: 'sept-24', sales: 102 },
    { month: 'oct-24', sales: 145 },
    { month: 'nov-24', sales: 167 },
    { month: 'dic-24', sales: 189 },
    { month: 'ene-25', sales: 92 },
    { month: 'feb-25', sales: 78 },
    { month: 'mar-25', sales: 134 },
    { month: 'abr-25', sales: 156 },
    { month: 'may-25', sales: 201 },
    { month: 'jun-25', sales: 234 },
    { month: 'jul-25', sales: 167 },
    { month: 'ago-25', sales: 189 },
    { month: 'sept-25', sales: 212 },
    { month: 'oct-25', sales: 256 }
  ]
};