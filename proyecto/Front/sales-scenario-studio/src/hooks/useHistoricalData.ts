import { useQuery } from '@tanstack/react-query';
import Papa from 'papaparse';

export interface HistoricalDataPoint {
  date: number; // Using timestamp for sortability
  sales: number;
}

type ProductHistory = Record<string, HistoricalDataPoint[]>;

const monthMap: Record<string, number> = {
  'ene': 0, 'feb': 1, 'mar': 2, 'abr': 3, 'may': 4, 'jun': 5,
  'jul': 6, 'ago': 7, 'sep': 8, 'oct': 9, 'nov': 10, 'dic': 11,
};

function parseCustomDate(dateStr: string): number {
  const [month, year] = dateStr.split('-');
  const monthIndex = monthMap[month.toLowerCase()];
  const fullYear = 2000 + parseInt(year, 10);
  return new Date(fullYear, monthIndex, 1).getTime();
}

async function fetchAndParseCsv(): Promise<ProductHistory> {
  const response = await fetch('/historico_ventas.csv');
  if (!response.ok) {
    throw new Error('Network response was not ok when fetching historical data.');
  }
  const csvText = await response.text();

  return new Promise((resolve, reject) => {
    Papa.parse(csvText, {
      header: true,
      dynamicTyping: true,
      skipEmptyLines: true,
      complete: (results) => {
        const rawData = results.data as Record<string, any>[];
        const transformedData: ProductHistory = {};
        
        if (rawData.length > 0) {
          const productCodes = Object.keys(rawData[0]).filter(key => 
            key.toLowerCase() !== 'fecha' && 
            key !== 'ds' &&
            key !== 'codigo_producto' &&
            !key.toLowerCase().includes('factor') &&
            !key.toLowerCase().includes('promocion') &&
            !key.toLowerCase().includes('feria') &&
            !key.toLowerCase().includes('lluvia')
          );

          for (const code of productCodes) {
            transformedData[code] = [];
          }

          for (const row of rawData) {
            const dateStr = row['Fecha'] || row['fecha'] || row['ds'];
            if (dateStr) {
              const timestamp = parseCustomDate(dateStr);
              for (const code of productCodes) {
                const sales = row[code];
                if (sales !== null && sales !== undefined) {
                  transformedData[code].push({ date: timestamp, sales });
                }
              }
            }
          }
          // Sort each product's history by date
          for (const code of productCodes) {
            transformedData[code].sort((a, b) => a.date - b.date);
          }
        }
        resolve(transformedData);
      },
      error: (err: any) => {
        reject(new Error(`Error parsing CSV: ${err.message}`));
      },
    });
  });
}

export function useHistoricalData() {
  const { data, isLoading, isError, error } = useQuery<ProductHistory, Error>({
    queryKey: ['historicalData'],
    queryFn: fetchAndParseCsv,
    staleTime: Infinity, // CSV data is static, so cache forever
    cacheTime: 1000 * 60 * 60, // Keep cached for 1 hour
  });

  const getHistoryForProduct = (productCode: string): HistoricalDataPoint[] | undefined => {
    return data?.[productCode];
  };

  return { getHistoryForProduct, isLoading, isError, error: error?.message };
}
