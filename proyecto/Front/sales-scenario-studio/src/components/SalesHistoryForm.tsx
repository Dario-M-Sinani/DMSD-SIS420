// components/SalesHistoryForm.tsx
'use client';

import { useState, useRef } from 'react';
import { Upload, FileSpreadsheet, X } from 'lucide-react';
import { Button } from '@/components/ui/button';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { toast } from 'sonner';
import styles from '@/styles/SalesHistoryForm.module.css';

interface SalesHistoryFormProps {
  onFileChange?: (file: File | null) => void;
}

export default function SalesHistoryForm({ onFileChange }: SalesHistoryFormProps) {
  const [file, setFile] = useState<File | null>(null);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [previewData, setPreviewData] = useState<{
    headers: string[];
    rows: string[][];
    totalRows: number;
  } | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (selectedFile: File | null) => {
    if (!selectedFile) return;

    const validTypes = [
      'text/csv',
      'application/vnd.ms-excel',
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    ];

    if (!validTypes.includes(selectedFile.type) && !selectedFile.name.endsWith('.csv') && !selectedFile.name.endsWith('.xlsx')) {
      toast.error('Por favor, sube un archivo CSV o XLSX');
      return;
    }

    setFile(selectedFile);
    onFileChange?.(selectedFile);
    parseFileForPreview(selectedFile);
  };

  const parseFileForPreview = (file: File) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      const text = e.target?.result as string;
      const lines = text.split(/\r\n|\n/).filter(line => line.trim());
      if (lines.length === 0) return;

      const headers = lines[0].split(',').map(h => h.trim().replace(/^"|"$/g, ''));
      const rows = lines.slice(1, 4).map(line =>
        line.split(',').map(cell => cell.trim().replace(/^"|"$/g, ''))
      );

      setPreviewData({
        headers,
        rows,
        totalRows: lines.length - 1,
      });
      setIsModalOpen(true);
    };
    reader.readAsText(file);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    const droppedFile = e.dataTransfer.files[0];
    handleFileChange(droppedFile);
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
  };

  return (
    <>
      <div
        className={styles.dropZone}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onClick={() => fileInputRef.current?.click()}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".csv,.xlsx"
          onChange={(e) => handleFileChange(e.target.files?.[0] || null)}
          className="hidden"
        />
        <Upload className="w-10 h-10 text-teal-600 mb-3" />
        {file ? (
          <div className="text-center">
            <p className="font-medium text-gray-900">{file.name}</p>
            <p className="text-sm text-gray-500">
              {(file.size / 1024).toFixed(2)} KB
            </p>
          </div>
        ) : (
          <div className="text-center">
            <p className="text-sm font-medium text-gray-700">
              Arrastra tu archivo CSV/XLSX aquí
            </p>
            <p className="text-xs text-gray-500">o haz clic para seleccionar</p>
          </div>
        )}
      </div>

      <Dialog open={isModalOpen} onOpenChange={setIsModalOpen}>
        <DialogContent className="max-w-4xl max-h-[80vh] overflow-y-auto bg-white">
          <DialogHeader>
            <DialogTitle className="flex items-center justify-between">
              <span>Vista previa: {file?.name}</span>
              <Button
                variant="ghost"
                size="icon"
                onClick={() => setIsModalOpen(false)}
              >
                <X className="h-4 w-4" />
              </Button>
            </DialogTitle>
            <DialogDescription>
              {previewData && (
                <p className="text-sm">
                  {previewData.headers.length} columnas • {previewData.totalRows} filas totales
                </p>
              )}
            </DialogDescription>
          </DialogHeader>

          {previewData && (
            <div className={styles.tableContainer}>
              <table className={styles.previewTable}>
                <thead>
                  <tr>
                    {previewData.headers.map((header, i) => (
                      <th key={i}>{header || `Col ${i + 1}`}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {previewData.rows.map((row, i) => (
                    <tr key={i}>
                      {row.map((cell, j) => (
                        <td key={j}>{cell}</td>
                      ))}
                    </tr>
                  ))}
                  {previewData.rows.length === 0 && (
                    <tr>
                      <td colSpan={previewData.headers.length} className="text-center text-gray-500">
                        No hay datos para mostrar
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
              {previewData.totalRows > 3 && (
                <p className="text-center text-sm text-gray-500 mt-3">
                  Mostrando 3 de {previewData.totalRows} filas
                </p>
              )}
            </div>
          )}
        </DialogContent>
      </Dialog>
    </>
  );
}