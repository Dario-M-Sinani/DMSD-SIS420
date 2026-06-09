import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { UploadCloud, Package, AlertCircle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { toast } from "sonner";
import PreviewModal from "@/components/Upload/PreviewModal";

interface FileData {
  headers: string[];
  rows: any[][];
  totalRows: number;
  totalColumns: number;
  hasHeaders: boolean;
}

export default function UploadPage() {
  const navigate = useNavigate();
  const [catalogFile, setCatalogFile] = useState<File | null>(null);
  const [fileData, setFileData] = useState<FileData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showModal, setShowModal] = useState(false);

  const parseCSV = (text: string): string[][] => {
    const lines: string[][] = [];
    let currentLine: string[] = [];
    let currentValue = '';
    let inQuotes = false;

    for (let i = 0; i < text.length; i++) {
      const char = text[i];
      const nextChar = text[i + 1];

      if (char === '"') {
        if (inQuotes && nextChar === '"') {
          currentValue += '"';
          i++;
        } else {
          inQuotes = !inQuotes;
        }
      } else if (char === ',' && !inQuotes) {
        currentLine.push(currentValue.trim());
        currentValue = '';
      } else if ((char === '\n' || char === '\r') && !inQuotes) {
        if (char === '\r' && nextChar === '\n') {
          i++;
        }
        if (currentValue || currentLine.length > 0) {
          currentLine.push(currentValue.trim());
          lines.push(currentLine);
          currentLine = [];
          currentValue = '';
        }
      } else {
        currentValue += char;
      }
    }

    if (currentValue || currentLine.length > 0) {
      currentLine.push(currentValue.trim());
      lines.push(currentLine);
    }

    return lines.filter(line => line.some(cell => cell !== ''));
  };

  const processFile = async (file: File) => {
    setLoading(true);
    setError(null);

    try {
      const text = await file.text();
      const parsedData = parseCSV(text);

      if (parsedData.length === 0) {
        setError('El archivo está vacío');
        setLoading(false);
        return;
      }

      const firstRow = parsedData[0];
      const hasHeaders = firstRow.every((cell: string) => {
        const trimmed = cell.trim();
        return trimmed !== '' && isNaN(Number(trimmed));
      });

      let headers: string[];
      let dataRows: string[][];

      if (hasHeaders) {
        headers = firstRow.map((h: string) => h || 'Sin nombre');
        dataRows = parsedData.slice(1);
      } else {
        headers = firstRow.map((_: string, index: number) => `Columna ${index + 1}`);
        dataRows = parsedData;
      }

      const maxColumns = Math.max(...parsedData.map(row => row.length));
      if (headers.length < maxColumns) {
        for (let i = headers.length; i < maxColumns; i++) {
          headers.push(`Columna ${i + 1}`);
        }
      }

      const normalizedRows = dataRows.map(row => {
        const newRow = [...row];
        while (newRow.length < maxColumns) {
          newRow.push('');
        }
        return newRow;
      });

      const previewRows = normalizedRows.slice(0, 3);

      setFileData({
        headers,
        rows: previewRows,
        totalRows: normalizedRows.length,
        totalColumns: headers.length,
        hasHeaders
      });

      setShowModal(true);
      setLoading(false);
    } catch (err) {
      setError('Error al procesar el archivo. Asegúrate de que sea un archivo CSV válido.');
      setLoading(false);
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
  };

  const handleFileDrop = (e: React.DragEvent) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file && file.type === "text/csv") {
      setCatalogFile(file);
      processFile(file);
      toast.success(`Archivo ${file.name} cargado correctamente`);
    } else {
      toast.error("Por favor, sube un archivo CSV válido");
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setCatalogFile(file);
      processFile(file);
      toast.success(`Archivo ${file.name} cargado correctamente`);
    }
  };

  const handleSubmit = async () => {
    if (!catalogFile || !fileData) return;

    // Se realiza la petición exclusivamente al Backend de Inventario (Puerto 8000)
    const formData = new FormData();
    formData.append('file', catalogFile);
    
    try {
      const response = await fetch(`${import.meta.env.VITE_API_URL}/api/productos/cargar-csv/`, {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        throw new Error("Error al cargar el archivo en el servidor.");
      }

      toast.success("Catálogo procesado y cargado exitosamente");
      setShowModal(false);
      setTimeout(() => navigate("/simulator"), 1000);
    } catch (err) {
      toast.error("Ocurrió un error de red al intentar subir el archivo.");
    }
  };

  const handleReset = () => {
    setCatalogFile(null);
    setFileData(null);
    setError(null);
    setShowModal(false);
  };

  const handleCloseModal = () => {
    setShowModal(false);
  };

  const handleProcess = () => {
    if (!catalogFile) {
      toast.error("Por favor, carga el archivo antes de continuar");
      return;
    }
    setShowModal(true);
  };

  return (
    <div className="min-h-screen bg-background p-8">
      <div className="mx-auto max-w-2xl">
        <div className="mb-8 text-center">
          <h1 className="text-3xl font-bold text-foreground">Carga de Catálogo</h1>
          <p className="mt-2 text-muted-foreground">
            Sube tu archivo CSV de catálogo de productos
          </p>
        </div>

        {/* Upload Section */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Package className="h-5 w-5" />
              Subir Catálogo de Productos (CSV)
            </CardTitle>
            <CardDescription>
              Arrastra tu archivo o haz clic para seleccionar
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div
              className={`flex flex-col items-center justify-center rounded-lg border-2 border-dashed p-12 transition-colors ${
                catalogFile ? "border-primary bg-primary/5" : "border-muted-foreground/25 hover:border-primary/50"
              }`}
              onDrop={handleFileDrop}
              onDragOver={handleDragOver}
            >
              <UploadCloud className="mb-4 h-16 w-16 text-muted-foreground" />
              <p className="mb-2 text-lg font-medium">
                {catalogFile ? catalogFile.name : "Arrastra tu archivo aquí"}
              </p>
              <p className="text-sm text-muted-foreground">o haz clic para seleccionar</p>
              <input
                type="file"
                accept=".csv"
                className="hidden"
                onChange={handleFileSelect}
                id="catalog-upload"
              />
              <label
                htmlFor="catalog-upload"
                className="mt-6 cursor-pointer rounded-md bg-primary px-6 py-3 text-sm text-primary-foreground hover:bg-primary/90"
              >
                Seleccionar Archivo
              </label>
            </div>

            {error && (
              <div className="mt-4 flex items-center gap-2 rounded-lg bg-destructive/15 p-3 text-sm text-destructive">
                <AlertCircle className="h-4 w-4" />
                {error}
              </div>
            )}

            {loading && (
              <div className="mt-4 flex flex-col items-center gap-2 p-4">
                <div className="h-8 w-8 animate-spin rounded-full border-4 border-primary border-t-transparent"></div>
                <p className="text-sm text-muted-foreground">Procesando archivo...</p>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Process Button */}
        <div className="mt-8 flex justify-center">
          <Button
            size="lg"
            onClick={handleProcess}
            disabled={!catalogFile}
            className="px-12"
          >
            Procesar Datos
          </Button>
        </div>
      </div>

      {/* Modal de Confirmación */}
      {showModal && fileData && (
        <PreviewModal
          fileData={fileData}
          onClose={handleCloseModal}
          onSubmit={handleSubmit}
          onReset={handleReset}
        />
      )}
    </div>
  );
}