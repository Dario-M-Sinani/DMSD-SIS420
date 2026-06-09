import styles from './PreviewModal.module.css';

interface FileData {
  headers: string[];
  rows: any[][];
  totalRows: number;
  totalColumns: number;
  hasHeaders: boolean;
}

interface PreviewModalProps {
  fileData: FileData;
  onClose: () => void;
  onSubmit: () => void;
  onReset: () => void;
}

const PreviewModal: React.FC<PreviewModalProps> = ({ fileData, onClose, onSubmit, onReset }) => {
  return (
    <>
      <div className={styles.modalOverlay} onClick={onClose}></div>
      <div className={styles.modal}>
        <div className={styles.modalHeader}>
          <h2>Vista previa del archivo</h2>
          <button className={styles.closeButton} onClick={onClose}>
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <line x1="18" y1="6" x2="6" y2="18" />
              <line x1="6" y1="6" x2="18" y2="18" />
            </svg>
          </button>
        </div>

        <div className={styles.modalBody}>
          <div className={styles.fileInfo}>
            <span className={styles.badge}>
              {fileData.totalRows} {fileData.totalRows === 1 ? 'fila' : 'filas'}
            </span>
            <span className={styles.badge}>
              {fileData.totalColumns} {fileData.totalColumns === 1 ? 'columna' : 'columnas'}
            </span>
            {!fileData.hasHeaders && (
              <span className={styles.badgeWarning}>
                Sin encabezados detectados
              </span>
            )}
          </div>

          <div className={styles.tableContainer}>
            <table className={styles.table}>
              <thead>
                <tr>
                  {fileData.headers.map((header, index) => (
                    <th key={index}>{header}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {fileData.rows.map((row, rowIndex) => (
                  <tr key={rowIndex}>
                    {row.map((cell, cellIndex) => (
                      <td key={cellIndex}>{cell || '-'}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {fileData.totalRows > 3 && (
            <div className={styles.moreRows}>
              Mostrando 3 de {fileData.totalRows} filas
            </div>
          )}
        </div>

        <div className={styles.modalFooter}>
          <button className={styles.btnSecondary} onClick={onReset}>
            Cancelar
          </button>
          <button className={styles.btnPrimary} onClick={onSubmit}>
            Cargar datos
          </button>
        </div>
      </div>
    </>
  );
};

export default PreviewModal;