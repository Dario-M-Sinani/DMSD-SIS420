// src/pages/Simulador/components/ProductTable.tsx

import React from 'react';
import type { Product } from '@/types';
import styles from './ProductTable.module.css';

interface ProductTableProps {
  products: Product[];
  selectedProducts: string[];
  onToggleProduct: (code: string) => void;
  onToggleAll: () => void;
}

const ProductTable: React.FC<ProductTableProps> = ({
  products,
  selectedProducts,
  onToggleProduct,
  onToggleAll
}) => {
  const allSelected = products.length > 0 && selectedProducts.length === products.length;

  const getSensitivityClass = (sensitivity: string) => {
    switch (sensitivity) {
      case 'Alta':
        return styles.badgeHigh;
      case 'Media':
        return styles.badgeMedium;
      case 'Baja':
        return styles.badgeLow;
      default:
        return '';
    }
  };

  return (
    <div className={styles.tableContainer}>
      <table className={styles.table}>
        <thead>
          <tr>
            <th className={styles.th}>
              <input
                type="checkbox"
                checked={allSelected}
                onChange={onToggleAll}
                className={styles.checkbox}
              />
            </th>
            <th className={styles.th}>Código</th>
            <th className={styles.th}>Producto</th>
            <th className={styles.th}>País de Origen</th>
            <th className={styles.th}>Lead Time (días)</th>
            <th className={styles.th}>Cantidad Mínima</th>
            <th className={styles.th}>Sensibilidad</th>
            <th className={styles.th}>Stock</th>
          </tr>
        </thead>
        <tbody>
          {products.map(product => (
            <tr key={product.code} className={styles.tr}>
              <td className={styles.td}>
                <input
                  type="checkbox"
                  checked={selectedProducts.includes(product.code)}
                  onChange={() => onToggleProduct(product.code)}
                  className={styles.checkbox}
                />
              </td>
              <td className={styles.td}>{product.code}</td>
              <td className={styles.td}>{product.name}</td>
              <td className={styles.td}>{product.country}</td>
              <td className={styles.td}>{product.leadTime}</td>
              <td className={styles.td}>{product.minQuantity}</td>
              <td className={styles.td}>
                <span className={`${styles.badge} ${getSensitivityClass(product.pricesensitivity)}`}>
                  {product.pricesensitivity}
                </span>
              </td>
              <td className={styles.td}>{product.stock}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default ProductTable;