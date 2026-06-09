import { useState, useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { getProducts, Product } from "@/services/productService";
import { Command, CommandInput, CommandList, CommandEmpty, CommandGroup, CommandItem } from "@/components/ui/command";
import { Popover, PopoverTrigger, PopoverContent } from "@/components/ui/popover";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { X, ChevronsUpDown } from "lucide-react";

interface ProductSearchProps {
  selectedProducts: Product[];
  onSelectionChange: (product: Product) => void;
}

export default function ProductSearch({ selectedProducts, onSelectionChange }: ProductSearchProps) {
  const [open, setOpen] = useState(false);
  const { data: products, isLoading, isError } = useQuery<Product[]>({
    queryKey: ["products"],
    queryFn: getProducts,
  });

  const selectedProductCodes = useMemo(() => new Set(selectedProducts.map(p => p.codigo)), [selectedProducts]);

  return (
    <div className="w-full">
      <Popover open={open} onOpenChange={setOpen}>
        <PopoverTrigger asChild>
          <Button
            variant="outline"
            role="combobox"
            aria-expanded={open}
            className="w-full justify-between h-auto min-h-10"
          >
            <div className="flex flex-wrap gap-1">
              {selectedProducts.length > 0 ? (
                selectedProducts.map(product => (
                  <Badge
                    variant="secondary"
                    key={product.id}
                    className="mr-1"
                    onClick={(e) => {
                      e.stopPropagation();
                      onSelectionChange(product);
                    }}
                  >
                    {product.nombre}
                    <X className="ml-1 h-3 w-3" />
                  </Badge>
                ))
              ) : (
                "Seleccionar productos..."
              )}
            </div>
            <ChevronsUpDown className="ml-2 h-4 w-4 shrink-0 opacity-50" />
          </Button>
        </PopoverTrigger>
        <PopoverContent className="w-[--radix-popover-trigger-width] p-0">
          <Command>
            <CommandInput placeholder="Buscar producto..." />
            <CommandList>
              <CommandEmpty>
                {isLoading ? "Cargando..." : isError ? "Error al cargar" : "No se encontraron productos."}
              </CommandEmpty>
              <CommandGroup>
                {products?.map((product) => (
                  <CommandItem
                    key={product.id}
                    value={product.nombre}
                    onSelect={() => {
                      onSelectionChange(product);
                    }}
                    className={selectedProductCodes.has(product.codigo) ? "bg-accent" : ""}
                  >
                    {product.nombre} ({product.codigo})
                  </CommandItem>
                ))}
              </CommandGroup>
            </CommandList>
          </Command>
        </PopoverContent>
      </Popover>
    </div>
  );
}
