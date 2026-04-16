import json

file_path = 'metricas_clasificacion_pytorch.ipynb'
with open(file_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if "pd.read_csv('mening_missing_12.csv')" in source:
            new_source = source.replace("pd.read_csv('mening_missing_12.csv')", "pd.read_csv('mening missing 12.csv')")
            cell['source'] = [line + '\n' for line in new_source.split('\n')]
            
# Also fix any other possible missing dependencies issues that were not patched
with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
print("Notebook filename fixed.")
