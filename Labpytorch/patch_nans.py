import json

file_path = 'metricas_clasificacion_pytorch.ipynb'
with open(file_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if "data.fillna(data.median(), inplace=True)" in source or "data[col].fillna(data[col].median(), inplace=True)" in source:
            # force numerical imputation correctly avoiding chained assignment issues or object dtype issues:
            new_source = source.replace(
                "data[col].fillna(data[col].median(), inplace=True)", 
                "data[col] = pd.to_numeric(data[col], errors='coerce')\n    data[col] = data[col].fillna(data[col].median())"
            )
            cell['source'] = [line + '\n' for line in new_source.split('\n')]

with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
print("Notebook NaNs patched.")
