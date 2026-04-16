import json

file_path = 'metricas_clasificacion_pytorch.ipynb'
with open(file_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if 'optim.SGD' in source:
            new_source = source.replace("optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)", "optim.Adam(model.parameters(), lr=1e-3)")
            cell['source'] = [line + '\n' for line in new_source.split('\n')]
        if 'nn.BCELoss()' in source:
            # We must fix BCELoss to BCEWithLogitsLoss to avoid sigmoid bounds error
            new_source = "".join(cell['source']).replace("nn.BCELoss()", "nn.BCEWithLogitsLoss()")
            cell['source'] = [line + '\n' for line in new_source.split('\n')]
        if 'torch.sigmoid(self.linear(x)).squeeze(1)' in source:
            new_source = "".join(cell['source']).replace("torch.sigmoid(self.linear(x)).squeeze(1)", "self.linear(x).squeeze(1)")
            cell['source'] = [line + '\n' for line in new_source.split('\n')]
        if 'probs = model(X).cpu().numpy()' in source:
            new_source = "".join(cell['source']).replace("probs = model(X).cpu().numpy()", "probs = torch.sigmoid(model(X)).cpu().numpy()")
            cell['source'] = [line + '\n' for line in new_source.split('\n')]

with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
print("Notebook model math patched.")
