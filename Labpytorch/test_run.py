import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, confusion_matrix, precision_score,
    recall_score, f1_score, roc_curve, roc_auc_score
)

import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('mening missing 12.csv')
data = df.copy()
data.drop(columns=['Patient_ID'], inplace=True)

num_cols = ['Age', 'WBC_Count', 'Protein_Level', 'Glucose_Level',
            'Hemoglobin', 'WBC_Blood_Count', 'Platelets', 'CRP_Level']
for col in num_cols:
    data[col].fillna(data[col].median(), inplace=True)

cat_cols = ['Gender', 'Pathogen_Present', 'Diagnosis', 'Outcome']
for col in cat_cols:
    data[col].fillna(data[col].mode()[0], inplace=True)

le = LabelEncoder()
for col in cat_cols:
    data[col] = le.fit_transform(data[col])

y = (data['Risk_Level'] == 'High Risk').astype(int).values
X = data.drop(columns=['Risk_Level']).values.astype(np.float32)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

X_train_t = torch.tensor(X_train, dtype=torch.float32)
X_test_t  = torch.tensor(X_test,  dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)
y_test_t  = torch.tensor(y_test,  dtype=torch.float32)

train_dataset = TensorDataset(X_train_t, y_train_t)
train_loader  = DataLoader(train_dataset, batch_size=32, shuffle=True)

class Perceptron(nn.Module):
    def __init__(self, input_size: int):
        super(Perceptron, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # RETURN LOGITS
        return self.linear(x).squeeze(1)

model = Perceptron(input_size=X_train.shape[1])
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)

model.train()
for epoch in range(1, 10 + 1):
    epoch_loss = 0.0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * len(y_batch)
    print("Epoch", epoch, "Loss:", epoch_loss / len(train_dataset))
    
@torch.no_grad()
def evaluate(model: nn.Module, X: torch.Tensor, threshold: float = 0.5) -> np.ndarray:
    model.eval()
    probs = torch.sigmoid(model(X)).cpu().numpy()
    return (probs >= threshold).astype(int)

y_pred = evaluate(model, X_test_t)
acc_test  = accuracy_score(y_test,  y_pred)
print("Accuracy test:", acc_test)
