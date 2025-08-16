# train.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from utils import load_text, clean_text
from tokenizer import CharTokenizer
from model import TinyTransformer

# Device config
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load and clean dataset
file_path = "data/fairy_tales.txt"
raw_text = load_text(file_path)
cleaned_text = clean_text(raw_text)

# Tokenizer
tokenizer = CharTokenizer(cleaned_text)
vocab_size = tokenizer.vocab_size
print(f"Vocab size: {vocab_size}")

# Tokenize full text
encoded = tokenizer.encode(cleaned_text)

# Create input-output sequences
block_size = 64
X, Y = [], []

for i in range(len(encoded) - block_size):
    chunk = encoded[i:i + block_size + 1]
    X.append(chunk[:-1])
    Y.append(chunk[1:])

# Convert to tensors
X = torch.tensor(X, dtype=torch.long)
Y = torch.tensor(Y, dtype=torch.long)

# Create dataset and split
dataset = TensorDataset(X, Y)
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_data, val_data = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = DataLoader(val_data, batch_size=64)

# Load model
model = TinyTransformer(vocab_size, block_size).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
loss_fn = nn.CrossEntropyLoss()

# Training loop
epochs = 5
for epoch in range(epochs):
    model.train()
    total_loss = 0

    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = loss_fn(logits.view(-1, vocab_size), yb.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    # Validation loss
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = loss_fn(logits.view(-1, vocab_size), yb.view(-1))
            val_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_loss:.4f}, Val Loss: {val_loss/len(val_loader):.4f}")

torch.save(model.state_dict(), "model.pth")
