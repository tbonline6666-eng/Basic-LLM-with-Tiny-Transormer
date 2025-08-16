# generate.py

import torch
from model import TinyTransformer
from tokenizer import CharTokenizer
from utils import load_text, clean_text

# Config
device = 'cuda' if torch.cuda.is_available() else 'cpu'
block_size = 64

# Load and clean data (only for tokenizer)
text = load_text("data/fairy_tales.txt")
text = clean_text(text)
tokenizer = CharTokenizer(text)
vocab_size = tokenizer.vocab_size

# Load model
model = TinyTransformer(vocab_size, block_size).to(device)
model.load_state_dict(torch.load("model.pth"))  # (save it after training)
model.eval()

# Generate function
@torch.no_grad()
def generate(prompt, max_new_tokens=100):
    model.eval()
    idx = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long).to(device)

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]
        logits = model(idx_cond)
        logits = logits[:, -1, :]  # last timestep
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, next_id), dim=1)

    return tokenizer.decode(idx[0].tolist())

# Example
prompt = "once upon a time"
generated_text = generate(prompt, max_new_tokens=200)
print("\n📜 Generated Text:\n")
print(generated_text)
