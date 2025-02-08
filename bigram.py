
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


batchSize = 32 # no of independent sequence the model process in parallel
blockSize = 8 # maximum context length for prediction
max_iters = 3000
eval_interval = 300
lr = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200

torch.manual_seed(1709)

txt_file = r"D:\Workspace\nanoGPT\got_script.txt"

with open(txt_file, 'r', encoding='utf-8') as f:
    text = f.read()


chars = sorted(list(set(text)))
vocabSize = len(chars)

#Token mapping

# Define the allowed character set
unk_token = "<unk>"

# Create mappings
token2id = {s: i for i, s in enumerate(chars)}
token2id[unk_token] = len(chars)  # Assign the next available index to <unk>

id2token = {i: s for s, i in token2id.items()}
id2token[len(chars)] = unk_token  # Reverse mapping for <unk>

# Encoding function with unknown token handling
def encode(token):
    return [token2id.get(char, token2id[unk_token]) for char in token]

# Decoding function
def decode(ids):
    return ''.join(id2token.get(i, unk_token) for i in ids)

text_tensor = torch.tensor(encode(text), dtype=torch.long)
splitSize = int(0.9 * len(text))
trainTensor = text_tensor[:splitSize]
validTensor = text_tensor[splitSize:]

def getBatch(splitType):

    #generate a small batch of inputs and targets
    data = trainTensor if splitType=="train" else validTensor
    ix = torch.randint(0, len(data) - blockSize, (batchSize,))
    x = torch.stack([data[i:i+blockSize] for i in ix])
    y = torch.stack([data[i+1:i+blockSize+1] for i in ix])

    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()

def estimateLoss():
    output = {}
    model.eval()
    for split in ["train", "valid"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = getBatch(split)
            logits, loss = model(X,Y)
            losses[k] = loss.item()
        output[split] = losses.mean()
    model.train()
    return output


class BigramLM(nn.Module):
    def __init__(self, vocabSize):
        super(BigramLM, self).__init__()
        self.tokenEmbeddings = nn.Embedding(vocabSize, vocabSize)  # (batch, time, channel)

    def forward(self, idx, targets=None):
        logits = self.tokenEmbeddings(idx)  # (batch, time, channel)

        if targets is not None:
            batch, time, channel = logits.shape
            logits = logits.view(batch * time, channel)  # Flatten for cross-entropy
            targets = targets.view(-1)  # Flatten targets
            loss = F.cross_entropy(logits, targets)
        else:
            loss = None

        return logits, loss  # Ensure logits are always returned

    def generate(self, idx, maxNewTokens):
        """Generate new tokens based on input idx"""
        for _ in range(maxNewTokens):
            logits, _ = self(idx)  # Get predictions
            logits = logits[:, -1, :]  # Take last token's logits
            probs = F.softmax(logits, dim=-1)  # Apply softmax
            idxNext = torch.multinomial(probs, num_samples=1)  # Sample from probs
            idx = torch.cat((idx, idxNext), dim=1)  # Append new token
        return idx

model = BigramLM(vocabSize)
model.to(device)
        
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

for itr in range(max_iters):

    if itr % eval_interval == 0:
        losses = estimateLoss()
        print(f"Step {itr}: train loss {losses['train']:.4f}, val loss {losses['valid']:.4f}")

    xBatch, yBatch = getBatch('train')

    logits, loss = model(xBatch, yBatch)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    
context = torch.zeros((1,1), dtype=torch.long)
print(decode(model.generate(context, maxNewTokens=1000)[0].tolist()))