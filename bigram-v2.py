
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


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

class Head(nn.Module):
    def __init__(self, headSize):
        super(Head, self).__init__()
        self.query = nn.Linear(noOfEmbeddings, headSize)
        self.key = nn.Linear(noOfEmbeddings, headSize)
        self.value = nn.Linear(noOfEmbeddings, headSize)
        self.register_buffer('tril', torch.tril(torch.ones(blockSize, blockSize)))
    def forward(self, X):
        B, T, C = X.shape
        k = self.key(X) # (B, T, headSize)
        q = self.query(X) # (B, T, headSize)
        v = self.value(X)

        weights = q @ k.transpose(-2, -1) * C**-0.5 # (B, T, headSize) * (B, headSize, T) -> (B, T, T)

        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        # print(weights)
        weights = F.softmax(weights, dim=-1)
        output = weights @ v

        return output

class MultiHeadAttention(nn.Module):
    def __init__(self, headSize, noOfHeads):
        super(MultiHeadAttention, self).__init__()
        self.heads = nn.ModuleList([Head(headSize) for _ in range(noOfHeads)])
        self.projection = nn.Linear(noOfEmbeddings, noOfEmbeddings)
    
    def forward(self, x):
        output = []
        for head in self.heads:
            output.append(head(x))
        out = torch.cat(output, dim=-1)
        return self.projection(out)

class FeedForwardNetwork(nn.Module):
    def __init__(self, hiddenSize):
        super(FeedForwardNetwork, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(hiddenSize, 4*hiddenSize),
                                 nn.ReLU(),
                                 nn.Linear(4*hiddenSize, hiddenSize))
    def forward(self, x):
        return self.fc1(x)
        

    
class Block(nn.Module):
    def __init__(self, noOfEmbeddings, noOfHeads):
        super(Block, self).__init__()
        headSize = noOfEmbeddings//noOfHeads
        self.selfAttention = MultiHeadAttention(headSize=headSize, noOfHeads=noOfHeads)
        self.feedForward = FeedForwardNetwork(hiddenSize=noOfEmbeddings)
    def forward(self, x):
        x = x + self.selfAttention(x)
        output = x + self.feedForward(x)
        return output
        

class BigramLM(nn.Module):
    def __init__(self):
        super(BigramLM, self).__init__()
        self.tokenEmbeddings = nn.Embedding(vocabSize, noOfEmbeddings)  # (batch, time, channel)
        self.posEmbeddings = nn.Embedding(blockSize, noOfEmbeddings)
        self.saHead = nn.Sequential(Block(noOfEmbeddings=noOfEmbeddings, noOfHeads=numHeads),
                                    Block(noOfEmbeddings=noOfEmbeddings, noOfHeads=numHeads),
                                    Block(noOfEmbeddings=noOfEmbeddings, noOfHeads=numHeads),
                                    Block(noOfEmbeddings=noOfEmbeddings, noOfHeads=numHeads))
        
        # self.saHead = MultiHeadAttention(noOfHeads=numHeads , headSize=noOfEmbeddings//numHeads) # 4 heads of 8 dimensional self attention
        self.ffn = FeedForwardNetwork(noOfEmbeddings)
        self.lmHead = nn.Linear(noOfEmbeddings, vocabSize)

    def forward(self, idx, targets=None):
        tokenEmbeddings = self.tokenEmbeddings(idx)  # (batch, time, channel)
        # B, T = idx.shape
        posEmbeddings = self.posEmbeddings(torch.arange(idx.shape[1], device=device).unsqueeze(0)) #(time, channel)
        x = tokenEmbeddings + posEmbeddings
        x = self.saHead(x)
        logits = self.lmHead(x)  # (batch, time, vocab)

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
            #crop idx to last blockSize token
            idxCond = idx[:, -blockSize:]
            logits, _ = self(idxCond)  # Get predictions
            logits = logits[:, -1, :]  # Take last token's logits
            probs = F.softmax(logits, dim=-1)  # Apply softmax
            idxNext = torch.multinomial(probs, num_samples=1)  # Sample from probs
            idx = torch.cat((idx, idxNext), dim=1)  # Append new token
        return idx

model = BigramLM()
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