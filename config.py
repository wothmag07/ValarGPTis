import torch

class Config():
    batchSize = 64 # no of independent sequence the model process in parallel
    blockSize = 256 # maximum context length for prediction
    max_iters = 5000
    eval_interval = 300
    lr = 3e-4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    eval_iters = 200
    noOfEmbeddings = 32
    numHeads = 6
    numLayers = 6

C = Config()