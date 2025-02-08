import torch
from torch.utils.data import Dataset
from config import C  # Import the config file

class CharDatasetBase(Dataset):
    """
    Emits batches of characters based on the configuration.
    """

    def __init__(self, data):
        self.config = C  # Load configuration from config.py
        self.blockSize = self.config.blockSize

        # Create character vocabulary
        chars = sorted(list(set(data)))
        self.vocabSize = len(chars)

        print(f"Data has {len(data)} characters, {self.vocabSize} unique.")

        # Character mappings
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

        self.data = data

    def get_vocabSize(self):
        return self.vocabSize

    def get_blockSize(self):
        return self.blockSize

    def __len__(self):
        return len(self.data) - self.blockSize

    def __getitem__(self, idx):
        # Extract a chunk of text (blockSize + 1) for input-output pairs
        chunk = self.data[idx : idx + self.blockSize + 1]

        # Convert characters to token IDs
        dix = [self.stoi[s] for s in chunk]

        # Input and target tensors
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)

        return x, y


