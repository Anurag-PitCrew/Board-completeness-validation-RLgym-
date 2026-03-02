import torch
if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
    print("MPS device found! Training on GPU.")
else:
    device = torch.device("cpu")
    print("MPS device not found. Training on CPU.")
