from dataclasses import dataclass
import torch
from torch import nn

class Memory(nn.Module):
    def __init__(self,high,low,dims,size,device,interval) -> None:
        self.memory = torch.Tensor([torch.rand(dims) * (high - low) + low, [0.0,0.0,0.0] for _ in range(size)], device=device)
        self.interval=interval

    def step(self,inputs):
        self.state = inputs # placeholder
        return self.state