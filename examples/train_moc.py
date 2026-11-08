from ..src.alpaca.moc import MoC, Memory, Cognition
import torch.nn as nn

class MLPCognition(Cognition):
    def __init__(self, model: nn.Module, name: str, interval: float):
        super().__init__(model, name, interval)
        #WIP NOT IMPLEMENTED
        raise NotImplementedError