import torch
import torch.nn as nn
from typing import Any

from typing import Any
from torch import nn

import asyncio

class Cognition(nn.Module):
    def __init__(self, model: nn.Module,name:str, interval:float):
        super().__init__()
        self.model = model
        self.gates:dict[str,nn.Module]={}
        self.entry:dict[str,Any]={}
        self.name=name
        self.interval=interval
        self.y=None
        self.hidden=None
        if self.interval<=0:
            raise ValueError("Interval must be positive.")
    def init_hidden(self,hidden):
        raise NotImplementedError
    def _fuse(self,inputs:dict[str,Any],hidden:Any)->torch.Tensor:
        raise NotImplementedError
    def _unfuse(self,input:torch.Tensor)->tuple[dict[str,Any],Any]:
        raise NotImplementedError
    def forward(self,X:dict[str,Any]):
        y=self._unfuse(self.model.forward(self._fuse(X,self.hidden)))
        self.hidden=y[1]
        for a,b in y[0].items():
            y[0][a]=self.gates[a].forward(b)
        return y

    async def run(self, stopflag:asyncio.Event):
        X=self.entry.copy()
        while not stopflag.is_set:
            self.y = await asyncio.to_thread(self, X)
            try:
                await asyncio.wait_for(stopflag.wait(), timeout=self.interval)
            except TimeoutError:
                pass