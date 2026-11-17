import asyncio
from collections.abc import Callable, Sequence
from math import prod
from typing import Any

import torch
from torch import nn


class Cognition(nn.Module):
    def __init__(self, model: nn.Module,name:str, interval:float, shape:dict[str,int]|None=None):
        super().__init__()
        self.model = model
        # Destination cognition name -> width of the output sent to it.
        self.shape:dict[str,int]={} if shape is None else shape.copy()
        self.gates: nn.ModuleDict = nn.ModuleDict()
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
        y=self._unfuse(self.model(self._fuse(X,self.hidden)))
        self.hidden=y[1]
        for a,b in y[0].items():
            y[0][a]=self.gates[a](b)
        return y[0]

    async def run(self, stopflag:asyncio.Event):
        while not stopflag.is_set():
            X=self.entry.copy()
            self.y = await asyncio.to_thread(self, X)
            try:
                await asyncio.wait_for(stopflag.wait(), timeout=self.interval)
            except TimeoutError:
                pass


def _linear_stack(layers: Sequence[tuple[int, int, float]],expected_width: int,activation: Callable[[], nn.Module]) -> tuple[list[nn.Module], int]:
    modules: list[nn.Module] = []
    for index, (input_size, output_size, dropout) in enumerate(layers, start=1):
        if input_size <= 0 or output_size <= 0:
            raise ValueError(f"Layer {index} must have positive input and output widths.")
        if input_size != expected_width:
            raise ValueError(f"Layer {index} expects {expected_width} inputs, got {input_size}.")
        if not 0.0 <= dropout <= 1.0:
            raise ValueError(f"Layer {index} dropout probability must be between 0 and 1.")

        modules.extend((nn.Linear(input_size, output_size), activation()))
        if dropout > 0:
            modules.append(nn.Dropout(dropout))
        expected_width = output_size
    return modules, expected_width


class MLPCognition(Cognition):
    """Apply a feedforward stack to the concatenated cognition inputs."""
    def __init__(self,name: str,interval: float,incoming: dict[str, int],outgoing: dict[str, int],layers: Sequence[tuple[int, int, float]],activation: Callable[[], nn.Module] = nn.ReLU):
        if not layers:
            raise ValueError("layers must contain at least one layer.")

        modules, output_width = _linear_stack(layers, sum(incoming.values()), activation)
        super().__init__(
            model=nn.Sequential(*modules),
            name=name,
            interval=interval,
            shape=outgoing,
        )
        self.input_names = tuple(sorted(incoming))
        self.gates = nn.ModuleDict({
            target: nn.Linear(output_width, width)
            for target, width in self.shape.items()
        })

    def init_hidden(self, hidden: object = None) -> None:
        self.hidden = None

    def _fuse(self, inputs: dict[str, torch.Tensor], hidden: object) -> torch.Tensor:
        return torch.cat([inputs[source] for source in self.input_names], dim=-1)

    def _unfuse(self, input: torch.Tensor) -> tuple[dict[str, torch.Tensor], None]:
        return {target: input for target in self.gates}, None


_SpatialSize = int | tuple[int, int]
_Conv2dLayer = tuple[int, int, _SpatialSize, _SpatialSize, _SpatialSize, float]


class CNNCognition(Cognition):
    """Encode named images with Conv2d blocks and optional dense layers.

    Declare each incoming image as (channels, height, width). Inputs can be
    image tensors or flattened MoC port vectors, with an optional batch axis.
    Each convolution is (in_channels, out_channels, kernel, stride, padding, dropout).
    Dense layers follow adaptive average pooling and use the MLP layer format.
    """

    def __init__(self,name: str,interval: float,incoming: dict[str, tuple[int, int, int]],outgoing: dict[str, int],conv_layers: Sequence[_Conv2dLayer],layers: Sequence[tuple[int, int, float]] = (),activation: Callable[[], nn.Module] = nn.ReLU,pool_size: tuple[int, int] = (1, 1)) -> None:
        if not incoming or any(len(shape) != 3 or min(shape) <= 0 for shape in incoming.values()):
            raise ValueError("incoming must declare positive (channels, height, width) shapes.")
        if len({shape[1:] for shape in incoming.values()}) != 1:
            raise ValueError("Incoming images must have the same height and width.")
        if not conv_layers:
            raise ValueError("conv_layers must contain at least one convolution.")
        if len(pool_size) != 2 or min(pool_size) <= 0:
            raise ValueError("pool_size must contain a positive height and width.")

        modules: list[nn.Module] = []
        expected_channels = sum(shape[0] for shape in incoming.values())
        for index, spec in enumerate(conv_layers, start=1):
            in_channels, out_channels, kernel, stride, padding, dropout = spec
            if in_channels != expected_channels or out_channels <= 0:
                raise ValueError(f"Convolution {index} requires {expected_channels} input channels and a positive output channel count.")
            if not 0.0 <= dropout <= 1.0:
                raise ValueError(f"Convolution {index} dropout probability must be between 0 and 1.")

            modules.extend((
                nn.Conv2d(in_channels, out_channels, kernel, stride=stride, padding=padding),
                activation(),
            ))
            if dropout > 0:
                modules.append(nn.Dropout(dropout))
            expected_channels = out_channels

        modules.extend((nn.AdaptiveAvgPool2d(pool_size), nn.Flatten(start_dim=-3)))
        dense_modules, output_width = _linear_stack(layers, expected_channels * prod(pool_size), activation)
        modules.extend(dense_modules)

        super().__init__(nn.Sequential(*modules), name, interval, shape=outgoing)
        self.input_shapes = dict(sorted(incoming.items()))
        self.gates = nn.ModuleDict({
            target: nn.Linear(output_width, width)
            for target, width in self.shape.items()
        })

    def init_hidden(self, hidden: object = None) -> None:
        self.hidden = None

    def _fuse(self, inputs: dict[str, torch.Tensor], hidden: object) -> torch.Tensor:
        images: list[torch.Tensor] = []
        for source, shape in self.input_shapes.items():
            image = inputs[source]
            if image.ndim in (1, 2) and image.shape[-1] == prod(shape):
                image = image.reshape(*image.shape[:-1], *shape)
            elif image.ndim not in (3, 4) or image.shape[-3:] != shape:
                raise ValueError(f"Input {source!r} must have image shape {shape} or flat width {prod(shape)}.")
            images.append(image)
        return torch.cat(images, dim=-3)

    def _unfuse(self, input: torch.Tensor) -> tuple[dict[str, torch.Tensor], None]:
        return {target: input for target in self.gates}, None


class GRUCognition(Cognition):
    def __init__(self, name, interval, incoming, outgoing, hidden_size):
        super().__init__(model=nn.GRUCell(sum(incoming.values()), hidden_size),name=name,interval=interval,shape=outgoing)
        self.input_names = tuple(sorted(incoming))
        self.gates = nn.ModuleDict({
            target: nn.Linear(hidden_size, width)
            for target, width in self.shape.items()
        })

    def init_hidden(self, hidden=None):
        self.hidden = hidden

    def forward(self, inputs):
        x = torch.cat([inputs[source] for source in self.input_names])
        self.hidden = self.model(x, self.hidden)
        return {
            target: projection(self.hidden)
            for target, projection in self.gates.items()
        }
