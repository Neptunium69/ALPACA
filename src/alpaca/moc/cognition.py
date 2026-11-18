import asyncio
from collections.abc import Callable, Mapping, Sequence
from math import prod
import math
from typing import Any, cast

import torch
from torch import nn

class Gate(nn.Module):
    def __init__(self, bandwidth: int, dims: int, heads: int):
        super().__init__()
        self.token_shape = (bandwidth, dims)
        self.context_projection = nn.LazyLinear(bandwidth * dims)
        self.slot_embeddings = nn.Parameter(torch.randn(bandwidth, dims) * 0.02)
        self.attention = nn.TransformerEncoderLayer(d_model=dims, nhead=heads, dim_feedforward=4 * dims,dropout=0.0, batch_first=True, norm_first=True)

    def forward(self, tokens, sender_hidden, receiver_hidden):
        if tokens.ndim not in (2, 3) or tokens.shape[-2:] != self.token_shape:
            raise ValueError(f"Expected tokens shaped [..., {self.token_shape[0]}, {self.token_shape[1]}].")
        batch_shape = tokens.shape[:-2]
        sender_hidden = sender_hidden.to(tokens).expand(*batch_shape, -1)
        receiver_hidden = receiver_hidden.to(tokens).expand(*batch_shape, -1)
        context = torch.cat([sender_hidden, receiver_hidden], dim=-1)
        context = self.context_projection(context).unflatten(-1, self.token_shape)
        return self.attention(tokens + context + self.slot_embeddings)


class Cognition(nn.Module):
    """General class for cognitions."""
    hidden: torch.Tensor
    inputs_tokens = True
    outputs_tokens = True

    def __init__(self, model: nn.Module, name: str, interval: float, hidden_shape: int = 1, shape: dict[str, int] | None = None, *, gates: Mapping[str, int | tuple[int, int]] | None = None):
        super().__init__()
        self.model = model
        self.shape: dict[str, int] = {}
        self._gates = nn.ModuleDict()
        self._token_heads = nn.ModuleDict()
        self.entry:dict[str,Any]={}
        self.name=name
        self.interval=interval
        self.y=None
        reference = next(model.parameters(), None)
        hidden = torch.zeros(hidden_shape) if reference is None else reference.new_zeros(hidden_shape)
        self.register_buffer("hidden", hidden)
        self.add_gates(gates if gates is not None else (shape or {}))

    def add_gates(self, gates: Mapping[str, int | tuple[int, int]], *, heads: int = 1) -> None:
        additions = {}
        token_heads = {}
        widths = {}
        for target, layout in gates.items():
            if target.startswith("HIDDEN_"):
                raise ValueError("HIDDEN_ prefix is reserved. Cognition names and gate names must not start with HIDDEN_.")
            if target in self._gates:
                raise ValueError(f"Gate {target!r} already exists.")
            if type(layout) is not int and (not isinstance(layout, tuple) or len(layout) != 2):
                raise ValueError("Gate layouts must be a width or (token slots, token width).")
            slots, width = layout if isinstance(layout, tuple) else (1, layout)
            if any(type(value) is not int or value <= 0 for value in (slots, width)) or width % heads:
                raise ValueError("Token dimensions must be positive and width divisible by heads.")
            gate = Gate(slots, width, heads)
            gate.to(device=self.hidden.device, dtype=self.hidden.dtype)
            gate.train(self.training)
            additions[target] = gate
            widths[target] = slots * width
            if not self.outputs_tokens:
                token_heads[target] = nn.Sequential(
                    nn.Linear(self.hidden.shape[-1], slots * width),
                    nn.Unflatten(-1, (slots, width)),
                ).to(self.hidden)
        self._gates.update(additions)
        self._token_heads.update(token_heads)
        self.shape.update(widths)

    def token_shape(self, target: str) -> tuple[int, int]:
        return cast(Gate, self._gates[target]).token_shape

    def remove_gates(self, names: Sequence[str]) -> None:
        """Remove gates and their output ports while MoC is stopped."""
        missing = set(names).difference(self._gates)
        if missing:
            raise KeyError(f"Unknown gates: {sorted(missing)}")
        for name in dict.fromkeys(names):
            del self._gates[name]
            if name in self._token_heads:
                del self._token_heads[name]
            del self.shape[name]
            if self.y is not None:
                self.y.pop(name, None)

    def init_hidden(self, hidden: torch.Tensor | None = None) -> None:
        if hidden is not None and (hidden.ndim not in (1, 2) or hidden.shape[-1] != self.hidden.shape[-1]):
            raise ValueError("Hidden state width must match hidden_shape.")
        self.hidden = torch.zeros_like(self.hidden) if hidden is None else hidden.to(self.hidden)

    @staticmethod
    def _communication_inputs(inputs):
        return {name: value for name, value in inputs.items() if not name.startswith("HIDDEN_")}

    def _gate_outputs(self, outputs, inputs):
        return {target: self._gates[target](values, inputs["HIDDEN_" + self.name], inputs["HIDDEN_" + target]) for target, values in outputs.items()}
    def _fuse(self,inputs:dict[str,Any],hidden:Any)->torch.Tensor:
        raise NotImplementedError
    def _unfuse(self,input:torch.Tensor)->tuple[dict[str,Any],Any]:
        """Return one token tensor per destination and the next hidden state."""
        raise NotImplementedError
    def forward(self,X:dict[str,Any]):
        y=self._unfuse(self.model(self._fuse(self._communication_inputs(X),self.hidden)))
        self.hidden=y[1]
        return self._gate_outputs(y[0], X)

    async def run(self, stopflag: asyncio.Event, on_step: Callable[[str], None] | None = None):
        while not stopflag.is_set():
            X=self.entry.copy()
            self.y = await asyncio.to_thread(self, X)
            if on_step is not None:
                on_step(self.name)
            try:
                await asyncio.wait_for(stopflag.wait(), timeout=self.interval)
            except TimeoutError:
                pass

def linstack(layers: Sequence[tuple[int, int, float]],expected_width: int,activation: Callable[[], nn.Module]) -> tuple[list[nn.Module], int]:
    """Creates a stack of nn.Linear."""
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


class Cog1D(Cognition):
    inputs_tokens = False
    outputs_tokens = False

    def _unfuse(self, input: torch.Tensor) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        return {target: head(input) for target, head in self._token_heads.items()}, input

    def _load_from_state_dict(self, state_dict, prefix, *args) -> None:
        for target in self._token_heads:
            for parameter in ("weight", "bias"):
                legacy_key = f"{prefix}_gates.{target}.projection.{parameter}"
                if legacy_key in state_dict:
                    state_dict.setdefault(f"{prefix}_token_heads.{target}.0.{parameter}", state_dict.pop(legacy_key))
        super()._load_from_state_dict(state_dict, prefix, *args)


class MLPCognition(Cog1D):
    """Apply a feedforward stack to the concatenated cognition inputs."""
    def __init__(self,name: str,interval: float,incoming: dict[str, int],outgoing: Mapping[str, int | tuple[int, int]],layers: Sequence[tuple[int, int, float]],activation: Callable[[], nn.Module] = nn.ReLU):
        if not layers:
            raise ValueError("layers must contain at least one layer.")

        modules, output_width = linstack(layers, sum(incoming.values()), activation)
        super().__init__(model=nn.Sequential(*modules),name=name,interval=interval,hidden_shape=output_width,gates=outgoing,)
        self.input_names = tuple(sorted(incoming))

    def _fuse(self, inputs: dict[str, torch.Tensor], hidden: object) -> torch.Tensor:
        return torch.cat([inputs[source] for source in self.input_names], dim=-1)

_SpatialSize = int | tuple[int, int]
_Conv2dLayer = tuple[int, int, _SpatialSize, _SpatialSize, _SpatialSize, float]


class CNNCognition(Cog1D):
    """Inputs [batch, channels, height, width] directly or reshapes a flat vector. Outputs regular MoC compatible tokens."""
    def __init__(self,name: str,interval: float,incoming: dict[str, tuple[int, int, int]],outgoing: Mapping[str, int | tuple[int, int]],conv_layers: Sequence[_Conv2dLayer],layers: Sequence[tuple[int, int, float]] = (),activation: Callable[[], nn.Module] = nn.ReLU,pool_size: tuple[int, int] = (1, 1)) -> None:
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
        dense_modules, output_width = linstack(layers, expected_channels * prod(pool_size), activation)
        modules.extend(dense_modules)

        super().__init__(nn.Sequential(*modules), name, interval, hidden_shape=output_width, gates=outgoing)
        self.input_shapes: dict[str, tuple[int, int, int]] = dict(sorted(incoming.items()))

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

class GRUCognition(Cog1D):
    def __init__(self, name, interval, incoming, outgoing, hidden_size):
        super().__init__(model=nn.GRUCell(sum(incoming.values()), hidden_size),name=name,interval=interval,hidden_shape=hidden_size,gates=outgoing)
        self.input_names = tuple(sorted(incoming))

    def forward(self, inputs):
        regular = self._communication_inputs(inputs)
        x = torch.cat([regular[source] for source in self.input_names], dim=-1)
        hidden = self.hidden
        if x.ndim == 2 and hidden.ndim == 1:
            hidden = hidden.expand(x.shape[0], -1)
        tokens, self.hidden = self._unfuse(self.model(x, hidden))
        return self._gate_outputs(tokens, inputs)
