import math
from collections.abc import Callable
import torch
from torch import nn
import asyncio

class Memory(nn.Module):
    """Spatial SNN with recurrent edges formed within a fixed Euclidean radius.

    Positions are trainable in training mode and frozen in evaluation mode.
    Each step discovers neighbors from the current positions.
    """

    input_mask: torch.Tensor
    voltage: torch.Tensor
    spikes: torch.Tensor

    def __init__(self, high: float, low: float, dims: int, size: int, device, interval: float,shape: dict[str, int], name: str, *, connection_radius: float | None = None,) -> None:
        super().__init__()
        if not math.isfinite(low) or not math.isfinite(high) or high <= low:
            raise ValueError("Position bounds must be finite and high must exceed low.")
        if dims <= 0 or size <= 0:
            raise ValueError("Spatial dimensions and neuron count must be positive.")
        if connection_radius is None:
            connection_radius = 0.3 * (high - low)
        if not math.isfinite(connection_radius) or connection_radius <= 0:
            raise ValueError("Connection radius must be finite and positive.")
        if not math.isfinite(interval) or interval <= 0:
            raise ValueError("Interval must be finite and positive.")
        self.name=name
        self.positions = nn.Parameter(torch.rand(size, dims, device=device) * (high - low) + low)
        self.weights = nn.Parameter(0.2 * torch.randn(size, device=device) / math.sqrt(size))
        self.receive_weights = nn.Parameter(torch.ones(size, device=device))
        self.register_buffer("input_mask", torch.empty(size, dtype=torch.bool, device=device))
        self.register_buffer("voltage", tensor=torch.zeros(size, device=device), persistent=False)
        self.register_buffer("spikes", torch.zeros(size, device=device), persistent=False)

        self.tau = 0.020
        self.threshold = 1.0
        self.connection_radius = connection_radius
        self.interval=interval
        self.entry:dict[str,torch.Tensor]={}
        self.set_ports(shape)

    def set_ports(self, shape: dict[str, int]) -> None:
        """Allocate paired input and output slices while MoC is stopped."""
        if any(not isinstance(key, str) or type(width) is not int or width <= 0 for key, width in shape.items()):
            raise ValueError("shape must map cognition names to positive integer widths.")
        size = self.voltage.numel()
        port_size = sum(shape.values())
        if 2 * port_size > size:
            raise ValueError(f"Separate input and output groups require at least {2 * port_size} neurons.")

        self.shape = shape.copy()
        self.input_slices: dict[str, slice] = {}
        self.output_slices: dict[str, slice] = {}
        input_offset, output_offset = 0, size - port_size
        for key, width in self.shape.items():
            self.input_slices[key] = slice(input_offset, input_offset + width)
            self.output_slices[key] = slice(output_offset, output_offset + width)
            input_offset += width
            output_offset += width
        self.input_mask = torch.arange(size, device=self.voltage.device) < port_size
        self.y = {key: self.voltage.new_zeros(width) for key, width in self.shape.items()}

    def train(self, mode: bool = True):
        super().train(mode)
        self.positions.requires_grad_(mode)
        if not mode:
            self.positions.grad = None
        return self

    @torch.no_grad()
    def _neighbor_pairs(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Sweep the widest axis, then test candidate pairs in bounded batches."""
        extent = positions.amax(dim=0) - positions.amin(dim=0)
        axis = int(extent.argmax().item())
        coordinates, order = positions[:, axis].sort()
        rows = torch.arange(positions.size(0), device=positions.device)
        ends = torch.searchsorted(coordinates, coordinates + self.connection_radius, right=True)
        counts = ends - rows - 1
        offsets = counts.cumsum(dim=0)
        starts = offsets - counts

        sources: list[torch.Tensor] = []
        targets: list[torch.Tensor] = []
        batch_size = 65536
        candidate_count = int(offsets[-1].item())
        for start in range(0, candidate_count, batch_size):
            candidates = torch.arange(start, min(start + batch_size, candidate_count), device=positions.device)
            row = torch.searchsorted(offsets, candidates, right=True)
            col = row + 1 + candidates - starts[row]
            source, target = order[row], order[col]
            delta = (positions[source] - positions[target]) / self.connection_radius
            connected = delta.square().sum(dim=-1) <= 1
            sources.append(source[connected])
            targets.append(target[connected])
        if not sources:
            return order[:0], order[:0]
        return torch.cat(sources), torch.cat(targets)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs,) -> None:
        # Approximate legacy per-edge weights by averaging each neuron's outgoing edges.
        weights = state_dict.get(prefix + "weights")
        positions = state_dict.get(prefix + "positions")
        size = self.positions.size(0)
        if (weights is not None and weights.shape == (size, size)and positions is not None and positions.shape == self.positions.shape):
            source, target = self._neighbor_pairs(positions)
            first = torch.cat((source, target)).to(weights.device)
            second = torch.cat((target, source)).to(weights.device)
            outgoing = weights[second, first]
            sums = weights.new_zeros(size).index_add_(0, first, outgoing)
            counts = weights.new_zeros(size).index_add_(0, first, torch.ones_like(outgoing)).clamp_min(1)
            state_dict[prefix + "weights"] = sums / counts
            state_dict[prefix + "receive_weights"] = weights.new_ones(size)
            state_dict.pop(prefix + "mask", None)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def step(self, inputs: torch.Tensor, dt: float = 0.001) -> None:
        source, target = self._neighbor_pairs(self.positions)
        delta = (self.positions[source] - self.positions[target]) / self.connection_radius
        proximity = torch.exp(-0.5 * delta.square().sum(dim=-1))
        outgoing = self.spikes * self.weights
        recurrent = torch.zeros_like(self.voltage)
        recurrent.index_add_(0, target, proximity * outgoing[source])
        recurrent.index_add_(0, source, proximity * outgoing[target])
        recurrent = recurrent * self.receive_weights

        decay = math.exp(-dt / self.tau)
        voltage = decay * self.voltage + (1 - decay) * (inputs * self.input_mask + recurrent)

        excess = voltage - self.threshold
        hard_spikes = (excess >= 0).to(voltage.dtype)
        soft_spikes = torch.sigmoid(5 * excess)
        self.spikes = hard_spikes + (soft_spikes - soft_spikes.detach())
        self.voltage = voltage - self.threshold * self.spikes.detach()
        fired_voltage = voltage * self.spikes
        self.y = {key:fired_voltage[indices] for key, indices in self.output_slices.items()}

    def _fuse(self,X:dict[str,torch.Tensor])->torch.Tensor:
        fused = torch.zeros_like(self.voltage)
        for key, indices in self.input_slices.items():
            if key not in X:
                continue
            data = X[key]
            fused[indices] = data
        return fused

    async def run(self, stopflag: asyncio.Event, on_step: Callable[[str], None] | None = None):
        while not stopflag.is_set():
            X=self.entry.copy()
            await asyncio.to_thread(self.step, self._fuse(X))
            if on_step is not None:
                on_step(self.name)
            try:
                await asyncio.wait_for(stopflag.wait(), timeout=self.interval)
            except TimeoutError:
                pass
