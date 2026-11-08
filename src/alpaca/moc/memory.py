import math
import torch
from torch import nn
import asyncio

class Memory(nn.Module):
    """Spatial SNN with fixed, disjoint input and output groups per cognition."""

    mask: torch.Tensor
    input_mask: torch.Tensor
    voltage: torch.Tensor
    spikes: torch.Tensor

    def __init__(self,high,low,dims,size,device,interval,shape:dict[str,int],name:str) -> None:
        super().__init__()
        if not math.isfinite(low) or not math.isfinite(high) or high <= low:
            raise ValueError("Position bounds must be finite and high must exceed low.")
        if dims <= 0 or size <= 0:
            raise ValueError("Spatial dimensions and neuron count must be positive.")
        if not math.isfinite(interval) or interval <= 0:
            raise ValueError("Interval must be finite and positive.")
        if any(not isinstance(key, str) or type(width) is not int or width <= 0 for key, width in shape.items()):
            raise ValueError("shape must map cognition names to positive integer widths.")
        port_size = sum(shape.values())
        if 2 * port_size > size:
            raise ValueError(f"Separate input and output groups require at least {2 * port_size} neurons.")

        self.name=name
        self.shape:dict[str,int]=shape.copy()
        self.input_slices:dict[str,slice]={}
        self.output_slices:dict[str,slice]={}
        input_offset, output_offset = 0, size - port_size
        for key, width in self.shape.items():
            self.input_slices[key] = slice(input_offset, input_offset + width)
            self.output_slices[key] = slice(output_offset, output_offset + width)
            input_offset += width
            output_offset += width

        self.positions = nn.Parameter(torch.rand(size, dims, device=device) * (high - low) + low)
        self.weights = nn.Parameter(0.2 * torch.randn(size, size, device=device) / math.sqrt(size))
        self.register_buffer("mask", 1 - torch.eye(size, device=device))
        self.register_buffer("input_mask", torch.arange(size, device=device) < port_size)
        self.register_buffer("voltage", tensor=torch.zeros(size, device=device), persistent=False)
        self.register_buffer("spikes", torch.zeros(size, device=device), persistent=False)

        self.tau = 0.020
        self.threshold = 1.0
        self.sigma = 0.3 * (high - low)
        self.interval=interval
        self.entry:dict[str,torch.Tensor]={}
        self.y:dict[str,torch.Tensor]={key:self.voltage.new_zeros(width) for key, width in self.shape.items()}

    def step(self, inputs: torch.Tensor, dt: float = 0.001) -> None:
        """Publish firing neurons' pre-reset voltages in ``y``; silent outputs are zero.

        ``inputs`` has one slot per neuron, but external stimulation is applied
        only to the designated input groups. ``dt`` is simulated time in seconds.
        """
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive.")
        if inputs.shape != self.voltage.shape:
            raise ValueError("inputs must contain one stimulation value per neuron.")
        if inputs.device != self.voltage.device or inputs.dtype != self.voltage.dtype:
            raise ValueError("inputs must match the network's device and dtype.")

        delta = self.positions[:, None, :] - self.positions[None, :, :]
        distance_sq = delta.square().sum(dim=-1)
        proximity = torch.exp(-distance_sq / (2 * self.sigma**2))
        connections = self.weights * proximity * self.mask

        recurrent = connections @ self.spikes
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
        """Map named inputs to fixed neuron slots; missing inputs supply zero drive."""
        unknown = X.keys() - self.shape.keys()
        if unknown:
            raise ValueError(f"Unknown input cognitions: {sorted(unknown)}")
        fused = torch.zeros_like(self.voltage)
        for key, indices in self.input_slices.items():
            if key not in X:
                continue
            data = X[key]
            if data.shape != (self.shape[key],):
                raise ValueError(f"Input {key!r} must have shape ({self.shape[key]},).")
            if data.device != self.voltage.device or data.dtype != self.voltage.dtype:
                raise ValueError(f"Input {key!r} must match the network's device and dtype.")
            fused[indices] = data
        return fused

    async def run(self, stopflag:asyncio.Event):
        while not stopflag.is_set():
            X=self.entry.copy()
            await asyncio.to_thread(self.step, self._fuse(X))
            try:
                await asyncio.wait_for(stopflag.wait(), timeout=self.interval)
            except TimeoutError:
                pass
