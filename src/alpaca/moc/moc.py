import torch
from .cognition import Cognition
from .memory import Memory
import asyncio

class MoC():
    def __init__(self,cogs:list[Cognition | Memory]) -> None:
        self._models:dict[str,Cognition | Memory]={cog.name: cog for cog in cogs}
        if len(self._models) != len(cogs):
            raise ValueError("Cognition names must be unique.")
        self.connections:dict[str,dict[str,int]]={}
        self._stopflag=asyncio.Event()
        self.initflag=False
        self._running=False

    def add_cogs(self,cogs:list[Cognition | Memory]) -> None:
        if self._running:
            raise RuntimeError("Stop MoC before changing its cognitions.")
        for cog in cogs:
            self._models[cog.name]=cog
        self.initflag=False
    def rm_cogs(self, cogs:list[str]):
        if self._running:
            raise RuntimeError("Stop MoC before changing its cognitions.")
        for cog in cogs:
            self._models.pop(cog)
        self.initflag=False
    async def run(self):
        if self._running:
            raise RuntimeError("MoC is already running.")
        self._running=True
        try:
            if self.initflag is False:
                connections:dict[str,dict[str,int]]={name:{} for name in sorted(self._models)}
                incoming:dict[str,dict[str,int]]={name:{} for name in self._models}
                for _,cog in self._models.items():
                    # Each component declares outgoing destinations and widths.
                    if any(not isinstance(target, str) or type(width) is not int or width <= 0 for target, width in cog.shape.items()):
                        raise ValueError(f"Invalid output schema for {cog.name!r}.")
                    for target, width in sorted(cog.shape.items()):
                        if target not in self._models:
                            raise ValueError(f"{cog.name!r} targets unknown cognition {target!r}.")
                        receiver = self._models[target]
                        if isinstance(receiver, Memory) and receiver.shape.get(cog.name) != width:
                            raise ValueError(f"{cog.name!r} -> {target!r} must match Memory's declared input width.")
                        connections[cog.name][target] = width
                        incoming[target][cog.name] = width

                # Allocate each input separately, on its receiver's device/dtype.
                entries:dict[str,dict[str,torch.Tensor]]={}
                for name,cog in self._models.items():
                    layout = cog.shape if isinstance(cog, Memory) else incoming[name]
                    reference = next(cog.parameters(), None)
                    if reference is None:
                        reference = next((buffer for buffer in cog.buffers() if buffer.is_floating_point()), None)
                    entries[name] = {
                        source:reference.new_zeros(width) if reference is not None else torch.zeros(width)
                        for source, width in sorted(layout.items())
                    }
                for name,cog in self._models.items():
                    cog.entry = entries[name]
                self.connections = connections
                self.initflag=True

            self._stopflag.clear()
            async with asyncio.TaskGroup() as group:
                for name, cog in self._models.items():
                    group.create_task(cog.run(self._stopflag), name=name)
        finally:
            self._running=False
    async def stop(self):
        self._stopflag.set()
        self.initflag=False
