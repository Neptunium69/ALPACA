import torch
from .cognition import Cognition
from .memory import Memory
import asyncio

class MoC():
    def __init__(self,cogs:list[Cognition | Memory],tokens:int,embedding_dims:int,n_batches:int,fc:bool=False) -> None:
        self._models:dict[str,Cognition | Memory]={cog.name: cog for cog in cogs}
        if len(self._models) != len(cogs):
            raise ValueError("Cognition names must be unique.")
        self.connections:dict[str,dict[str,int]]={}
        self._stopflag=asyncio.Event()
        self.initflag=False
        self._running=False
        self.fc = fc
        self._fc_token_shape = (tokens, embedding_dims)
        if self.fc:
            self._fully_connect()

    def _fully_connect(self) -> None:
        """Use one gate per ordered pair with a cognition at either end."""
        tokens, dims = self._fc_token_shape
        if tokens <= 0 or dims <= 0:
            raise ValueError("Fully connected gates require positive tokens and embedding_dims.")
        width = tokens * dims
        cognition_names = sorted(name for name, cog in self._models.items() if isinstance(cog, Cognition))
        memory_ports = dict.fromkeys(cognition_names, width)
        required_neurons = 2 * len(cognition_names) * width
        for cog in self._models.values():
            if isinstance(cog, Memory) and cog.voltage.numel() < required_neurons:
                raise ValueError(f"Fully connected memory {cog.name!r} requires at least {required_neurons} neurons.")

        for name, cog in self._models.items():
            if isinstance(cog, Memory):
                if cog.shape != memory_ports:
                    cog.set_ports(memory_ports)
                continue
            destinations = set(self._models) - {name}
            cog.remove_gates([target for target in cog.shape if target not in destinations or cog.token_shape(target) != self._fc_token_shape])
            cog.add_gates({target: self._fc_token_shape for target in sorted(destinations) if target not in cog.shape})

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
                if self.fc:
                    self._fully_connect()
                connections:dict[str,dict[str,int]]={name:{} for name in sorted(self._models)}
                incoming:dict[str,dict[str,int]]={name:{} for name in self._models}
                for _,cog in self._models.items():
                    if cog.name.startswith("HIDDEN_"):
                        raise ValueError("Cognition names cannot start with HIDDEN_.")
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
                    entries[name] = {}
                    for source, width in sorted(layout.items()):
                        sender = self._models[source]
                        input_shape = (width,)
                        if isinstance(cog, Cognition) and cog.inputs_tokens:
                            if isinstance(sender, Cognition):
                                input_shape = sender.token_shape(name)
                            elif source in cog.shape:
                                input_shape = cog.token_shape(source)
                            else:
                                input_shape = (1, width)
                        entries[name][source] = reference.new_zeros(input_shape) if reference is not None else torch.zeros(input_shape)
                for name,cog in self._models.items():
                    cog.entry = entries[name]
                self.connections = connections
                self.initflag=True

            for name in self._models:
                self._refresh_entries(name)
            self._stopflag.clear()
            async with asyncio.TaskGroup() as group:
                for name, cog in self._models.items():
                    group.create_task(cog.run(self._stopflag, on_step=self._refresh_entries), name=name)
        finally:
            self._running=False

    def _refresh_entries(self, source: str) -> None:
        """Publish one component's gate outputs and hidden state on the event loop."""
        sender = self._models[source]
        outputs = sender.y
        updates: dict[str, torch.Tensor] = {}
        if outputs is not None:
            for target, width in self.connections[source].items():
                if target not in outputs:
                    continue
                value = outputs[target]
                receiver = self._models[target]
                output_shape = sender.token_shape(target) if isinstance(sender, Cognition) else (width,)
                if value.shape[-len(output_shape):] != output_shape:
                    raise ValueError(f"Invalid output shape for {source!r} -> {target!r}.")
                reference = receiver.entry[source]
                if isinstance(receiver, Cognition) and receiver.inputs_tokens:
                    if isinstance(sender, Memory):
                        value = value.unflatten(-1, reference.shape[-2:])
                elif isinstance(sender, Cognition):
                    value = value.flatten(start_dim=-2)
                updates[target] = value.to(reference)
        for target, value in updates.items():
            self._models[target].entry[source] = value

        hidden_key = "HIDDEN_" + source
        hidden = sender.hidden if isinstance(sender, Cognition) else sender.voltage
        for cog in self._models.values():
            if isinstance(cog, Cognition):
                cog.entry[hidden_key] = hidden

    async def stop(self):
        self._stopflag.set()
        self.initflag=False
