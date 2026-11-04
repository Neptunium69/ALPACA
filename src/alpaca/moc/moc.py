import torch
from torch import nn
from .cognition import Cognition
import asyncio

class MoC():
    def __init__(self,cogs:list[Cognition]) -> None:
        self._models:dict[str,Cognition]={cog.name: cog for cog in cogs}
        self._stopflag=asyncio.Event()

    def add_cogs(self,cogs:list[Cognition]) -> None:
        for cog in cogs:
            self._models[cog.name]=cog
    def rm_cogs(self, cogs:list[str]):
        for cog in cogs:
            self._models.pop(cog)
    async def run(self):
        async with asyncio.TaskGroup() as group:
            for name, cog in self._models.items():
                group.create_task(cog.run(self._stopflag), name=name)
    async def stop(self):
        self._stopflag.set()