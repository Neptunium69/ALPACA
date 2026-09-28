import warnings
from ..dataclasses import Function

class VARegistry():
    def __init__(self,functions:list[Function]):
        self.tools=functions if functions is not None else []
    def add(self,fcn:Function)->None:
        if fcn.name not in [fc.name for fc in self.tools]:
            self.tools.append(fcn)
        else:
            warnings.warn(f"Tool {fcn.name} is already present, no changes were made.")
    def remove(self, name:str)->None:
        if name not in [fc.name for fc in self.tools]:
            warnings.warn(f"Tool {name} is not in tool registry, no changes were made.")
        else:
            for fcn in self.tools:
                if fcn.name==name:
                    self.tools.remove(fcn)
                    break