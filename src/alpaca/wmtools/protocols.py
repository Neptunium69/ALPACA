from typing import Protocol,Any
from ..worldmodel import WorldModel, Object
from .dataclasses import WMQuery

class WorldModelInterface(Protocol):
    model: WorldModel
    def read_wm(query:dict[str: Any])->list[Object]:
        """Reads world model."""
        raise NotImplementedError("This method should be implemented by subclasses.")
    def write_wm(arguments: dict[str:Any])->dict[str:Any]:
        """Writes world model."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    