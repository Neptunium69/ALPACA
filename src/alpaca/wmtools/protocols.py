from typing import Protocol,Any
from ..worldmodel import WorldModel, Object
from ..vehicleadapter import State
from .dataclasses import WMQuery

class WorldModelInterface(Protocol):
    model: WorldModel
    def read_wm(self,query:dict[str: Any])->list[Object]:
        """Reads world model."""
        raise NotImplementedError("This method should be implemented by subclasses.")
    def write_wm(self,arguments: Object)->None:
        """Writes world model."""
        raise NotImplementedError("This method should be implemented by subclasses.")
    def robot_state(self,arguments:dict[str:Any])->State:
        """Obtain robot state"""
        raise NotImplementedError("This method should be implemented by subclasses.")
    