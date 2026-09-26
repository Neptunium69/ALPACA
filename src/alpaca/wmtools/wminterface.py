from typing import Protocol,Any
from collections.abc import Collection
from ..worldmodel import WorldModel,Object,Position
from ..vehicleadapter import State

class WorldModelInterface(Protocol):
    model: WorldModel
    detection_classes:Collection[str]
    def read_wm(self,query:dict[str,Any])->list[Object]:
        """Reads world model."""
        raise NotImplementedError("This method should be implemented by subclasses.")
    def write_wm(self,arguments: Object)->None:
        """Writes world model."""
        raise NotImplementedError("This method should be implemented by subclasses.")
    def robot_state(self)->State:
        """Obtain robot state"""
        raise NotImplementedError("This method should be implemented by subclasses.")
    def bb2w(self,bounding_box:Any)->Position:
        """Convert an image bounding box into a world position."""
        raise NotImplementedError("This method should be implemented by subclasses.")
    def record(self)->None:
        """Record stuff"""
        raise NotImplementedError("This method should be implemented by subclasses.")
    def stop_record(self)->dict[str,Any]:
        """Stop recording stuff"""
        raise NotImplementedError("This method should be implemented by subclasses.")
