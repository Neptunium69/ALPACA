from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import rospy
from std_msgs.msg import String
import json


@dataclass
class Object:
    id: str
    class_name: str

@dataclass
class Position:
    x: float
    y: float
    z: float
@dataclass
class Orientation:
    pitch: float
    yaw: float
    roll: float

@dataclass
class Object:
    id: str
    class_name: str
    attributes: dict[str, Any] = field(default_factory=dict)
    confidence_scores: dict[str, float] = field(default_factory=dict)

@dataclass
class Model:
    objects: list[Object] = field(default_factory=list)

class WorldModel:
    def __init__(self, observations: Model):
        self.observations = observations
        self.record=[]

    def initwm(self,detection_topic:str="yolo_detections"):
        # Automatic world model updater
        rospy.init_node("world_model_node", anonymous=True)
        rospy.Subscriber(detection_topic,String,callback=self._update_callback,queue_size=1)
        rospy.spin()

    def _update_callback(detections):
        detect_dict=json.loads(detections)
        
        # placeholder for a wm update system

    # Insert record and stop record functions

    def query_world(self,object_id: str | None = None,class_name: str | None = None,) -> list[Object]:
        """Return objects matching an ID, a class, or all objects."""
        if object_id is not None:
            return [
                world_object
                for world_object in self.observations.objects
                if world_object.id == object_id
            ]

        if class_name is not None:
            return [
                world_object
                for world_object in self.observations.objects
                if world_object.class_name == class_name
            ]

        return list(self.observations.objects)

    def _add_object(self, world_object: Object) -> None:
        if any(
            existing_object.id == world_object.id
            for existing_object in self.observations.objects
        ):
            raise ValueError(f"An object with ID {world_object.id!r} already exists.")

        self.observations.objects.append(world_object)

    def _remove_object(self, object_id: str) -> Object:
        for index, world_object in enumerate(self.observations.objects):
            if world_object.id == object_id:
                return self.observations.objects.pop(index)

        raise KeyError(f"No object with ID {object_id!r} exists.")

    def write(self,world_object: Object) -> None:
        world_object.attributes["Notes"]="Artificially written by VLM, not added by automatic updater"
        self._add_object(world_object)
