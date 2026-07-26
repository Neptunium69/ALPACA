from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


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
        self._add_object(world_object)
