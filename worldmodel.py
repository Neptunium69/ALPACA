from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from copy import deepcopy
from threading import RLock
import rospy
from .yolo import YOLOAdapter
from std_msgs.msg import String
from math import e
import json


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
    position: Position
    attributes: dict[str, Any] = field(default_factory=dict)
    confidence_scores: dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        defaults = {
            "existence_rawscore": -2.0,
            "existence_score": 0.1192,
            "positioning_rawscore": -2.0,
            "positioning_score": 0.1192,
        }
        for key, value in defaults.items():
            self.confidence_scores.setdefault(key, value)

@dataclass
class Detection:
    class_name: str
    confidence: float
    position: list[float]
    distance: float | None

@dataclass
class Model:
    objects: list[Object] = field(default_factory=list)

class WorldModel:
    def __init__(self, observations: Model, detection_classes):
        self.observations = observations
        self.detection_classes = tuple(detection_classes)
        self.record=[]
        self._lock = RLock()

    def initwm(self,detection_topic:str="yolo_detections"):
        # Automatic world model updater
        rospy.init_node("world_model_node", anonymous=True)
        rospy.Subscriber(detection_topic,String,callback=self._update_callback,queue_size=1)
        rospy.spin()

    @staticmethod
    def sigmoid(x):
        return 1/(1+e**-x)

    def _update_callback(self, detections):
        try:
            detect_dict=json.loads(detections.data)
            detection_list=[]
            for i in range(len(detect_dict["class_names"])):
                detection_list.append(Detection(detect_dict["class_names"][i],detect_dict["confidences"][i],detect_dict["position"][i],detect_dict["detection_distance"][i]))
        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as error:
            rospy.logerr("Invalid detection message: %s", error)
            return

        with self._lock:
            try:
                working_objects = deepcopy(self.observations.objects)
                self._apply_detections(working_objects, detection_list)
            except Exception as error:
                rospy.logerr("World-model update failed: %s", error)
                return

            self.observations.objects[:] = working_objects

    def _apply_detections(self, objects: list[Object], detection_list: list[Detection]) -> None:
        for detection in detection_list:
            if(detection.position!=[None,None,None]):
                if len(objects)>0:
                    flag=0
                    for object in objects:
                        if((object.class_name==detection.class_name) and (object.position.x-0.1)<detection.position[0]<(object.position.x+0.1) and ((object.position.y-0.1)<detection.position[1]<(object.position.y+0.1)) and ((object.position.z-0.1)<detection.position[2]<(object.position.z+0.1))):
                            rawscore = object.confidence_scores.get("existence_rawscore")
                            if rawscore is None:
                                rawscore = -2.0
                            rawscore += 0.1*detection.confidence
                            object.confidence_scores["existence_rawscore"] = rawscore
                            flag=1
                            break
                    if(flag==1):
                        object.confidence_scores["existence_score"]=self.sigmoid(object.confidence_scores["existence_rawscore"])
                    if(detection.distance!=None and detection.position!=[None,None,None] and flag==1):
                        distance_scaled=detection.distance/10
                        positioning_rawscore = object.confidence_scores.get("positioning_rawscore")
                        if positioning_rawscore is None:
                            positioning_rawscore = -2.0
                        positioning_increment = 0.1*((1-distance_scaled)**2+0.05)
                        positioning_rawscore += positioning_increment
                        object.confidence_scores["positioning_rawscore"] = positioning_rawscore
                        object.confidence_scores["positioning_score"]=self.sigmoid(positioning_rawscore)
                        object.position.x=(object.position.x*object.confidence_scores["positioning_score"])+((1-object.confidence_scores["positioning_score"])*detection.position[0])
                        object.position.y=(object.position.y*object.confidence_scores["positioning_score"])+((1-object.confidence_scores["positioning_score"])*detection.position[1])
                        object.position.z=(object.position.z*object.confidence_scores["positioning_score"])+((1-object.confidence_scores["positioning_score"])*detection.position[2])

                    else:
                        pass

                    if(flag==0):
                        objects.append(Object(id=f"{detection.class_name}_{self._get_latest(detection.class_name, objects)+1}", class_name=detection.class_name, position=Position(detection.position[0],detection.position[1],detection.position[2]),confidence_scores={"existence_rawscore":-2.0,"existence_score":self.sigmoid(-2.0)}))
                        objects[-1].confidence_scores["positioning_rawscore"]=-2.0
                        objects[-1].confidence_scores["positioning_score"]=self.sigmoid(objects[-1].confidence_scores["positioning_rawscore"])
                else:
                    objects.append(Object(id=f"{detection.class_name}_{self._get_latest(detection.class_name, objects)+1}", class_name=detection.class_name, position=Position(detection.position[0],detection.position[1],detection.position[2]),confidence_scores={"existence_rawscore":-2.0,"existence_score":self.sigmoid(-2.0)}))
                    objects[-1].confidence_scores["positioning_rawscore"]=-2.0
                    objects[-1].confidence_scores["positioning_score"]=self.sigmoid(objects[-1].confidence_scores["positioning_rawscore"])


    # Insert record and stop record functions

    def _get_latest(self,class_name:str, objects: list[Object] | None = None)->int:
        with self._lock:
            largest_number = 0
            source_objects = self.observations.objects if objects is None else objects

            for world_object in source_objects:
                if world_object.class_name == class_name:
                    number = int(world_object.id.rsplit("_", 1)[1])

                    if number > largest_number:
                        largest_number = number

            return largest_number

    def query_world(self,object_id: str | None = None,class_name: str | None = None,) -> list[Object]:
        """Return objects matching an ID, a class, or all objects."""
        with self._lock:
            if object_id is not None:
                result = [
                    world_object
                    for world_object in self.observations.objects
                    if world_object.id == object_id
                ]
            elif class_name is not None:
                result = [
                    world_object
                    for world_object in self.observations.objects
                    if world_object.class_name == class_name
                ]
            else:
                result = self.observations.objects

            return deepcopy(result)

    def _add_object(self, world_object: Object) -> None:
        with self._lock:
            if any(
                existing_object.id == world_object.id
                for existing_object in self.observations.objects
            ):
                raise ValueError(f"An object with ID {world_object.id!r} already exists.")

            self.observations.objects.append(world_object)

    def _remove_object(self, object_id: str) -> Object:
        with self._lock:
            for index, world_object in enumerate(self.observations.objects):
                if world_object.id == object_id:
                    return self.observations.objects.pop(index)

            raise KeyError(f"No object with ID {object_id!r} exists.")

    def write(self,world_object: Object) -> None:
        candidate = deepcopy(world_object)
        for detection_class in self.detection_classes:
            if f"{detection_class}_" not in candidate.class_name:
                candidate.attributes["notes"]="Artificially written by VLM, not added by automatic updater"
                candidate.confidence_scores["existence_rawscore"]=0.0
                candidate.confidence_scores["existence_score"]=float(self.sigmoid(candidate.confidence_scores["existence_rawscore"]))
            else:
                raise ValueError(f"Class name {candidate.class_name} is invalid because it contains the reserved phrase '{detection_class}_'")

        with self._lock:
            self._add_object(candidate)
