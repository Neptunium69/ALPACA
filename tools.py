from __future__ import annotations

from . import worldmodel as wm
from . import vehicleadapter as va
from . import risk as rcm
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from .yolo import YOLOAdapter
from dataclasses import dataclass,fields
from typing import Any
import rospy

command_list=["query","robot_state","write","observe","scan","navigate","inspect","ask_user","finish"]

@dataclass(frozen=True)
class Command:
    id: str
    command_type: str
    arguments: dict[str, Any]

@dataclass(frozen=True)
class Return:
    id: str
    command_type: str
    status: str
    results: dict[str, Any]

class ToolExecutor:
    def __init__(self, model: wm.WorldModel, adapter: va.UGVAdapter, riskcalibration:rcm.RiskCal, detector:YOLOAdapter | None=None) -> None:
        self.model = model
        self.adapter = adapter
        self.risk=riskcalibration
        self.detector=detector
        self.bridge=CvBridge()

    def execute(self, commands: list[Command]) -> list[Return]:
        results=[]
        for command in commands:
            if command.command_type not in command_list:
                results.append(Return(id=command.id, command_type=command.command_type, status="Failed", results={"Error:":f"{command.command_type} is not a valid command."}))
            else:
                results.append(self._command_reader(command))
        return results
    def _command_reader(self, command: Command) -> Return:
        tool_function = getattr(self, f"_tool_{command.command_type}")
        return tool_function(command)

    def _tool_query(self, command: Command) -> Return:
        object_id = command.arguments.get("object_id")
        class_name = command.arguments.get("class_name")

        if object_id is not None and class_name is not None:
            return Return(
                id=command.id,
                command_type=command.command_type,
                status="Failed",
                results={
                    "Objects Found": [],
                    "Error": "Query by either object_id or class_name, not both.",
                },
            )

        objects = self.model.query_world(
            object_id=object_id,
            class_name=class_name,
        )

        if len(objects) == 0:
            return Return(
                id=command.id,
                command_type=command.command_type,
                status="Failed",
                results={
                    "Objects Found": [],
                    "Error": "No objects found.",
                },
            )

        return Return(
            id=command.id,
            command_type=command.command_type,
            status="Success",
            results={"Objects Found": objects},
        )
    
    def _tool_robot_state(self, command: Command) -> Return:
        state=self.adapter.get_state()
        rv=Return(id=command.id, command_type=command.command_type, status="Success", results={})
        for field in fields(state):
            rv.results[field.name] = getattr(state, field.name)
        return rv

    def _tool_write(self, command: Command) -> Return:
        object_data = command.arguments.get("object")

        if not isinstance(object_data,dict):
            return Return(
                id=command.id,
                command_type=command.command_type,
                status="Failed",
                results={"Error":"Object must be a dictionary containing id, class_name, and bounding_box."},
            )

        missing_arguments=[name for name in ("id","class_name","bounding_box") if name not in object_data]
        if missing_arguments:
            return Return(id=command.id, command_type=command.command_type, status="Failed", results={"Error":f"Missing required object arguments: {', '.join(missing_arguments)}"})

        attributes=object_data.get("attributes",{})
        if not isinstance(object_data["id"],str) or not object_data["id"] or not isinstance(object_data["class_name"],str) or not object_data["class_name"]:
            return Return(id=command.id, command_type=command.command_type, status="Failed", results={"Error":"id and class_name must be non-empty strings."})
        detection_classes=self.detector.get_classes() if self.detector is not None else self.model.detection_classes
        if object_data["class_name"] in detection_classes:
            return Return(id=command.id, command_type=command.command_type, status="Failed", results={"Error":f"Class name {object_data['class_name']!r} is reserved and cannot be used."})
        if not isinstance(attributes,dict) or not all(isinstance(name,str) for name in attributes):
            return Return(id=command.id, command_type=command.command_type, status="Failed", results={"Error":"attributes must be a dictionary with string keys."})
        forbidden_attributes={"Notes","VLMFlag"}.intersection(attributes)
        if forbidden_attributes:
            return Return(id=command.id, command_type=command.command_type, status="Failed", results={"Error":f"Forbidden attribute keys: {', '.join(sorted(forbidden_attributes))}"})

        try:
            position=self.model.bounding_box_to_world(object_data["bounding_box"])
            world_object=wm.Object(id=object_data["id"],class_name=object_data["class_name"],size=wm.Size(),position=position,attributes=dict(attributes))
        except (TypeError,ValueError) as error:
            return Return(id=command.id, command_type=command.command_type, status="Failed", results={"Error":f"Could not calculate object position: {error}"})

        try:
            self.model.write(world_object)
        except ValueError as error:
            return Return(
                id=command.id,
                command_type=command.command_type,
                status="Failed",
                results={"Error": str(error)},
            )

        return Return(
            id=command.id,
            command_type=command.command_type,
            status="Success",
            results={"Object Written": world_object},
        )

    def _tool_observe(self, command:Command) -> Return:
        try:
            message=rospy.wait_for_message("camera_color", Image, timeout=5.0)
            image=self.bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
            height,width=image.shape[:2]
            timestamp=message.header.stamp.to_nsec()
            image_path=f"/tmp/camera_image_{timestamp}.png"
            if not cv2.imwrite(image_path,image):
                raise IOError(f"Could not write {image_path}")
            return Return(id=command.id,command_type=command.command_type,status="Success",results={
                "image_path":image_path,
                "width":width,
                "height":height,
                "timestamp_ns":timestamp,
                "frame_id":message.header.frame_id,
            })
        except Exception as error:
            return Return(id=command.id,command_type=command.command_type,status="Failed",results={"Error":str(error)})
        
    def _tool_scan(self, command: Command) -> Return:
        try:
            self.model.record()
            self.risk.inspect(command)
            self.adapter.scan()
            changes=self.model.stop_record()
        except rospy.ROSInterruptException as error:
            return Return(
                id=command.id,
                command_type=command.command_type,
                status="Interrupted",
                results={"Reason": str(error)},
            )
        except Exception as error:
            return Return(
                id=command.id,
                command_type=command.command_type,
                status="Failed",
                results={"Error": str(error)},
            )

        return Return(
            id=command.id,
            command_type=command.command_type,
            status="Success",
            results={"Scan": "Completed",
                     "Major Changes": changes,
        })
    
