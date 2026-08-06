from __future__ import annotations

from . import worldmodel as wm
from . import vehicleadapter as va
from . import risk as rcm
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from .yolo import YOLOAdapter
from dataclasses import dataclass,fields
import math
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

    def get_command_requirements(self) -> list[Command]:
        """Return a list of commands, each with their required formats. __relations__ contains the relations between two arguments."""
        return [
            Command(id="<id>",command_type="query",arguments={
                "object_id": {"type":"str","required":False},
                "class_name": {"type":"str","required":False},
                "__relations__": [{"type":"exactly_one_of","arguments":["object_id","class_name"]}],
            }),
            Command(id="<id>",command_type="robot_state",arguments={}),
            Command(id="<id>",command_type="write",arguments={
                "object": {
                    "type":"dict",
                    "required":True,
                    "properties": {
                        "id": {"type":"str","required":True},
                        "class_name": {"type":"str","required":True},
                        "bounding_box": {"type":"list[float]","required":True},
                        "attributes": {"type":"dict[str, Any]","required":False},
                    },
                },
            }),
            Command(id="<id>",command_type="observe",arguments={}),
            Command(id="<id>",command_type="scan",arguments={}),
            Command(id="<id>",command_type="navigate",arguments={
                "destination": {
                    "type":"dict",
                    "required":True,
                    "properties": {
                        "x": {"type":"float","required":True},
                        "y": {"type":"float","required":True},
                        "z": {"type":"float","required":True},
                    },
                },
                "orientation": {
                    "type":"dict",
                    "required":True,
                    "properties": {
                        "pitch": {"type":"float","required":True},
                        "yaw": {"type":"float","required":True},
                        "roll": {"type":"float","required":True},
                    },
                },
                "frame_id": {"type":"str","required":False},
            }),
            Command(id="<id>",command_type="inspect",arguments={
                "target":{
                    "type":"string",
                    "required": True
                }
            }),
            Command(id="<id>",command_type="ask_user",arguments={
                "text_message":{
                    "type": "string",
                    "required": False
                },
                "image_message":{
                    "type": "string",
                    "required": False,
                    "notes": "Send the file path of the image."
                },
                "__relations__": [{"type":"at_least_one_of", "arguments":["text_message","image_message"]}]
            }),
            Command(id="<id>",command_type="finish",arguments={
                "completion_text_message":{
                    "type": "string",
                    "required": False
                },
                "completion_image_message":{
                    "type": "string",
                    "required": False,
                    "notes": "Send the file path of the image."
                },
                "__relations__": [{"type":"at_least_one_of", "arguments":["completion_text_message","completion_image_message"]}]
            }),
        ]

    def execute(self, commands: list[Command]) -> list[Return]:
        results=[]
        for command in commands:
            if command.command_type not in command_list:
                results.append(Return(id=command.id, command_type=command.command_type, status="Failed", results={"Error:":f"{command.command_type} is not a valid command."}))
            else:
                results.append(self._command_reader(command))
        return results
    def _command_reader(self, command: Command) -> Return:
        if not isinstance(command.id,str) or not command.id.strip():
            return Return(id=command.id,command_type=command.command_type,status="Failed",results={"Error":"Command id must be a non-empty string."})
        requirement=next((item for item in self.get_command_requirements() if item.command_type==command.command_type),None)
        if requirement is None:
            return Return(id=command.id,command_type=command.command_type,status="Failed",results={"Error":f"No requirements are defined for the {command.command_type} command."})
        validation_error=self._validate_arguments(command.arguments,requirement.arguments)
        if validation_error is not None:
            return Return(id=command.id,command_type=command.command_type,status="Failed",results={"Error":validation_error})
        tool_function = getattr(self,f"_tool_{command.command_type}",None)
        if tool_function is None:
            return Return(id=command.id,command_type=command.command_type,status="Failed",results={"Error":f"The {command.command_type} command is not implemented."})
        return tool_function(command)

    def _validate_arguments(self, arguments: Any, requirements: dict[str, Any], path: str="arguments") -> str | None:
        if not isinstance(arguments,dict):
            return f"{path} must be a dictionary."
        argument_requirements={name:specification for name,specification in requirements.items() if not name.startswith("__")}
        unexpected_arguments=sorted(set(arguments)-set(argument_requirements))
        if unexpected_arguments:
            return f"Unexpected {path}: {', '.join(unexpected_arguments)}."
        missing_arguments=[name for name,specification in argument_requirements.items() if specification.get("required") is True and (name not in arguments or arguments[name] is None)]
        if missing_arguments:
            return f"Missing required {path}: {', '.join(missing_arguments)}."
        for name,value in arguments.items():
            specification=argument_requirements[name]
            expected_type=specification.get("type")
            if not self._argument_matches_type(value,expected_type):
                return f"{path}.{name} must be of type {expected_type}."
            properties=specification.get("properties")
            if properties is not None:
                nested_error=self._validate_arguments(value,properties,f"{path}.{name}")
                if nested_error is not None:
                    return nested_error
        for relation in requirements.get("__relations__",[]):
            relation_type=relation.get("type")
            related_arguments=relation.get("arguments",[])
            present_arguments=[name for name in related_arguments if name in arguments and arguments[name] is not None]
            if relation_type=="exactly_one_of" and len(present_arguments)!=1:
                return f"Exactly one of {', '.join(related_arguments)} must be provided."
            if relation_type=="at_least_one_of" and not present_arguments:
                return f"At least one of {', '.join(related_arguments)} must be provided."
            if relation_type not in {"exactly_one_of","at_least_one_of"}:
                return f"Unsupported argument relation: {relation_type}."
        return None

    @staticmethod
    def _argument_matches_type(value: Any, expected_type: str | None) -> bool:
        if expected_type is None:
            return True
        if expected_type in {"str","string"}:
            return isinstance(value,str)
        if expected_type in {"float","number"}:
            return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)
        if expected_type in {"int","integer"}:
            return isinstance(value,int) and not isinstance(value,bool)
        if expected_type in {"bool","boolean"}:
            return isinstance(value,bool)
        if expected_type in {"dict","object"}:
            return isinstance(value,dict)
        if expected_type in {"list","array"}:
            return isinstance(value,list)
        if expected_type=="list[float]":
            return isinstance(value,list) and all(isinstance(item,(int,float)) and not isinstance(item,bool) and math.isfinite(item) for item in value)
        if expected_type=="dict[str, Any]":
            return isinstance(value,dict) and all(isinstance(name,str) for name in value)
        if expected_type in {"Any","any"}:
            return True
        return False

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
            if message is None:
                raise RuntimeError("Camera returned no image message")
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
            risk_result=self.risk.checkcmd(command)
            if risk_result.get("Safe") is False:
                return Return(id=command.id,command_type=command.command_type,status="Failed",results={"Reason":risk_result.get("Reason","Command failed the safety check.")})
            self.model.record()
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

    def _tool_navigate(self, command: Command, redirects_left:int=3) -> Return:
        try:
            destination_data=command.arguments.get("destination")
            orientation_data=command.arguments.get("orientation")
            if not isinstance(destination_data,dict) or not isinstance(orientation_data,dict):
                return Return(id=command.id,command_type=command.command_type,status="Failed",results={"Error":"Destination and orientation must be dictionaries."})
            if set(destination_data)!={"x","y","z"} or set(orientation_data)!={"pitch","yaw","roll"}:
                return Return(id=command.id,command_type=command.command_type,status="Failed",results={"Error":"Destination must contain exactly x, y, and z; orientation must contain exactly pitch, yaw, and roll."})
            goal_values=[destination_data["x"],destination_data["y"],destination_data["z"],orientation_data["roll"],orientation_data["pitch"],orientation_data["yaw"]]
            if any(not isinstance(value,(int,float)) or isinstance(value,bool) or not math.isfinite(value) for value in goal_values):
                return Return(id=command.id,command_type=command.command_type,status="Failed",results={"Error":"Destination and orientation values must be finite numbers."})
            frame_id=command.arguments.get("frame_id",self.adapter.navigation_frame)
            if not isinstance(frame_id,str) or not frame_id.strip():
                return Return(id=command.id,command_type=command.command_type,status="Failed",results={"Error":"frame_id must be a non-empty string."})
            destination=va.Position(x=float(destination_data["x"]),y=float(destination_data["y"]),z=float(destination_data["z"]))
            orientation=va.Orientation(pitch=float(orientation_data["pitch"]),yaw=float(orientation_data["yaw"]),roll=float(orientation_data["roll"]))
            risk_result=self.risk.checkcmd(command,destination,orientation,frame_id)
            if risk_result.get("Safe") is not True:
                reason=risk_result.get("Reason","Command failed the safety check.")
                best_vantage=risk_result.get("Best Vantage Point")
                if risk_result.get("Needs Information") is not True or best_vantage is None:
                    return Return(id=command.id,command_type=command.command_type,status="Failed",results={"Reason":reason,"Inform User":risk_result.get("Inform User",False)})
                if redirects_left<=0:
                    return Return(id=command.id,command_type=command.command_type,status="Failed",results={"Reason":"Maximum number of vantage-point redirects reached.","Original Reason":reason})
                if not isinstance(best_vantage,dict):
                    return Return(id=command.id,command_type=command.command_type,status="Failed",results={"Error":"The risk model returned an invalid vantage-point record."})
                destination_new=best_vantage.get("Position")
                orientation_new=best_vantage.get("Orientation")
                if not isinstance(destination_new,va.Position) or not isinstance(orientation_new,va.Orientation):
                    return Return(id=command.id,command_type=command.command_type,status="Failed",results={"Error":"The risk model returned invalid vantage-point coordinates."})
                cmd_new=Command(f"{command.id}_vantage","navigate",{
                    "destination":{"x":destination_new.x,"y":destination_new.y,"z":destination_new.z},
                    "orientation":{"pitch":orientation_new.pitch,"yaw":orientation_new.yaw,"roll":orientation_new.roll},
                    "frame_id":frame_id,
                })
                result=self._tool_navigate(cmd_new,redirects_left-1)
                if result.status!="Success":
                    return Return(id=command.id,command_type=command.command_type,status="Failed",results={"Reason":"Unable to navigate to a safe vantage point.","Original Reason":reason,"Vantage Result":result.results})
                return self._tool_navigate(command,redirects_left-1)
            self.adapter.navigate(destination,orientation,frame_id)
        except Exception as error:
            return Return(id=command.id,command_type=command.command_type,status="Failed",results={"Error":str(error)})
        return Return(id=command.id,command_type=command.command_type,status="Success",results={"Navigation":"Completed"})

    def _tool_inspect(self, command:Command):
        try:
            stet=self.adapter.get_state()
            target=self.model.hiddenpoint(object_id=command.arguments["target"],vehicle_position=stet.position, state=stet, distance_behind=0.7)
            targetpose=stet.orientation
            cmd=Command(id=f"{command.id}_navigate",command_type="navigate", arguments={"destination":{"x":target.x,"y":target.y,"z":target.z},
                "orientation":{"pitch":targetpose.pitch,"yaw":targetpose.yaw,"roll":targetpose.roll},
                "frame_id":self.adapter.navigation_frame,})
            
            result=self._tool_navigate(cmd)
            if result.status!="Success":
                return Return(id=command.id, command_type=command.command_type, status="Failed", results={"Notes": "Navigation related error","Error": result.results})
            cmd=Command(id=f"{command.id}_scan", command_type="scan", arguments={})
            result=self._tool_scan(cmd)
            if result.status!="Success":
                return Return(id=command.id, command_type=command.command_type, status="Failed", results={"Notes": "Scan related error.", "Error":result.results})
            return Return(id=command.id, command_type=command.command_type, status="Success", results={"Inspection":"Completed","Major Changes":result.results.get("Major Changes")})
        except(KeyError,ValueError,TypeError) as error:
            return Return(id=command.id, command_type=command.command_type, status="Failed", results={"Error": str(error)})

    

    
