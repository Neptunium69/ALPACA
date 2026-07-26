from __future__ import annotations

from . import worldmodel as wm
from . import vehicleadapter as va
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
    def __init__(self, model: wm.WorldModel, adapter: va.UGVAdapter) -> None:
        self.model = model
        self.adapter = adapter

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

        if isinstance(object_data, wm.Object):
            world_object = object_data
        elif isinstance(object_data, dict):
            try:
                world_object = wm.Object(**object_data)
            except TypeError as error:
                return Return(
                    id=command.id,
                    command_type=command.command_type,
                    status="Failed",
                    results={"Error": f"Invalid object format: {error}"},
                )
        else:
            return Return(
                id=command.id,
                command_type=command.command_type,
                status="Failed",
                results={
                    "Error": "Object must be an Object instance or a dictionary."
                },
            )

        confidence_scores_are_valid = (
            isinstance(world_object.confidence_scores, dict)
            and all(
                isinstance(name, str)
                and isinstance(score, (int, float))
                and not isinstance(score, bool)
                for name, score in world_object.confidence_scores.items()
            )
        )

        if (
            not isinstance(world_object.id, str)
            or not isinstance(world_object.class_name, str)
            or not isinstance(world_object.attributes, dict)
            or not all(
                isinstance(name, str) for name in world_object.attributes
            )
            or not confidence_scores_are_valid
        ):
            return Return(
                id=command.id,
                command_type=command.command_type,
                status="Failed",
                results={
                    "Error": "Object does not match the world-model Object format."
                },
            )

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
    def _tool_scan(self, command: Command) -> Return:
        try:
            self.adapter.scan()
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
            results={"Scan": "Completed"},
        )
    


