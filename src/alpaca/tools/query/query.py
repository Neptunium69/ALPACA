from dataclasses import dataclass
from typing import Any, Protocol
from ..toolshared import Tool,ToolArgument,ToolList,ToolSpecification,Command,Return
from importlib.resources import files
from ...wmtools.protocols import WorldModelInterface

class Query(Tool):
    def __init__(self,interface:WorldModelInterface):
        self.interface=interface
    @property
    def specification(self)->ToolSpecification:
        description=files(__package__).joinpath("specification.md").read_text(encoding="utf-8")
        return ToolSpecification(name="Query", description=description,arguments=[ToolArgument(name="object_id",required=False,type="str"), ToolArgument(name="class_name",required=False,type="str")])

    def run(self, command: Command) -> Return:
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
        else:
            objects = self.interface.read_wm({"object_id":object_id, "class_name": class_name})

        if len(objects) == 0:
            return Return(
                id=command.id,
                command_type=command.command_type,
                status="Success",
                results={
                    "Objects Found": [],
                    "Notes": "No objects found.",
                },
            )
        return Return(
            id=command.id,
            command_type=command.command_type,
            status="Success",
            results={"Objects Found": objects},
        )
