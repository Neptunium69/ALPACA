from dataclasses import dataclass,fields
from typing import Any, Protocol
from ..toolshared import Tool,ToolArgument,ToolList,ToolSpecification,Command,Return
from importlib.resources import files
from ...wmtools.wminterface import WorldModelInterface

class RobotState(Tool):
    def __init__(self,interface:WorldModelInterface):
        self.interface=interface
    @property
    def specification(self)->ToolSpecification:
        description=files(__package__).joinpath("specification.md").read_text(encoding="utf-8")
        return ToolSpecification(name="Robot State", description=description,arguments=[])

    def run(self, command: Command) -> Return:
        state=self.interface.robot_state()
        rv=Return(id=command.id, command_type=command.command_type, status="Success", results={})
        for field in fields(state):
            rv.results[field.name] = getattr(state, field.name)
        return rv