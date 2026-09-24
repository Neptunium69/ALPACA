from dataclasses import fields
from importlib.resources import files

from ...wmtools.wminterface import WorldModelInterface
from ..toolshared import Command,Return,Tool,ToolSpecification

class RobotState(Tool):
    def __init__(self,interface:WorldModelInterface):
        self.interface=interface
        self.specification=ToolSpecification(
            name="Robot State",
            description=files(__package__).joinpath("specification.md").read_text(encoding="utf-8"),
            arguments=[],
        )

    def run(self, command: Command) -> Return:
        state=self.interface.robot_state()
        rv=Return(id=command.id, command_type=command.command_type, status="Success", results={})
        for field in fields(state):
            rv.results[field.name] = getattr(state, field.name)
        return rv
