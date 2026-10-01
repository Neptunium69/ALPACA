from dataclasses import fields
try:
    from importlib.resources import files
except ImportError:
    from importlib_resources import files

from ...dataclasses import FCExecute
from ...vatools import VAInterface
from ..toolshared import Command,Return,Tool,ToolSpecification

class RobotState(Tool):
    def __init__(self,interface:VAInterface):
        self.interface=interface
        self.specification=ToolSpecification(
            name="robot_state",
            description=files(__name__.rsplit(".",1)[0]).joinpath("specification.md").read_text(encoding="utf-8"),
            arguments=[],
            tag="query",
        )

    def run(self, command: Command) -> Return:
        state=self.interface.execute(FCExecute(function="get_state",arguments={})).data
        rv=Return(id=command.id, command_type=command.command_type, status="Success", results={})
        for field in fields(state):
            rv.results[field.name] = getattr(state, field.name)
        return rv
