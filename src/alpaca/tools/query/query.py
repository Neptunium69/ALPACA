try:
    from importlib.resources import files
except ImportError:
    from importlib_resources import files

from ...dataclasses import FCExecute
from ...wmtools.wminterface import WorldModelInterface
from ..toolshared import Command,Return,Tool,ToolArgument,ToolSpecification

class Query(Tool):
    def __init__(self,interface:WorldModelInterface):
        self.interface=interface
        self.specification=ToolSpecification(
            name="Query",
            description=files(__name__.rsplit(".",1)[0]).joinpath("specification.md").read_text(encoding="utf-8"),
            arguments=[
                ToolArgument(name="object_id",required=False,type="str"),
                ToolArgument(name="class_name",required=False,type="str"),
            ],
            tag="query",
        )

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
            objects = self.interface.execute(FCExecute(function="query_world",arguments={"object_id":object_id,"class_name":class_name})).data

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
