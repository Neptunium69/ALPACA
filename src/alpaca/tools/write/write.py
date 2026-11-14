try:
    from importlib.resources import files
except ImportError:
    from importlib_resources import files

from ...dataclasses import FCExecute
from ...wmtools.wminterface import WorldModelInterface
from ...worldmodel import Object,Size
from ..toolshared import Command,Return,Tool,ToolArgument,ToolSpecification

class Write(Tool):
    def __init__(self,interface:WorldModelInterface):
        self.interface=interface
        self.specification=ToolSpecification(
            name="Write",
            description=files(__name__.rsplit(".",1)[0]).joinpath("specification.md").read_text(encoding="utf-8"),
            arguments=[ToolArgument(name="object",required=True,type="dict")],
            tag="query",
        )

    def run(self, command: Command) -> Return:
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
        detection_classes=self.interface.execute(FCExecute(function="get_detection_classes",arguments={}),submodel="objects").data
        if object_data["class_name"] in detection_classes:
            return Return(id=command.id, command_type=command.command_type, status="Failed", results={"Error":f"Class name {object_data['class_name']!r} is reserved and cannot be used."})
        if not isinstance(attributes,dict) or not all(isinstance(name,str) for name in attributes):
            return Return(id=command.id, command_type=command.command_type, status="Failed", results={"Error":"attributes must be a dictionary with string keys."})
        forbidden_attributes={"Notes","VLMFlag"}.intersection(attributes)
        if forbidden_attributes:
            return Return(id=command.id, command_type=command.command_type, status="Failed", results={"Error":f"Forbidden attribute keys: {', '.join(sorted(forbidden_attributes))}"})

        try:
            position=self.interface.execute(FCExecute(function="bounding_box_to_world",arguments={"bounding_box":object_data["bounding_box"]}),submodel="objects").data
            world_object=Object(id=object_data["id"],class_name=object_data["class_name"],size=Size(),position=position,attributes=dict(attributes))
        except (TypeError,ValueError) as error:
            return Return(id=command.id, command_type=command.command_type, status="Failed", results={"Error":f"Could not calculate object position: {error}"})
        try:
            self.interface.execute(FCExecute(function="write",arguments={"world_object":world_object}),submodel="objects")
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
            results={"Object Written": world_object.id},
        )
