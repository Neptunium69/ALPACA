from __future__ import annotations

import json
from importlib.resources import files

import rospy
from std_msgs.msg import String

from ..toolshared import Command,Return,Tool,ToolArgument,ToolSpecification


class Finish(Tool):
    def __init__(self,output_text_publisher:rospy.Publisher,output_image_publisher:rospy.Publisher,completion_publisher:rospy.Publisher | None=None):
        self.output_text_publisher=output_text_publisher
        self.output_image_publisher=output_image_publisher
        self.completion_publisher=completion_publisher
        self.specification=ToolSpecification(name="Finish",description=files(__name__.rsplit(".",1)[0]).joinpath("specification.md").read_text(encoding="utf-8"),arguments=[ToolArgument(name="completion_text_message",required=False,type="str"),ToolArgument(name="completion_image_message",required=False,type="str")])

    def run(self,command:Command)->Return:
        if self.output_text_publisher is None or self.output_image_publisher is None:
            raise RuntimeError("Output publishers has not been initialized.")

        completion_text=command.arguments.get("completion_text_message")
        completion_image=command.arguments.get("completion_image_message")
        if completion_text is None and completion_image is None:
            return Return(id=command.id,command_type=command.command_type,status="Failed",results={"Error":"At least one of completion_text_message, completion_image_message must be provided."})
        completion_payload={"text":"" if completion_text is None else completion_text,"image":"" if completion_image is None else completion_image}
        if self.completion_publisher is not None:
            self.completion_publisher.publish(String(data=json.dumps(completion_payload,ensure_ascii=False)))
        else:
            self.output_text_publisher.publish(String(data=completion_payload["text"]))
            self.output_image_publisher.publish(String(data=completion_payload["image"]))

        return Return(id=command.id, command_type=command.command_type, status="Success", results={"Mission Completed":True})
