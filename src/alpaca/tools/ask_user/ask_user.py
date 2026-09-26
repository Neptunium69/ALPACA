from importlib.resources import files

import rospy
from std_msgs.msg import String

from ..toolshared import Command,Return,Tool,ToolArgument,ToolSpecification


class AskUser(Tool):
    def __init__(self,output_text_publisher:rospy.Publisher,output_image_publisher:rospy.Publisher):
        self.output_text_publisher=output_text_publisher
        self.output_image_publisher=output_image_publisher
        self.specification=ToolSpecification(name="ask_user",description=files(__name__.rsplit(".",1)[0]).joinpath("specification.md").read_text(encoding="utf-8"),arguments=[ToolArgument(name="text_message",required=True,type="str"),ToolArgument(name="image_message",required=False,type="str")])

    def run(self,command:Command)->Return:
        if self.output_text_publisher is None or self.output_image_publisher is None:
            raise RuntimeError("Output publishers has not been initialized.")
        text_message=command.arguments.get("text_message")
        image_message=command.arguments.get("image_message")
        self.output_text_publisher.publish(String(data="" if text_message is None else text_message))
        self.output_image_publisher.publish(String(data="" if image_message is None else image_message))

        input=rospy.wait_for_message("user_input",String)
        if input is not None:
            return Return(id=command.id,command_type=command.command_type,status="Success",results={"User Response":input.data})
        else:
            raise ValueError("User input is none")
