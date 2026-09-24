import os
from importlib.resources import files
from pathlib import Path

import cv2
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

from ..toolshared import Command,Return,Tool,ToolSpecification

class Observe(Tool):
    def __init__(self):
        self.bridge=CvBridge()
        self.specification=ToolSpecification(
            name="Observe",
            description=files(__package__).joinpath("specification.md").read_text(encoding="utf-8"),
            arguments=[],
        )

    def run(self, command:Command) -> Return:
        try:
            calltime=rospy.Time.now()
            while True:
                message=rospy.wait_for_message("camera_color", Image, timeout=5.0)
                if message is None:
                    raise RuntimeError("Camera returned no image message")
                stamp=message.header.stamp
                age_s = (calltime - stamp).to_sec()
                image=self.bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
                height,width=image.shape[:2]
                timestamp=message.header.stamp.to_nsec()
                image_directory=Path(os.environ.get("ALPACA_LOG_DIR","/tmp"))/"images"
                image_directory.mkdir(parents=True,exist_ok=True)
                image_path=str(image_directory/f"camera_image_{timestamp}.png")
                if not cv2.imwrite(image_path,image):
                    raise IOError(f"Could not write {image_path}")
                if(age_s<0.5):
                    break
            
            return Return(id=command.id,command_type=command.command_type,status="Success",results={
                "image_path":image_path,
                "width":width,
                "height":height,
                "timestamp_ns":timestamp,
                "frame_id":message.header.frame_id,
            })
        except Exception as error:
            return Return(id=command.id,command_type=command.command_type,status="Failed",results={"Error":str(error)})
