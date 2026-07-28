from std_msgs.msg import String
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry

import message_filters
import numpy as np

from ultralytics import YOLO
import cv2
from cv_bridge import CvBridge
from tf.transformations import euler_from_quaternion
import rospy
import json

class YOLOAdapter():
    def __init__(self, model_path: str, image_topic: str = "camera_color", depth_topic: str = "camera_depth"):
        self.model_path = model_path
        self.model = self._load_model(model_path)
        self.image_topic = image_topic
        self.depth_topic = depth_topic

    def init_yolo(self):
        rospy.init_node("yolo_node", anonymous=True)
        self.rate = rospy.Rate(15)
        self.publisher = rospy.Publisher("yolo_detections",String,queue_size=1)
        self.bridge = CvBridge()

        self.color_subscriber = message_filters.Subscriber(self.image_topic, Image, queue_size=1)
        self.depth_subscriber = message_filters.Subscriber(self.depth_topic, Image, queue_size=1)
        self.pose_subscriber=message_filters.Subscriber("vehicle_pose",Odometry,queue_size=1)
        self.synchronizer = message_filters.ApproximateTimeSynchronizer([self.color_subscriber, self.depth_subscriber,self.pose_subscriber], queue_size=10, slop=0.05)
        self.synchronizer.registerCallback(self._image_callback)
        rospy.spin()

    def _load_model(self, model_path: str) -> YOLO:
        model=YOLO(model_path)
        return model

    def _tometer(self, depth_value: float, encoding:str) -> float:
        """Convert depth value to meters."""
        encoding = encoding.lower()
        if encoding in ["16uc1","mono16"]:
            return depth_value.astype(np.float32) / 1000.0  # Assuming depth is in millimeters
        elif encoding=="32fc1":
            return depth_value.astype(np.float32)  # Already in meters
        else:
            raise ValueError("Unsupported depth image format")

    def _get_distance(self, depth, box,depth_encoding):
        """Get distance to the object based on the bounding box and depth image."""
        height, width = depth.shape[:2]
        x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()

        x1 = max(0, min(width - 1, int(x1)))
        y1 = max(0, min(height - 1, int(y1)))

        x2 = max(0, min(width, int(x2)))
        y2 = max(0, min(height, int(y2)))

        # calculate box dimensions
        box_width = x2 - x1
        box_height = y2 - y1

        # this removes the 25% of the bounding box from each side to avoid edge pixels that may be noisy.
        cropped_x1=x1+box_width//4
        cropped_x2=x2-box_width//4
        cropped_y1=y1+box_height//4
        cropped_y2=y2-box_height//4

        cropped_depth = depth[cropped_y1:cropped_y2, cropped_x1:cropped_x2]
        cropped_depth = self._tometer(cropped_depth,depth_encoding)

        samples = cropped_depth.flatten()

        valid = samples[np.isfinite(samples) & (samples > 0.1) & (samples < 10.0)] # extract valid depth values between 0.1 and 10 meters(10 meters is D435 depth camera's maximum range)
        return float(np.median(valid)) if valid.size > 0 else None # return the median of valid depth values, or None if no valid values are found

    def _image_callback(self, color_msg, depth_msg, pose_msg):
        try:
            frame=self.bridge.imgmsg_to_cv2(color_msg,desired_encoding="bgr8")
            depth=self.bridge.imgmsg_to_cv2(depth_msg,desired_encoding="passthrough")
            results = self.model(frame,verbose=False)[0]
            output={
                "class_names": [],
                "confidences": [],
                "position":[]
            }

            for box in results.boxes:
                distance=self._get_distance(depth,box,depth_msg.encoding)

                current_x=pose_msg.pose.pose.position.x
                current_y=pose_msg.pose.pose.position.y
                current_z=pose_msg.pose.pose.position.z
                q=pose_msg.pose.pose.orientation
                roll,pitch,yaw=euler_from_quaternion([q.x,q.y,q.z,q.w])

                if distance!=None:
                    # Find the coordinates of the target.
                    target_x=current_x+(np.cos(yaw)*distance)
                    target_y=current_y+(np.sin(yaw)*distance)
                    target_pos=(target_x,target_y)
                else:
                    target_pos=(None,None)

                class_id = int(box.cls[0].item())
                class_name=results.names[class_id]
                confidence=box.conf[0].item()
                output["class_names"].append(class_name)
                output["confidences"].append(confidence)
                output["position"].append(target_pos)

            message=String()
            message.data=json.dumps(output)
            self.publisher.publish(message)
        except Exception as error:
            rospy.logerr("YOLO callback failed: %s",error)
    def init_yolo_legacy(self, webcam):
        if webcam==None:
            raise ValueError("Webcam is None. Please provide a valid webcam path.")
        rospy.init_node("yolo_node", anonymous=True)
        rate = rospy.Rate(15)
        publisher = rospy.Publisher("yolodetections",String,queue_size=1)
        cap = cv2.VideoCapture(webcam)

        try:
            if not cap.isOpened():
                raise RuntimeError(f"Could not open camera: {webcam}")
            while not rospy.is_shutdown():
                ret, frame = cap.read()
                if not ret:
                    break
                output={
                    "class_names": [],
                    "confidences": [],
                    "bounding_boxes": {
                        "x1s": [],
                        "y1s": [],
                        "x2s": [],
                        "y2s": []
                    }
                }
                result = self.model(frame,verbose=False)[0]
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()
                    class_id = int(box.cls[0].item())
                    class_name = result.names[class_id]
                    confidence = box.conf[0].item()

                    output["class_names"].append(class_name)
                    output["confidences"].append(confidence)
                    output["bounding_boxes"]["x1s"].append(x1)
                    output["bounding_boxes"]["y1s"].append(y1)
                    output["bounding_boxes"]["x2s"].append(x2)
                    output["bounding_boxes"]["y2s"].append(y2)

                message=String()
                message.data=json.dumps(output)
                publisher.publish(message)
                rate.sleep()
        finally:
            cap.release()
