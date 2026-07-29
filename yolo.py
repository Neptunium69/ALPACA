from std_msgs.msg import String, Header
from sensor_msgs.msg import Image,CameraInfo
from nav_msgs.msg import Odometry
from sensor_msgs.msg import CameraInfo
from alpaca_msgs.msg import ObservationFrame, ObjectDetection

import message_filters
import numpy as np
from math import sqrt

from ultralytics import YOLO
import cv2
from cv_bridge import CvBridge
from tf.transformations import euler_from_quaternion
import rospy
import json

class YOLOAdapter():
    def __init__(self, model_path: str, cammountpos: tuple=(0.15, 0.0, 0.0), cammountrot: tuple=(0.0, np.deg2rad(20), 0.0)):
        self.model_path = model_path
        self.model = self._load_model(model_path)
        self.cammountpos=np.asarray(cammountpos,dtype=float).reshape(3)
        self.cammountrot=np.asarray(cammountrot,dtype=float).reshape(3)
    def init_yolo(self):
        rospy.init_node("yolo_node", anonymous=True)
        self.rate = rospy.Rate(15)
        self.publisher = rospy.Publisher("detections",ObservationFrame,queue_size=1)
        self.bridge = CvBridge()

        self.color_subscriber = message_filters.Subscriber("camera_color", Image, queue_size=1)
        self.camera_info_subscriber = message_filters.Subscriber("camera_info", CameraInfo, queue_size=1)
        self.depth_subscriber = message_filters.Subscriber("camera_depth", Image, queue_size=1)
        self.pose_subscriber=message_filters.Subscriber("vehicle_pose",Odometry,queue_size=1)
        self.synchronizer = message_filters.ApproximateTimeSynchronizer([self.color_subscriber, self.depth_subscriber,self.pose_subscriber,self.camera_info_subscriber], queue_size=10, slop=0.05)
        self.synchronizer.registerCallback(self._image_callback)
        rospy.spin()

    def get_classes(self):
        return list(self.model.names.values())
    
    def _load_model(self, model_path: str) -> YOLO:
        model=YOLO(model_path)
        return model

    def _tometer(self, depth_value: float, encoding:str) -> float:
        """Convert depth value to meters."""
        encoding = encoding.lower()
        if encoding == "16uc1":
            return depth_value.astype(np.float32) / 1000.0  # Assuming depth is in millimeters
        elif encoding=="32fc1":
            return depth_value.astype(np.float32)  # Already in meters
        else:
            raise ValueError("Unsupported depth image format")

    def _get_distance(self, depth, box,depth_encoding):
        """Get depth to the object based on the bounding box and depth image."""
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

        valid_mask = np.isfinite(cropped_depth) & (cropped_depth > 0.1) & (cropped_depth < 10.0)
        valid_y, valid_x = np.nonzero(valid_mask)
        valid_depth = cropped_depth[valid_mask]
        if valid_depth.size == 0:
            return None

        median_depth = np.median(valid_depth)
        sample_index = np.lexsort(((valid_x-cropped_depth.shape[1]/2)**2+(valid_y-cropped_depth.shape[0]/2)**2, np.abs(valid_depth-median_depth)))[0]
        return (
            float(valid_depth[sample_index]),
            int(cropped_x1 + valid_x[sample_index]),
            int(cropped_y1 + valid_y[sample_index]),
        )

    def _image_callback(self, color_msg, depth_msg, pose_msg, camera_info_msg):
        try:
            frame=self.bridge.imgmsg_to_cv2(color_msg,desired_encoding="bgr8")
            depth=self.bridge.imgmsg_to_cv2(depth_msg,desired_encoding="passthrough")
            color_height,color_width=frame.shape[:2]
            depth_height,depth_width=depth.shape[:2]

            # Validations
            if color_msg.header.frame_id != depth_msg.header.frame_id:
                raise ValueError("Color and aligned-depth frames do not match")
            if pose_msg.child_frame_id != "base_link":
                rospy.logwarn(f"Unexpected body frame: {pose_msg.child_frame_id}")
                return
            if (color_width,color_height)!=(depth_width,depth_height):
                raise ValueError("Color and aligned-depth dimensions do not match")
            if (camera_info_msg.width,camera_info_msg.height)!=(depth_width,depth_height):
                raise ValueError("CameraInfo and aligned-depth dimensions do not match")
            if camera_info_msg.header.frame_id!=depth_msg.header.frame_id:
                raise ValueError("CameraInfo and aligned-depth frame IDs do not match")
            if not pose_msg.header.frame_id:
                raise ValueError("Vehicle pose frame ID is empty")

            results = self.model(frame,verbose=False)[0]
            output=ObservationFrame()
            output.header.stamp = depth_msg.header.stamp
            output.header.frame_id = camera_info_msg.header.frame_id

            for box in results.boxes:
                try:
                    class_id = int(box.cls[0].item())
                    class_name=results.names[class_id]
                    confidence=box.conf[0].item()
                except Exception as error:
                    rospy.logwarn("Skipping invalid YOLO detection: %s",error)
                    continue

                target_pos=[None,None,None]
                distance=None

                try:
                    height, width = depth.shape[:2]
                    x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()

                    x1 = max(0, min(width - 1, int(x1)))
                    y1 = max(0, min(height - 1, int(y1)))

                    x2 = max(0, min(width, int(x2)))
                    y2 = max(0, min(height, int(y2)))

                    depth_sample=self._get_distance(depth,box,depth_msg.encoding)

                    current_x=pose_msg.pose.pose.position.x
                    current_y=pose_msg.pose.pose.position.y
                    current_z=pose_msg.pose.pose.position.z
                    q=pose_msg.pose.pose.orientation
                    roll,pitch,yaw=euler_from_quaternion([q.x,q.y,q.z,q.w])

                    if depth_sample is not None:
                        # Find the coordinates of the target.
                        Zc,u,v=depth_sample
                        cx=camera_info_msg.K[2]
                        cy=camera_info_msg.K[5]
                        fx=camera_info_msg.K[0]
                        fy=camera_info_msg.K[4]

                        if fx==0 or fy==0:
                            raise ValueError("Invalid fx or fy values")

                        camera_matrix=np.asarray(camera_info_msg.K,dtype=np.float64).reshape(3,3)
                        distortion=np.asarray(camera_info_msg.D,dtype=np.float64)
                        pixel=np.asarray([[[u,v]]],dtype=np.float64)

                        if distortion.size > 0 and np.any(np.abs(distortion) > 1e-12):
                            if camera_info_msg.distortion_model in ["plumb_bob","rational_polynomial"]:
                                undistorted_pixel=cv2.undistortPoints(pixel,camera_matrix,distortion)
                            elif camera_info_msg.distortion_model=="equidistant":
                                undistorted_pixel=cv2.fisheye.undistortPoints(pixel,camera_matrix,distortion[:4].reshape(4,1))
                            else:
                                raise ValueError(f"Unsupported distortion model: {camera_info_msg.distortion_model}")
                            Xc=undistorted_pixel[0,0,0]*Zc
                            Yc=undistorted_pixel[0,0,1]*Zc
                        else:
                            Xc=(u-cx)*Zc/fx
                            Yc=(v-cy)*Zc/fy

                        body_transform=np.array([[0,0,1],[-1,0,0],[0,-1,0]],dtype=float)
                        camera_roll,camera_pitch,camera_yaw=self.cammountrot
                        camera_rx=np.array([[1,0,0],[0,np.cos(camera_roll),-np.sin(camera_roll)],[0,np.sin(camera_roll),np.cos(camera_roll)]])
                        camera_ry=np.array([[np.cos(camera_pitch),0,np.sin(camera_pitch)],[0,1,0],[-np.sin(camera_pitch),0,np.cos(camera_pitch)]])
                        camera_rz=np.array([[np.cos(camera_yaw),-np.sin(camera_yaw),0],[np.sin(camera_yaw),np.cos(camera_yaw),0],[0,0,1]])
                        body_transform=camera_rz@camera_ry@camera_rx@body_transform
                        pc=np.array([Xc,Yc,Zc])
                        pb=body_transform@pc+self.cammountpos

                        distance = float(np.linalg.norm(pb))

                        rx=np.array([[1,0,0],[0,np.cos(roll),-np.sin(roll)],[0,np.sin(roll),np.cos(roll)]])
                        ry=np.array([[np.cos(pitch),0,np.sin(pitch)],[0,1,0],[-np.sin(pitch),0,np.cos(pitch)]])
                        rz=np.array([[np.cos(yaw),-np.sin(yaw),0],[np.sin(yaw),np.cos(yaw),0],[0,0,1]])

                        r=rz@ry@rx

                        pw=r@pb

                        pv=np.array([current_x,current_y,current_z],dtype=float)

                        pt=pv+pw

                        target_pos=pt.tolist()
                except Exception as error:
                    rospy.logwarn("Could not localize %s detection: %s",class_name,error)

                detection_output=ObjectDetection()
                detection_output.class_name=class_name
                detection_output.confidence=float(confidence)

                detection_output.x_min=float(x1)
                detection_output.y_min=float(y1)
                detection_output.x_max=float(x2)
                detection_output.y_max=float(y2)

                if distance is not None:
                    detection_output.has_depth=True
                    detection_output.optical_depth=float(Zc)
                    detection_output.depth_sample_u=float(u)
                    detection_output.depth_sample_v=float(v)
                    detection_output.camera_range=float(np.linalg.norm(pc))
                else:
                    detection_output.has_depth=False

                if target_pos != [None,None,None]:
                    detection_output.has_world_position=True
                    detection_output.world_position.x=float(target_pos[0])
                    detection_output.world_position.y=float(target_pos[1])
                    detection_output.world_position.z=float(target_pos[2])
                else:
                    detection_output.has_world_position=False

                output.detections.append(detection_output)

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
