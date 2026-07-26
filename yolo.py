from std_msgs.msg import String

from ultralytics import YOLO
import cv2
import rospy
import json

class YOLOAdapter():
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = self.load_model(model_path)
    def load_model(self, model_path: str):
        model=YOLO(model_path)
        return model
    def init_yolo(self, webcam):
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
