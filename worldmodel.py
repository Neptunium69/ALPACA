from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from copy import deepcopy
from threading import RLock
import rospy
import numpy as np
from tf.transformations import quaternion_matrix
from scipy.spatial.transform import Rotation
import message_filters
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from geometry_msgs.msg import Quaternion
import math
from std_msgs.msg import String
from nav_msgs.msg import Odometry
from sensor_msgs.msg import CameraInfo
from alpaca_msgs.msg import ObservationFrame, ObjectDetection
from sklearn.neighbors import KNeighborsClassifier
from math import e
import json


@dataclass
class Position:
    x: float
    y: float
    z: float

@dataclass
class Size:
    conservative_width: float | None = None
    conservative_height: float | None = None

@dataclass
class Orientation:
    pitch: float
    yaw: float
    roll: float

@dataclass
class Object:
    id: str
    class_name: str
    size: Size
    position: Position
    attributes: dict[str, Any] = field(default_factory=dict)
    confidence_scores: dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        defaults = {
            "existence_rawscore": -2.0,
            "existence_score": 0.1192,
            "positioning_rawscore": -2.0,
            "positioning_score": 0.1192,
        }
        for key, value in defaults.items():
            self.confidence_scores.setdefault(key, value)

@dataclass
class Model:
    objects: list[Object] = field(default_factory=list)

class WorldModel:
    def __init__(self, observations: Model, detection_classes):
        self.observations = observations
        self.detection_classes = tuple(detection_classes)
        self.record=[]
        self._bridge = CvBridge()
        self._latest_frame=None
        self._latest_depth=None
        self._lock = RLock()

    def initwm(self):
        # Automatic world model updater
        rospy.init_node("world_model_node", anonymous=True)
        self.detection_subscriber=message_filters.Subscriber("detections", ObservationFrame)
        self.depth_subscriber=message_filters.Subscriber("camera_depth", Image)
        syncronizer=message_filters.ApproximateTimeSynchronizer([self.depth_subscriber,self.detection_subscriber], queue_size=10, slop=0.05)
        syncronizer.registerCallback(self._update_callback)
        rospy.spin()

    @staticmethod
    def sigmoid(x):
        return 1/(1+e**-x)

    def _update_callback(self, depth, detections):
        try:
            detection_list=detections.detections
        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as error:
            rospy.logerr("Invalid detection message: %s", error)
            return

        with self._lock:
            self._latest_frame=detections
            self._latest_depth=depth
            try:
                working_objects = deepcopy(self.observations.objects)
                self._apply_detections(working_objects, detection_list, detections, depth)
            except Exception as error:
                rospy.logerr("World-model update failed: %s", error)
                return

            self.observations.objects[:] = working_objects

    def _apply_detections(self, objects: list[Object], detection_list: list[ObjectDetection], frame: ObservationFrame, depthmap: Image) -> None:
        observed_list=[]
        for detection in detection_list:
            detected_size=None
            fx=frame.camera_info.K[0]
            fy=frame.camera_info.K[4]
            if detection.has_depth and detection.optical_depth>0 and fx>0 and fy>0:
                detected_width=((detection.x_max-detection.x_min)*detection.optical_depth)/fx
                detected_height=((detection.y_max-detection.y_min)*detection.optical_depth)/fy
                if detected_width>0 and detected_height>0:
                    detected_size=Size(detected_width,detected_height)

            if(detection.has_world_position):
                if len(objects)>0:
                    flag=0
                    for object in objects:
                        if((object.class_name==detection.class_name) and (object.position.x-0.1)<detection.world_position.x<(object.position.x+0.1) and ((object.position.y-0.1)<detection.world_position.y<(object.position.y+0.1)) and ((object.position.z-0.1)<detection.world_position.z<(object.position.z+0.1))):
                            rawscore = object.confidence_scores.get("existence_rawscore")
                            if rawscore is None:
                                rawscore = -2.0
                            rawscore += 0.1*detection.confidence
                            object.confidence_scores["existence_rawscore"] = rawscore
                            flag=1
                            observed_list.append(object.id)
                            break
                    if(flag==1):
                        object.confidence_scores["existence_score"]=self.sigmoid(object.confidence_scores["existence_rawscore"])
                    if(detection.has_depth and flag==1):
                        distance_scaled=detection.camera_range/10
                        positioning_rawscore = object.confidence_scores.get("positioning_rawscore")
                        if positioning_rawscore is None:
                            positioning_rawscore = -2.0
                        positioning_increment = 0.1*((1-distance_scaled)**2+0.05)
                        positioning_rawscore += positioning_increment
                        object.confidence_scores["positioning_rawscore"] = positioning_rawscore
                        object.confidence_scores["positioning_score"]=self.sigmoid(positioning_rawscore)
                        object.position.x=(object.position.x*object.confidence_scores["positioning_score"])+((1-object.confidence_scores["positioning_score"])*detection.world_position.x)
                        object.position.y=(object.position.y*object.confidence_scores["positioning_score"])+((1-object.confidence_scores["positioning_score"])*detection.world_position.y)
                        object.position.z=(object.position.z*object.confidence_scores["positioning_score"])+((1-object.confidence_scores["positioning_score"])*detection.world_position.z)
                        if detected_size is not None:
                            if object.size.conservative_width is None or object.size.conservative_height is None:
                                object.size=detected_size
                            else:
                                object.size.conservative_width=(object.size.conservative_width*object.confidence_scores["positioning_score"])+((1-object.confidence_scores["positioning_score"])*detected_size.conservative_width)
                                object.size.conservative_height=(object.size.conservative_height*object.confidence_scores["positioning_score"])+((1-object.confidence_scores["positioning_score"])*detected_size.conservative_height)

                    else:
                        pass

                    if(flag==0):
                        id=f"{detection.class_name}_{self._get_latest(detection.class_name, objects)+1}"
                        objects.append(Object(id=id, class_name=detection.class_name, size=detected_size if detected_size is not None else Size(), position=Position(detection.world_position.x,detection.world_position.y,detection.world_position.z),confidence_scores={"existence_rawscore":-2.0,"existence_score":self.sigmoid(-2.0)}))
                        observed_list.append(id)
                        for i in range(len(objects)):
                            if objects[i].id==id:
                                objects[i].confidence_scores["positioning_rawscore"]=-2.0
                                objects[i].confidence_scores["positioning_score"]=self.sigmoid(objects[-1].confidence_scores["positioning_rawscore"])
                else:
                    id=f"{detection.class_name}_{self._get_latest(detection.class_name, objects)+1}"
                    observed_list.append(id)
                    objects.append(Object(id=id, class_name=detection.class_name, size=detected_size if detected_size is not None else Size(), position=Position(detection.world_position.x,detection.world_position.y,detection.world_position.z),confidence_scores={"existence_rawscore":-2.0,"existence_score":self.sigmoid(-2.0)}))
                    for i in range(len(objects)):
                        if objects[i].id==id:
                            objects[i].confidence_scores["positioning_rawscore"]=-2.0
                            objects[i].confidence_scores["positioning_score"]=self.sigmoid(objects[-1].confidence_scores["positioning_rawscore"])

                # Prevents the rawscore form being too large, which may cause world model updates to be slow if the object's position is changed.
                if(objects[-1].confidence_scores["existence_rawscore"]>5.0):
                    objects[-1].confidence_scores["existence_rawscore"]=5.0

        for object in objects:
            if object.attributes.get("VLMFlag")==1:
                observed_list.append(object.id)

        # First check distance
        unobserved_objects=[]
        campos=self._camera_transform(frame,0)[:3,3]
        for object in objects:
            if object.id not in observed_list:
                if math.sqrt(((object.position.x-campos[0])**2)+((object.position.y-campos[1])**2)+((object.position.z-campos[2])**2))<=10.0:
                    unobserved_objects.append(object)

        caminfo=frame.camera_info

        fx = caminfo.K[0]
        fy = caminfo.K[4]
        cx = caminfo.K[2]
        cy = caminfo.K[5]

        unobserved_objects_new=[]

        for world_object in unobserved_objects:
            objpos_c=self.w2c(frame,[world_object.position.x,world_object.position.y,world_object.position.z])
            z=objpos_c[2]
            if(z<=0):
                continue

            u=fx*(objpos_c[0]/z)+cx
            v=fy*(objpos_c[1]/z)+cy
            center_inside_margin=(20<u<(caminfo.width-20)) and (20<v<(caminfo.height-20))
            if(not center_inside_margin):
                continue

            conservative_width=world_object.size.conservative_width
            conservative_height=world_object.size.conservative_height
            if(conservative_width is None or conservative_height is None or conservative_width<=0 or conservative_height<=0):
                continue

            pred_w_obj=(fx*conservative_width)/z
            pred_h_obj=(fy*conservative_height)/z

            x1=u-(pred_w_obj/2)
            x2=u+(pred_w_obj/2)
            y1=v-(pred_h_obj/2)
            y2=v+(pred_h_obj/2)

            x1_i=max(0, int(math.floor(x1)))
            x2_i=min(caminfo.width, int(math.ceil(x2)))
            y1_i=max(0, int(math.floor(y1)))
            y2_i=min(caminfo.height, int(math.ceil(y2)))

            intersection_w=max(0.0,min(x2_i,caminfo.width)-max(x1_i,0.0))
            intersection_h=max(0.0,min(y2_i,caminfo.height)-max(y1_i,0.0))
            visible_fraction=(intersection_w*intersection_h)/(pred_w_obj*pred_h_obj)

            if(visible_fraction<0.8):
                continue

            depthmap_new=np.array([])
            depthmap_data=self._bridge.imgmsg_to_cv2(depthmap.data)

            if depthmap.encoding=="16UC1":
                for row in depthmap_data:
                    for pixel in row:
                        pixel=pixel/1000
                    

            for x in range(x1_i,x2_i):
                np.append(depthmap_new,[])
                for y in range(y1_i,y2_i):
                    if(type(depthmap_data)==int):
                        if(depthmap_data[x,y]<(z-0.2)):
                            np.append(depthmap_new[x],0)
                        elif((z+0.2)>depthmap_data[x,y]>(z-0.2)):
                            np.append(depthmap_new[x],1)
                        elif(depthmap_new[x]>(z+0.2)):
                            np.append(depthmap_new[x],2)
                    else:
                        np.append(depthmap_new[x],3)
                
            rows,cols=np.indices(depthmap_new.shape)
            known_mask=depthmap_new!=3
            unknown_mask=depthmap_new==3

            if known_mask.any() and unknown_mask.any():
                known_coordinates=np.column_stack((rows[known_mask],cols[known_mask]))
                known_categories=depthmap_new[known_mask]
                unknown_coordinates=np.column_stack((rows[unknown_mask],cols[unknown_mask]))

                neighbour_count=min(5,len(known_categories))
                imputer=KNeighborsClassifier(n_neighbors=neighbour_count,weights="distance")
                imputer.fit(known_coordinates,known_categories)
                depthmap_new[unknown_mask]=imputer.predict(unknown_coordinates)

            closer=0
            samedistance=0
            further=0

            for i in depthmap_new:
                for j in i:
                    if j==0:
                        closer+=1
                    elif j==1:
                        samedistance+=1
                    elif j==2:
                        further+=1

            fnear=(closer+samedistance)/(closer+samedistance+further)
            if(fnear>=0.3):
                continue
            
            unobserved_objects_new.append(world_object)

        for object in unobserved_objects_new:
            for i in range(len(self.observations.objects)):
                if(object.id==self.observations.objects[i].id):
                    objects[i].confidence_scores["positioning_rawscore"]-=0.1
                    objects[i].confidence_scores["existence_rawscore"]-=0.1

                    objects[i].confidence_scores["positioning_score"]=self.sigmoid(self.observations.objects[i].confidence_scores["positioning_rawscore"])
                    objects[i].confidence_scores["existence_score"]=self.sigmoid(self.observations.objects[i].confidence_scores["existence_rawscore"])

        pop_indexes=[]
        for i in range(len(self.observations.objects)):
            if self.observations.objects[i].confidence_scores["existence_score"]<0.08:
                self._remove_object(object_id=self.observations.objects[i].id)
        
    def _camera_transform(self,observation: ObservationFrame,flag: int=0)->np.ndarray:
        pose=observation.vehicle_pose.pose.pose
        q_WB=pose.orientation

        T_WB=np.eye(4,dtype=float)
        T_WB[:3,:3]=Rotation.from_quat([q_WB.x,q_WB.y,q_WB.z,q_WB.w]).as_matrix()
        T_WB[:3,3]=[pose.position.x,pose.position.y,pose.position.z]

        camera_transform=observation.camera_pose_in_body.transform
        q_BC=camera_transform.rotation

        T_BC=np.eye(4,dtype=float)
        T_BC[:3,:3]=Rotation.from_quat([q_BC.x,q_BC.y,q_BC.z,q_BC.w]).as_matrix()
        T_BC[:3,3]=[camera_transform.translation.x,camera_transform.translation.y,camera_transform.translation.z]

        T_WC=T_WB@T_BC

        if flag==0:
            return T_WC
        if flag==1:
            return np.linalg.inv(T_WC)

        raise ValueError("flag must be 0 for camera-to-world or 1 for world-to-camera")

    def w2c(self,frame, position:np.ndarray)->np.ndarray:
        position=np.append(np.asarray(position,dtype=float),1.0)
        transform=self._camera_transform(frame,1)
        newpos=transform@position
        return newpos[:3]
    
    def c2w(self,frame, position:np.ndarray)->np.ndarray:
        position=np.append(np.asarray(position,dtype=float),1.0)
        transform=self._camera_transform(frame,0)
        newpos=transform@position
        return newpos[:3]

    def bounding_box_to_world(self,bounding_box)->Position:
        with self._lock:
            if self._latest_frame is None or self._latest_depth is None:
                raise ValueError("No synchronized camera observation is available")

            box=np.asarray(bounding_box,dtype=float)
            if box.shape!=(4,) or not np.all(np.isfinite(box)):
                raise ValueError("bounding_box must contain four finite pixel coordinates: [x_min,y_min,x_max,y_max]")

            x1,y1,x2,y2=box
            if x2<=x1 or y2<=y1:
                raise ValueError("bounding_box maximum coordinates must be greater than its minimum coordinates")

            depthmap_data=self._bridge.imgmsg_to_cv2(self._latest_depth,desired_encoding="passthrough")
            if self._latest_depth.encoding=="16UC1":
                depthmap_data=depthmap_data.astype(np.float32)/1000.0
            elif self._latest_depth.encoding=="32FC1":
                depthmap_data=depthmap_data.astype(np.float32)
            else:
                raise ValueError(f"Unsupported depth image encoding: {self._latest_depth.encoding}")

            height,width=depthmap_data.shape[:2]
            if x2<=0 or y2<=0 or x1>=width or y1>=height:
                raise ValueError("bounding_box does not overlap the depth image")
            x1_i=max(0,min(width-1,int(math.floor(x1))))
            x2_i=max(0,min(width,int(math.ceil(x2))))
            y1_i=max(0,min(height-1,int(math.floor(y1))))
            y2_i=max(0,min(height,int(math.ceil(y2))))
            if x2_i<=x1_i or y2_i<=y1_i:
                raise ValueError("bounding_box does not overlap the depth image")

            box_width=x2_i-x1_i
            box_height=y2_i-y1_i
            sample_x1=x1_i+box_width//4
            sample_x2=x2_i-box_width//4
            sample_y1=y1_i+box_height//4
            sample_y2=y2_i-box_height//4
            cropped_depth=depthmap_data[sample_y1:sample_y2,sample_x1:sample_x2]
            valid_mask=np.isfinite(cropped_depth) & (cropped_depth>0.1) & (cropped_depth<10.0)
            if not valid_mask.any():
                raise ValueError("No valid depth was found inside the bounding box")

            valid_y,valid_x=np.nonzero(valid_mask)
            valid_depth=cropped_depth[valid_mask]
            median_depth=float(np.median(valid_depth))
            median_index=int(np.argmin(np.abs(valid_depth-median_depth)))
            u=float(sample_x1+valid_x[median_index])
            v=float(sample_y1+valid_y[median_index])

            caminfo=self._latest_frame.camera_info
            fx=caminfo.K[0]
            fy=caminfo.K[4]
            cx=caminfo.K[2]
            cy=caminfo.K[5]
            if fx<=0 or fy<=0:
                raise ValueError("Camera focal lengths must be positive")

            camera_position=[((u-cx)*median_depth)/fx,((v-cy)*median_depth)/fy,median_depth]
            world_position=self.c2w(self._latest_frame,camera_position)
            return Position(float(world_position[0]),float(world_position[1]),float(world_position[2]))

    # Insert record and stop record functions

    def _get_latest(self,class_name:str, objects: list[Object] | None = None)->int:
        with self._lock:
            largest_number = 0
            source_objects = self.observations.objects if objects is None else objects

            for world_object in source_objects:
                if world_object.class_name == class_name:
                    number = int(world_object.id.rsplit("_", 1)[1])

                    if number > largest_number:
                        largest_number = number

            return largest_number

    def query_world(self,object_id: str | None = None,class_name: str | None = None,) -> list[Object]:
        """Return objects matching an ID, a class, or all objects."""
        with self._lock:
            if object_id is not None:
                result = [
                    world_object
                    for world_object in self.observations.objects
                    if world_object.id == object_id
                ]
            elif class_name is not None:
                result = [
                    world_object
                    for world_object in self.observations.objects
                    if world_object.class_name == class_name
                ]
            else:
                result = self.observations.objects

            return deepcopy(result)

    def _add_object(self, world_object: Object) -> None:
        with self._lock:
            if any(existing_object.id == world_object.id for existing_object in self.observations.objects):
                raise ValueError(f"An object with ID {world_object.id!r} already exists.")

            self.observations.objects.append(world_object)

    def _remove_object(self, object_id: str) -> Object:
        with self._lock:
            for index, world_object in enumerate(self.observations.objects):
                if world_object.id == object_id:
                    return self.observations.objects.pop(index)

            raise KeyError(f"No object with ID {object_id!r} exists.")

    def write(self,world_object: Object) -> None:
        candidate = deepcopy(world_object)
        for detection_class in self.detection_classes:
            if f"{detection_class}_" not in candidate.class_name:
                candidate.attributes["Notes"]="Artificially written by VLM, not added by automatic updater"
                candidate.attributes["VLMFlag"]=1
                candidate.confidence_scores["existence_rawscore"]=0.0
                candidate.confidence_scores["existence_score"]=float(self.sigmoid(candidate.confidence_scores["existence_rawscore"]))
            else:
                raise ValueError(f"Class name {candidate.class_name} is invalid because it contains the reserved phrase '{detection_class}_'")

        with self._lock:
            self._add_object(candidate)
