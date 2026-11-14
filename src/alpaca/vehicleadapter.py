from __future__ import annotations

import time
import numpy as np
import torch
from scipy.ndimage import label
from scipy.spatial import cKDTree
from typing import Protocol,Any
from dataclasses import dataclass,field
from math import pi,isfinite
import actionlib
from actionlib_msgs.msg import GoalStatus
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from std_msgs.msg import Header
import sensor_msgs.point_cloud2 as pc2
import rospy
import message_filters
from threading import Condition,Lock,Thread
from copy import deepcopy
from nav_msgs.msg import Odometry,Path
from nav_msgs.srv import GetPlan,GetPlanRequest
from geometry_msgs.msg import PoseStamped,TwistStamped,Quaternion
from alpaca_msgs.msg import SpatialObservationFrame, VoxelObservation as VoxelObservationMsg, ObservationFrame
from sensor_msgs.msg import Image, BatteryState,CameraInfo, PointCloud2
from cv_bridge import CvBridge
from mavros_msgs.msg import State as MavrosState
from tf.transformations import euler_from_quaternion,quaternion_from_euler,quaternion_matrix
from . import worldmodel as wm
from . import risk
from .worldmodel import Position,Orientation
from sensor_msgs.msg import Imu
from .risk import PathVoxelAssessment, VantagePoint
import random
from .vatools import VARegistry,Function
from .depth import depth_points

@dataclass
class State():
    adapter: str
    data: dict[str,Any] = field(default_factory=dict)

    def __getattr__(self,name:str)->Any:
        data=self.__dict__.get("data")
        if data is not None and name in data:
            return data[name]
        raise AttributeError(f"{type(self).__name__!s} has no state value {name!r}.")

    def __setattr__(self,name:str,value:Any)->None:
        if name in {"adapter","data"} or "data" not in self.__dict__:
            object.__setattr__(self,name,value)
        else:
            self.data[name]=value

class NavigationReadinessError(RuntimeError):
    pass

@dataclass
class VoxelObservation():
    location: tuple[int,int,int]
    occupied: int
    free: int


class VehicleAdapter(Protocol):
    state: State
    registry:VARegistry

class UGVAdapter(VehicleAdapter):
    def __init__(self, worldmodel: wm.WorldModel, risk:risk.RiskCal | None=None, **kwargs) -> None:
        self.registry=VARegistry([
            Function(name="init_sensordata",arguments={}),
            Function(name="get_navigation_frame",arguments={}),
            Function(name="get_state",arguments={}),
            Function(name="require_navigation_ready",arguments={"action_name":("str",True)}),
            Function(name="get_collision_geometry",arguments={}),
            Function(name="plan_path",arguments={"destination":("Position",True),"orientation":("Orientation",True),"frame_id":("str",False),"tolerance":("float",False)}),
            Function(name="navigate",arguments={"destination":("Position",True),"orientation":("Orientation",True),"frame_id":("str",False),"path":("Path",True),"arrival_tolerance":("float",False)}),
            Function(name="send_occ",arguments={"voxels":("list",True)}),
            Function(name="scan",arguments={"yaw_margin":("float",False)}),
            Function(name="vantagepoint",arguments={"pathvoxel":("PathVoxelAssessment",True),"max_updates":("int",False)}),
            Function(name="hiddenpoints",arguments={"object_id":("str",True),"state":("State",True),"distance_behind":("float",False)}),
            Function(name="hiddenpoint",arguments={"object_id":("str",True),"state":("State",True),"distance_behind":("float",False)}),
            Function(name="image",arguments={"targpos":("Position",True),"yaw_margin":("float",False)}),
        ])
        objectmodel=worldmodel.get("objects")
        voxelmodel=worldmodel.get("voxels_master")
        if not isinstance(objectmodel,wm.ObjectWorldModel) or not isinstance(voxelmodel,wm.VoxelWorldModel):
            raise TypeError("UGVAdapter requires objects and voxels_master submodels of the appropriate types.")
        self.objectmodel=objectmodel
        self.voxelmodel=voxelmodel
        self.riskcal=risk
        self.state_lock=Lock()
        self.client=None
        self.make_plan_service=None
        navigation_frame=kwargs.get("navigation_frame","map")
        if not isinstance(navigation_frame,str) or not navigation_frame.strip():
            raise ValueError("Navigation frame must be a non-empty string.")
        self._navigation_frame:str=navigation_frame
        self.voxel_size = 0.05
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vehicle_pose=None
        self.last_pose_monotonic:float | None=None
        self.max_pose_age=0.5
        self.last_state_monotonic:float | None=None
        self.max_state_age=2.0
        self.lidar_points_world=None
        self.lidar_origin_world=None
        self.lidar_condition=Condition()
        self.max_vp_voxels=2000000
        self.pending_lidar=None
        self.camera_condition=Condition()
        self.pending_camera=None
        self.camera_bridge=CvBridge()
        self.camera_to_body_translation=np.asarray(kwargs.get("cammountpos",(0.15,0.0,0.0)),dtype=np.float64)
        camera_angles=np.asarray(kwargs.get("cammountrot",(0.0,0.0,0.0)),dtype=np.float64)
        if self.camera_to_body_translation.shape!=(3,) or camera_angles.shape!=(3,) or not np.all(np.isfinite(self.camera_to_body_translation)) or not np.all(np.isfinite(camera_angles)):
            raise ValueError("Camera mount position and rotation must contain three finite values.")
        self.camera_to_body_rotation=quaternion_matrix(quaternion_from_euler(*camera_angles))[:3,:3]@np.array([[0,0,1],[-1,0,0],[0,-1,0]],dtype=np.float64)
        self.stopping=False
        self.stopped=False
        self.vehicle_min = np.array([-0.15, -0.17, -0.25])
        self.vehicle_max = np.array([0.15, 0.17, 0.0])
        self.hiddenpoint_neighbourhood_radius=0.5
        self.maximum_nearby_occupied_voxels=5
        self.height_margin = 0.05
        self.search_offset = np.array([0.30, 0.30, 0.05])
        self.roi_min = self.vehicle_min - self.search_offset
        self.lidar_to_body_translation = np.array([-0.011, -0.02329, 0.04412],dtype=np.float64)
        self.lidar_to_body_rotation = np.eye(3,dtype=np.float64)
        self.whereitisnt:PoseStamped|None=None
        self.intermediate_wp_marg=0.4
        self.final_wp_marg=0.15
        self.moveflag=False
        self.theta=None
        self.current_yawrate=None
        self.num_lid_vox=500
        self.cam_stride=3

    @property
    def navigation_frame(self)->str:
        return self._navigation_frame

    def get_navigation_frame(self)->str:
        return self.navigation_frame

    def init_sensordata(self) -> None:
        with self.state_lock:
            self.state=State(
                adapter=type(self).__name__,
                data={
                    "battery_level":None,
                    "position":Position(x=0.0,y=0.0,z=0.0),
                    "orientation":Orientation(pitch=0.0,yaw=0.0,roll=0.0),
                    "armed":False,
                    "mode":"",
                    "connected":False,
                },
            )
        translation=rospy.get_param("/mapping/extrinsic_T",self.lidar_to_body_translation.tolist())
        rotation=rospy.get_param("/mapping/extrinsic_R",self.lidar_to_body_rotation.reshape(-1).tolist())
        self.lidar_to_body_translation=np.asarray(translation,dtype=np.float64)
        self.lidar_to_body_rotation=np.asarray(rotation,dtype=np.float64).reshape(3,3)
        if self.lidar_to_body_translation.shape!=(3,) or not np.all(np.isfinite(self.lidar_to_body_translation)) or not np.all(np.isfinite(self.lidar_to_body_rotation)):
            raise ValueError("FAST-LIO extrinsic translation and rotation must contain finite values.")
        self.client=actionlib.SimpleActionClient("/move_base",MoveBaseAction)

        self.voxel_publisher = rospy.Publisher("voxel_observations",SpatialObservationFrame,queue_size=1)
        self.camera_voxel_publisher=rospy.Publisher("camera_voxel_observations",SpatialObservationFrame,queue_size=1)
        self.obstacle_publisher=rospy.Publisher("/alpaca/blocked_voxels",PointCloud2,queue_size=1,latch=True)
        self.movement_publisher = rospy.Publisher("/mavros/setpoint_position/local",PoseStamped,queue_size=1)
        self.pose_publisher = rospy.Publisher("vehicle_pose",Odometry,queue_size=1)
        self.camera_info_publisher = rospy.Publisher("camera_info", CameraInfo, queue_size=1)
        self.battery_publisher = rospy.Publisher("battery_level",BatteryState,queue_size=1)
        self.color_publisher = rospy.Publisher("camera_color",Image,queue_size=1)
        self.depth_publisher = rospy.Publisher("camera_depth",Image,queue_size=1)
        if not self.client.wait_for_server(rospy.Duration(secs=5)):
            raise RuntimeError("The move_base action server is unavailable.")
        try:
            rospy.wait_for_service("/move_base/make_plan",timeout=5.0)
        except rospy.ROSException as error:
            raise RuntimeError("The move_base planning service is unavailable.") from error
        self.make_plan_service=rospy.ServiceProxy("/move_base/make_plan",GetPlan)

        self.gnss_position_subscriber = message_filters.Subscriber("/Odometry",Odometry,queue_size=10)
        self.gnss_position_subscriber.registerCallback(self._pose_callback)
        self.color_subscriber = message_filters.Subscriber("/camera/color/image_raw", Image, queue_size=1)
        self.depth_subscriber = message_filters.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, queue_size=1)
        self.camera_info_subscriber = message_filters.Subscriber("/camera/aligned_depth_to_color/camera_info",CameraInfo,queue_size=1)
        self.battery_subscriber = rospy.Subscriber("/mavros/battery", BatteryState, queue_size=1,callback=self._battery_callback)
        self.mavros_state_subscriber = rospy.Subscriber("/mavros/state",MavrosState,queue_size=1,callback=self._mavros_state_callback)
        self.lidar_subscriber = message_filters.Subscriber("/cloud_registered_body",PointCloud2,queue_size=5)
        self.imu_subscriber=rospy.Subscriber("/mavros/imu/data",Imu,self._imu_callback,queue_size=1)

        self.px4_pose:PoseStamped|None=None
        self.randompx4thing=rospy.Subscriber("/mavros/local_position/pose",PoseStamped,callback=self._px4_callback)
        self.camera_synchronizer = message_filters.ApproximateTimeSynchronizer([self.color_subscriber,self.depth_subscriber,self.camera_info_subscriber], queue_size=10, slop=0.05)
        self.camera_synchronizer.registerCallback(self._adapter_callback)
        self.camera_voxel_synchronizer=message_filters.ApproximateTimeSynchronizer([self.depth_subscriber,self.camera_info_subscriber,self.gnss_position_subscriber],queue_size=10,slop=0.05)
        self.camera_worker=Thread(target=self._process_camera,daemon=True)
        self.camera_worker.start()
        self.camera_voxel_synchronizer.registerCallback(self._queue_camera)
        self.lidar_synchronizer = message_filters.ApproximateTimeSynchronizer([self.lidar_subscriber,self.gnss_position_subscriber],queue_size=10,slop=0.03)
        self.lidar_worker=Thread(target=self._process_lidar,daemon=True)
        self.lidar_worker.start()
        self.lidar_synchronizer.registerCallback(self._queue_lidar)
        self.timer = rospy.Timer(rospy.Duration.from_sec(0.05), lambda event: self.update_target())

    def _queue_lidar(self,cloud:PointCloud2,vehicle_pose:Odometry)->None:
        with self.lidar_condition:
            self.pending_lidar=(cloud,vehicle_pose)
            self.lidar_condition.notify()

    def _process_lidar(self)->None:
        while not rospy.is_shutdown():
            with self.lidar_condition:
                self.lidar_condition.wait_for(lambda:self.pending_lidar is not None or rospy.is_shutdown(),timeout=0.1)
                if rospy.is_shutdown(): return
                pending=self.pending_lidar
                self.pending_lidar=None
            if pending is None: continue
            try:
                self._lidar_callback(*pending)
            except Exception as error:
                rospy.logerr("LiDAR processing failed: %s",error)

    def _queue_camera(self,depth_msg:Image,camera_info:CameraInfo,vehicle_pose:Odometry)->None:
        with self.camera_condition:
            self.pending_camera=(depth_msg,camera_info,vehicle_pose)
            self.camera_condition.notify()

    def _process_camera(self)->None:
        while not rospy.is_shutdown():
            with self.camera_condition:
                self.camera_condition.wait_for(lambda:self.pending_camera is not None or rospy.is_shutdown(),timeout=0.1)
                if rospy.is_shutdown(): return
                pending=self.pending_camera
                self.pending_camera=None
            if pending is None: continue
            try:
                self._camera_callback(*pending)
            except Exception as error:
                rospy.logerr("Camera voxel processing failed: %s",error)

    def _imu_callback(self,msg:Imu):
        self.current_yawrate=msg.angular_velocity.z

    def update_target(self):
        if self.last_pose_monotonic is None or self.px4_pose is None:
            return
        state=self.get_state()
        if self.whereitisnt is None:
            self.whereitisnt=PoseStamped()
            self.whereitisnt.header.stamp = rospy.Time.now()
            self.whereitisnt.pose.position.x=state.data["position"].x
            self.whereitisnt.pose.position.y=state.data["position"].y
            self.whereitisnt.pose.position.z=state.data["position"].z

            q=quaternion_from_euler(state.data["orientation"].roll,state.data["orientation"].pitch,state.data["orientation"].yaw)
            self.whereitisnt.pose.orientation=Quaternion(x=float(q[0]),y=float(q[1]),z=float(q[2]),w=float(q[3]))
        if self.moveflag==False or getattr(self,"stopping",False):
            whereitis=deepcopy(self.px4_pose)
            whereitis.header.stamp=rospy.Time.now()
            self.movement_publisher.publish(whereitis)
        else:
            whereitisnt=whereitis=self._waypointconverter(state,self.px4_pose,deepcopy(self.whereitisnt))
            self.movement_publisher.publish(whereitisnt)

    def vantagepoint(self,pathvoxel:PathVoxelAssessment,max_updates:int=100)->set[tuple[int,int,int]]:
        if not isinstance(max_updates,int) or isinstance(max_updates,bool) or max_updates<1:
            raise ValueError("max_updates must be a positive integer.")
        if not isinstance(pathvoxel,PathVoxelAssessment):
            raise TypeError("pathvoxel must be a PathVoxelAssessment.")
        if not pathvoxel.unknown_voxels:
            return set()
        if self.riskcal is None:
            raise RuntimeError("Vantage search requires the destination risk settings.")

        voxel_size=self.voxelmodel.get_voxel_size()
        state=self.get_state()
        position=state.data.get("position")
        if not isinstance(position,Position):
            raise RuntimeError("Vehicle position is unavailable.")
        current_voxel=self.voxelmodel.world_position_to_voxel(position)
        candidate_z=current_voxel[2]

        remaining=set(pathvoxel.unknown_voxels)
        components:list[set[tuple[int,int,int]]]=[]
        while remaining:
            seed=remaining.pop()
            component={seed}
            stack=[seed]
            while stack:
                x,y,z=stack.pop()
                neighbours={
                    (x+dx,y+dy,z+dz)
                    for dx in (-1,0,1)
                    for dy in (-1,0,1)
                    for dz in (-1,0,1)
                    if dx!=0 or dy!=0 or dz!=0
                }
                connected=neighbours & remaining
                remaining.difference_update(connected)
                component.update(connected)
                stack.extend(connected)
            components.append(component)
        relevant_unknown=min(
            components,
            key=lambda component:min(
                (x-current_voxel[0])**2+(y-current_voxel[1])**2+(z-current_voxel[2])**2
                for x,y,z in component
            ),
        )

        maximum_x=max(abs(float(self.vehicle_min[0])),abs(float(self.vehicle_max[0])))
        maximum_y=max(abs(float(self.vehicle_min[1])),abs(float(self.vehicle_max[1])))
        footprint_radius=float(np.hypot(maximum_x,maximum_y))
        half_voxel=voxel_size/2.0

        maximum_horizontal_offset=int(np.ceil((footprint_radius+half_voxel)/voxel_size))
        occupancy_min=self.vehicle_min.copy()
        occupancy_min[2]+=0.03
        minimum_vertical_offset=int(np.floor(float(occupancy_min[2])/voxel_size-0.5))
        maximum_vertical_offset=int(np.ceil(float(self.vehicle_max[2])/voxel_size+0.5))
        footprint_offsets=[]
        for dx in range(-maximum_horizontal_offset,maximum_horizontal_offset+1):
            nearest_x=max(abs(dx)*voxel_size-half_voxel,0.0)
            for dy in range(-maximum_horizontal_offset,maximum_horizontal_offset+1):
                nearest_y=max(abs(dy)*voxel_size-half_voxel,0.0)
                if np.hypot(nearest_x,nearest_y)>footprint_radius:
                    continue
                for dz in range(minimum_vertical_offset,maximum_vertical_offset+1):
                    voxel_bottom=(dz-0.5)*voxel_size
                    voxel_top=(dz+0.5)*voxel_size
                    if voxel_top>float(occupancy_min[2]) and voxel_bottom<float(self.vehicle_max[2]):
                        footprint_offsets.append((dx,dy,dz))
        if not footprint_offsets:
            raise RuntimeError("Vehicle collision geometry produced an empty voxel footprint.")

        unknown_x=[voxel[0] for voxel in relevant_unknown]
        unknown_y=[voxel[1] for voxel in relevant_unknown]
        minimum_x=min(min(unknown_x)-max_updates,current_voxel[0]-1)
        maximum_x_index=max(max(unknown_x)+max_updates,current_voxel[0]+1)
        minimum_y=min(min(unknown_y)-max_updates,current_voxel[1]-1)
        maximum_y_index=max(max(unknown_y)+max_updates,current_voxel[1]+1)

        x_count=maximum_x_index-minimum_x+1
        y_count=maximum_y_index-minimum_y+1
        minimum_dx=min(offset[0] for offset in footprint_offsets)
        maximum_dx=max(offset[0] for offset in footprint_offsets)
        minimum_dy=min(offset[1] for offset in footprint_offsets)
        maximum_dy=max(offset[1] for offset in footprint_offsets)
        minimum_dz=min(offset[2] for offset in footprint_offsets)
        maximum_dz=max(offset[2] for offset in footprint_offsets)
        query_minimum_x=minimum_x+minimum_dx
        query_maximum_x=maximum_x_index+maximum_dx
        query_minimum_y=minimum_y+minimum_dy
        query_maximum_y=maximum_y_index+maximum_dy
        query_minimum_z=candidate_z+minimum_dz
        query_maximum_z=candidate_z+maximum_dz
        query_x_count=query_maximum_x-query_minimum_x+1
        query_y_count=query_maximum_y-query_minimum_y+1
        query_z_count=query_maximum_z-query_minimum_z+1
        region_cell_count=query_x_count*query_y_count*query_z_count
        if region_cell_count>self.max_vp_voxels:
            raise RuntimeError(
                f"Vantage search region contains {region_cell_count} cells, reduce max_updates or split the unknown path region.")

        region_indices=[
            (x,y,z)
            for x in range(query_minimum_x,query_maximum_x+1)
            for y in range(query_minimum_y,query_maximum_y+1)
            for z in range(query_minimum_z,query_maximum_z+1)
        ]
        region_observations=self.voxelmodel.states_scores(region_indices)
        persistent_occupied=np.fromiter(
            (observation is not None and observation[0]==1 and observation[1]>=self.riskcal.occupied_confidence_threshold for observation in region_observations),
            dtype=bool,
            count=region_cell_count,
        ).reshape(query_x_count,query_y_count,query_z_count)

        occupied_counts=np.zeros((x_count,y_count),dtype=np.int32)
        for dx,dy,dz in footprint_offsets:
            x_start=minimum_x+dx-query_minimum_x
            y_start=minimum_y+dy-query_minimum_y
            z_index=candidate_z+dz-query_minimum_z
            occupied_counts += persistent_occupied[
                x_start:x_start+x_count,
                y_start:y_start+y_count,
                z_index,
            ]

        fits_vehicle=occupied_counts<self.riskcal.minimum_persistent_occupied_voxels
        current_local=(current_voxel[0]-minimum_x,current_voxel[1]-minimum_y)
        if not fits_vehicle[current_local]:
            return set()

        connectivity=np.array([[0,1,0],[1,1,1],[0,1,0]],dtype=np.uint8)
        connected_regions=np.zeros(fits_vehicle.shape,dtype=np.int32)
        label(fits_vehicle,structure=connectivity,output=connected_regions)
        current_region=connected_regions[current_local]
        reachable=connected_regions==current_region

        reachable_candidates=np.argwhere(fits_vehicle & reachable)
        candidate_voxels=np.column_stack((
            reachable_candidates[:,0]+minimum_x,
            reachable_candidates[:,1]+minimum_y,
            np.full(len(reachable_candidates),candidate_z,dtype=np.int64),
        ))
        unknown_tree=cKDTree(np.asarray(list(relevant_unknown),dtype=np.float64)*voxel_size)
        distance_to_unknown,_=unknown_tree.query(candidate_voxels.astype(np.float64)*voxel_size,k=1)
        observation_radius=max_updates*voxel_size
        candidates=candidate_voxels[distance_to_unknown<=observation_radius]
        return {
            (int(x),int(y),int(z))
            for x,y,z in candidates
        }

    def _camera_callback(self,depth_msg:Image,camera_info:CameraInfo,vehicle_pose:Odometry)->None:
        if rospy.is_shutdown(): return
        try:
            if not depth_msg.header.frame_id or depth_msg.header.frame_id!=camera_info.header.frame_id:
                raise ValueError("CameraInfo and depth image frame IDs do not match.")
            if vehicle_pose.header.frame_id not in (self.navigation_frame,"camera_init") or vehicle_pose.child_frame_id not in ("base_link","body"):
                raise ValueError("Camera voxel observations require the navigation and vehicle body frames.")
            if depth_msg.header.stamp.to_nsec()<=0 or vehicle_pose.header.stamp.to_nsec()<=0 or abs((depth_msg.header.stamp-vehicle_pose.header.stamp).to_sec())>0.05:
                raise ValueError("Camera depth and vehicle pose timestamps must be valid and within 0.05 seconds.")
            depth=self.camera_bridge.imgmsg_to_cv2(depth_msg,desired_encoding="passthrough")
            points=depth_points(depth,depth_msg.encoding,camera_info,pixel_stride=self.cam_stride)
            points_body=points@self.camera_to_body_rotation.T+self.camera_to_body_translation
            points_body=points_body[(points_body[:,2]>=self.vehicle_min[2]-self.height_margin)&(points_body[:,2]<=self.vehicle_max[2]+self.height_margin)]
            # if len(points_body)>300:
            #     points_body=points_body[random.sample(range(len(points_body)),300)]
            pose=vehicle_pose.pose.pose
            translation=np.array([pose.position.x,pose.position.y,pose.position.z],dtype=np.float64)
            quaternion=np.array([pose.orientation.x,pose.orientation.y,pose.orientation.z,pose.orientation.w],dtype=np.float64)
            if not np.all(np.isfinite(translation)) or not np.all(np.isfinite(quaternion)) or np.linalg.norm(quaternion)==0:
                raise ValueError("Camera voxel observations require a finite pose and non-zero quaternion.")
            rotation=quaternion_matrix(quaternion)[:3,:3]
            points_world=points_body@rotation.T+translation
            camera_origin_world=rotation@self.camera_to_body_translation+translation
            voxel_counts=self._raycast_voxels(camera_origin_world,points_world,self.roi_min,self.voxel_size)
            voxel_frame=SpatialObservationFrame()
            voxel_frame.header.stamp=depth_msg.header.stamp
            voxel_frame.header.frame_id=self.navigation_frame
            voxel_frame.source_id="UGV Camera"
            voxel_frame.voxel_size=self.voxel_size
            voxel_frame.grid_origin.x=float(self.roi_min[0])
            voxel_frame.grid_origin.y=float(self.roi_min[1])
            voxel_frame.grid_origin.z=float(self.roi_min[2])
            voxel_frame.sensor_position.x=float(camera_origin_world[0])
            voxel_frame.sensor_position.y=float(camera_origin_world[1])
            voxel_frame.sensor_position.z=float(camera_origin_world[2])
            voxel_frame.voxels=[VoxelObservationMsg(x=voxel[0],y=voxel[1],z=voxel[2],free_observations=counts[0],occupied_observations=counts[1]) for voxel,counts in voxel_counts.items()]
            self._publish_if_active(self.camera_voxel_publisher,voxel_frame)
        except Exception as error:
            rospy.logerr_throttle(5.0,"Camera voxel processing failed: %s",error)

    def _px4_callback(self,msg:PoseStamped):
        self.px4_pose=msg

    def _lidar_callback(self,cloud: PointCloud2,vehicle_pose: Odometry)->None:
        points_body=np.asarray(list(pc2.read_points(cloud,field_names=("x","y","z"),skip_nans=True)),dtype=np.float64).reshape(-1,3)
        points_body=points_body[np.isfinite(points_body).all(axis=1)]
        points_body=points_body[(points_body[:,2]>=self.vehicle_min[2]-self.height_margin)&(points_body[:,2]<=self.vehicle_max[2]+self.height_margin)]
        self.lidar_points_world,self.lidar_origin_world=self._lidar_to_world(points_body,vehicle_pose)

        selected = self.lidar_points_world.tolist()
        if len(selected)>self.num_lid_vox:
            selected = random.sample(selected, k=self.num_lid_vox)

        ground_z=vehicle_pose.pose.pose.position.z+float(self.vehicle_min[2])
        selected=np.asarray(selected,dtype=np.float64).reshape(-1,3)
        voxel_counts=self._raycast_voxels(self.lidar_origin_world,selected,self.roi_min,self.voxel_size,occupied_mask=np.abs(selected[:,2]-ground_z)>0.005)
        voxel_messages=[VoxelObservationMsg(x=voxel[0],y=voxel[1],z=voxel[2],free_observations=counts[0],occupied_observations=counts[1]) for voxel,counts in voxel_counts.items()]
        voxel_frame=SpatialObservationFrame()
        voxel_frame.header.stamp=cloud.header.stamp
        voxel_frame.header.frame_id=self.navigation_frame
        voxel_frame.voxels=voxel_messages
        voxel_frame.voxel_size=self.voxel_size
        voxel_frame.grid_origin.x=self.roi_min[0]
        voxel_frame.grid_origin.y=self.roi_min[1]
        voxel_frame.grid_origin.z=self.roi_min[2]
        voxel_frame.sensor_position.x=float(self.lidar_origin_world[0])
        voxel_frame.sensor_position.y=float(self.lidar_origin_world[1])
        voxel_frame.sensor_position.z=float(self.lidar_origin_world[2])
        voxel_frame.source_id="UGV LiDAR"

        self.voxel_publisher.publish(voxel_frame)

    def _battery_callback(self,battery_msg):
        if rospy.is_shutdown():
            return
        percentage=float(battery_msg.percentage)
        valid_percentage=isfinite(percentage) and 0.0<=percentage<=1.0
        with self.state_lock:
            self.state.data["battery_level"]=percentage if valid_percentage else None
        if not valid_percentage:
            rospy.logwarn_throttle(30.0,"MAVROS battery percentage is unavailable or invalid: %r",percentage)
        self._publish_if_active(self.battery_publisher,battery_msg)

    def _mavros_state_callback(self,state_msg:MavrosState)->None:
        if rospy.is_shutdown():
            return
        with self.state_lock:
            self.state.data["connected"]=bool(state_msg.connected)
            self.state.data["armed"]=bool(state_msg.armed)
            self.state.data["mode"]=str(state_msg.mode)
            self.last_state_monotonic=time.monotonic()

    def _adapter_callback(self,cameracolor,cameradepth,camerainfo):
        if rospy.is_shutdown():
            return
        self._publish_if_active(self.depth_publisher,cameradepth)
        self._publish_if_active(self.color_publisher,cameracolor)
        self._publish_if_active(self.camera_info_publisher,camerainfo)

    def _pose_callback(self,gnss_position):
        if rospy.is_shutdown():
            return
        vehicle_pose=deepcopy(gnss_position)
        if vehicle_pose.header.frame_id=="camera_init":
            vehicle_pose.header.frame_id=self.navigation_frame
        if vehicle_pose.child_frame_id=="body":
            vehicle_pose.child_frame_id="base_link"
        q=vehicle_pose.pose.pose.orientation
        roll,pitch,yaw=euler_from_quaternion([q.x,q.y,q.z,q.w])
        with self.state_lock:
            self.state.data["position"]=Position(x=vehicle_pose.pose.pose.position.x,y=vehicle_pose.pose.pose.position.y,z=vehicle_pose.pose.pose.position.z)
            self.state.data["orientation"]=Orientation(pitch=pitch,yaw=yaw,roll=roll)
            self.vehicle_pose=vehicle_pose
            self.last_pose_monotonic=time.monotonic()
        self._publish_if_active(self.pose_publisher,vehicle_pose)

    def send_occ(self,voxels:list[set[tuple[int,int,int]]])->None:
        points=[]
        padding=1 # Extra horizontal cells for the planner's blocked regions.
        inflated={(x+dx,y+dy,z) for cluster in voxels for x,y,z in cluster for dx in range(-padding,padding+1) for dy in range(-padding,padding+1)}
        with self.voxelmodel._lock:
            frame=self.voxelmodel.get_voxel_map_frame()
            if not frame:
                raise ValueError("Voxel map frame is unavailable.")
            for index in inflated:
                position=self.voxelmodel.voxel_to_position(index)
                points.append((position.x,position.y,position.z))
        message=pc2.create_cloud_xyz32(Header(stamp=rospy.Time.now(),frame_id=frame),points)
        self.obstacle_publisher.publish(message)

    @staticmethod
    def _publish_if_active(publisher,message)->None:
        if rospy.is_shutdown():
            return
        try:
            publisher.publish(message)
        except rospy.exceptions.ROSException as error:
            if "closed topic" not in str(error).lower():
                raise

    def _waypointconverter(self,state:State,msg:PoseStamped,waypoint:PoseStamped)->PoseStamped:
        if self.theta is None:
            q = msg.pose.orientation
            _,_,yawp = euler_from_quaternion([q.x,q.y,q.z,q.w])
            self.theta = (yawp-state.data["orientation"].yaw+np.pi)%(2*np.pi)-np.pi
            self.c,self.s = np.cos(self.theta),np.sin(self.theta)
            xl=state.data["position"].x
            yl=state.data["position"].y
            yp=msg.pose.position.y
            xp=msg.pose.position.x
            self.tx = xp-(self.c*xl-self.s*yl)
            self.ty = yp-(self.s*xl+self.c*yl)
        x,y=waypoint.pose.position.x,waypoint.pose.position.y

        waypoint.pose.position.x=self.c*x-self.s*y+self.tx
        waypoint.pose.position.y=self.s*x+self.c*y+self.ty

        q=waypoint.pose.orientation
        roll,pitch,yaw=euler_from_quaternion([q.x,q.y,q.z,q.w])
        q=quaternion_from_euler(roll,pitch,yaw+self.theta)
        waypoint.pose.orientation=Quaternion(x=float(q[0]),y=float(q[1]),z=float(q[2]),w=float(q[3]))
        
        waypoint.header.frame_id=msg.header.frame_id
        waypoint.header.stamp=rospy.Time.now()
        return waypoint

    def _lidar_to_world(self,points_body: np.ndarray,vehicle_pose: Odometry)->tuple[np.ndarray,np.ndarray]:
        """Transform body-frame LiDAR points and the LiDAR origin into the pose's world frame."""
        points=np.asarray(points_body,dtype=np.float64)
        if points.ndim!=2 or points.shape[1]!=3:
            raise ValueError("points_body must be an Nx3 array of body-frame xyz coordinates.")
        if not np.all(np.isfinite(points)):
            raise ValueError("points_body must contain only finite values.")

        pose=vehicle_pose.pose.pose
        translation=np.array([pose.position.x,pose.position.y,pose.position.z],dtype=np.float64)
        quaternion=np.array([pose.orientation.x,pose.orientation.y,pose.orientation.z,pose.orientation.w],dtype=np.float64)
        if not np.all(np.isfinite(translation)) or not np.all(np.isfinite(quaternion)) or np.linalg.norm(quaternion)==0:
            raise ValueError("vehicle_pose must contain a finite position and a non-zero finite quaternion.")

        rotation=quaternion_matrix(quaternion)[:3,:3]
        points_world=points@rotation.T+translation
        lidar_origin_world=rotation@self.lidar_to_body_translation+translation
        return points_world,lidar_origin_world

    def _raycast_voxels(self,origin_world: np.ndarray,endpoint_world: np.ndarray,grid_origin: np.ndarray,voxel_size: float,occupied_mask:np.ndarray|None=None)->Any:
        """Trace every endpoint; batches return voxel -> [free, occupied] counts, single rays retain their list/endpoint result."""
        origin=np.asarray(origin_world,dtype=np.float64)
        endpoint=np.asarray(endpoint_world,dtype=np.float64)
        grid_origin=np.asarray(grid_origin,dtype=np.float64)

        single=endpoint.shape==(3,)
        if origin.shape!=(3,) or grid_origin.shape!=(3,) or not (single or endpoint.ndim==2 and endpoint.shape[1]==3):
            raise ValueError("Origins must have shape (3,); endpoints must have shape (3,) or (N,3).")
        if not np.all(np.isfinite(origin)) or not np.all(np.isfinite(endpoint)) or not np.all(np.isfinite(grid_origin)):
            raise ValueError("Ray coordinates must be finite")
        if not np.isfinite(voxel_size) or voxel_size<=0:
            raise ValueError("voxel_size must be finite and positive")

        endpoint=endpoint.reshape(-1,3)
        if occupied_mask is None:
            occupied_mask=np.ones(len(endpoint),dtype=bool)
        occupied_mask=np.asarray(occupied_mask,dtype=bool)
        if occupied_mask.shape!=(len(endpoint),):
            raise ValueError("occupied_mask must have one entry per endpoint.")
        if not len(endpoint):
            return {}
        start=np.floor((origin-grid_origin)/voxel_size).astype(np.int64)
        targets=np.floor((endpoint-grid_origin)/voxel_size).astype(np.int64)
        batch_size=len(endpoint)
        origin_t=torch.as_tensor(origin,dtype=torch.float64,device=self.device)
        grid_t=torch.as_tensor(grid_origin,dtype=torch.float64,device=self.device)
        start_t=torch.as_tensor(start,dtype=torch.int64,device=self.device)
        endpoints_t=torch.as_tensor(endpoint,dtype=torch.float64,device=self.device)
        targets_t=torch.as_tensor(targets,dtype=torch.int64,device=self.device)
        occupied_t=torch.as_tensor(occupied_mask,dtype=torch.bool,device=self.device)
        batches=[]
        for offset in range(0,len(endpoint),batch_size):
            target=targets_t[offset:offset+batch_size]
            direction=endpoints_t[offset:offset+batch_size]-origin_t
            crossings=[torch.zeros((len(target),1),dtype=torch.float64,device=self.device),torch.ones((len(target),1),dtype=torch.float64,device=self.device)]
            for axis in range(3):
                lower=torch.minimum(start_t[axis],target[:,axis])
                upper=torch.maximum(start_t[axis],target[:,axis])
                plane_count=int(np.abs(targets[offset:offset+batch_size,axis]-start[axis]).max())
                planes=lower[:,None]+1+torch.arange(plane_count,device=self.device)[None,:]
                boundaries=grid_t[axis]+planes.to(torch.float64)*voxel_size
                delta=direction[:,axis,None]
                times=(boundaries-origin_t[axis])/torch.where(delta!=0,delta,1.0)
                valid=(delta!=0)&(planes<=upper[:,None])&(times>0)&(times<1)
                crossings.append(torch.where(valid,times,torch.ones_like(times)))
            times=torch.cat(crossings,dim=1).sort(dim=1).values
            valid=times[:,1:]-times[:,:-1]>1e-12
            midpoints=(times[:,1:]+times[:,:-1])*0.5
            cells=torch.floor((origin_t+midpoints[:,:,None]*direction[:,None,:]-grid_t)/voxel_size).long()
            cells=torch.where(direction[:,None,:]==0,start_t,cells)
            valid=valid&torch.any(cells!=target[:,None,:],dim=2)
            if single:
                free=torch.cat((start_t[None,:],cells[valid]),dim=0)
                free=torch.unique_consecutive(free[torch.any(free!=target[0],dim=1)],dim=0)
                return [tuple(voxel) for voxel in free.cpu().tolist()],tuple(target[0].cpu().tolist())
            owners=torch.arange(len(target),device=self.device)
            ray_ids=owners[:,None].expand_as(valid)
            pairs=torch.cat((ray_ids[valid][:,None],cells[valid]),dim=1)
            origins=torch.cat((owners[:,None],start_t.expand(len(target),3)),dim=1)
            pairs=torch.cat((pairs,origins[torch.any(target!=start_t,dim=1)]),dim=0)
            pairs=torch.unique(pairs,dim=0)
            free,free_counts=torch.unique(pairs[:,1:],dim=0,return_counts=True)
            occupied,occupied_counts=torch.unique(target[occupied_t[offset:offset+len(target)]],dim=0,return_counts=True)
            batches.append(torch.cat((torch.cat((free,free_counts[:,None],torch.zeros_like(free_counts[:,None])),dim=1),torch.cat((occupied,torch.zeros_like(occupied_counts[:,None]),occupied_counts[:,None]),dim=1)),dim=0))
        observations=torch.cat(batches,dim=0)
        indices,inverse=torch.unique(observations[:,:3],dim=0,return_inverse=True)
        counts=torch.zeros((len(indices),2),dtype=torch.int64,device=self.device)
        counts.index_add_(0,inverse,observations[:,3:])
        return {(x,y,z):[free,occupied] for x,y,z,free,occupied in torch.cat((indices,counts),dim=1).cpu().tolist()}


    def get_state(self):
        with self.state_lock:
            return deepcopy(self.state)

    def _require_vehicle_ready(self,action_name:str)->None:
        if rospy.is_shutdown():
            raise NavigationReadinessError(f"Cannot {action_name}: ROS is shutting down.")
        now=time.monotonic()
        with self.state_lock:
            state=deepcopy(self.state)
            last_state_monotonic=self.last_state_monotonic
            last_pose_monotonic=self.last_pose_monotonic
        if last_state_monotonic is None or now-last_state_monotonic>self.max_state_age:
            raise NavigationReadinessError(f"Cannot {action_name}: MAVROS vehicle state is unavailable or stale.")
        if not state.data["connected"]:
            raise NavigationReadinessError(f"Cannot {action_name}: MAVROS is not connected to the flight controller.")
        if not state.data["armed"]:
            raise NavigationReadinessError(f"Cannot {action_name}: vehicle is not armed.")
        if state.data["mode"]!="OFFBOARD":
            raise NavigationReadinessError(f"Cannot {action_name}: vehicle is in {state.data['mode'] or 'an unknown mode'}, not OFFBOARD.")
    def require_navigation_ready(self,action_name:str)->None:
        self._require_vehicle_ready(action_name)

    def get_collision_geometry(self)->tuple[Odometry | None,np.ndarray,np.ndarray]:
        with self.state_lock:
            return deepcopy(self.vehicle_pose),self.vehicle_min.copy(),self.vehicle_max.copy()

    def plan_path(self,destination:Position,orientation:Orientation,frame_id:str | None=None,tolerance:float=0.0)->Path:
        """Plan a path for RCM to evaluate."""
        if self.make_plan_service is None:
            raise RuntimeError("move_base is not initialized yet.")
        goal_values=np.array([destination.x,destination.y,destination.z,orientation.roll,orientation.pitch,orientation.yaw],dtype=np.float64)
        if not np.all(np.isfinite(goal_values)):
            raise ValueError("Planning position and orientation must contain only finite values.")
        frame_id=self.navigation_frame if frame_id is None else frame_id
        if not isinstance(frame_id,str) or not frame_id.strip():
            raise ValueError("Planning frame must be a non-empty string.")
        if not isinstance(tolerance,(int,float)) or isinstance(tolerance,bool) or not isfinite(tolerance) or tolerance<0:
            raise ValueError("Planning tolerance must be a finite, non-negative number.")
        self._require_vehicle_ready("plan navigation")
        with self.state_lock:
            vehicle_pose=deepcopy(self.vehicle_pose)
        if vehicle_pose is None:
            raise RuntimeError("Vehicle pose is unavailable.")
        if vehicle_pose.header.frame_id!=frame_id:
            raise ValueError(f"Planning frame mismatch: vehicle={vehicle_pose.header.frame_id!r}, goal={frame_id!r}.")

        start=PoseStamped()
        start.header.stamp=rospy.Time.now()
        start.header.frame_id=frame_id
        start.pose=deepcopy(vehicle_pose.pose.pose)
        goal=PoseStamped()
        goal.header.stamp=start.header.stamp
        goal.header.frame_id=frame_id
        goal.pose.position.x=destination.x
        goal.pose.position.y=destination.y
        goal.pose.position.z=destination.z
        qx,qy,qz,qw=quaternion_from_euler(orientation.roll,orientation.pitch,orientation.yaw)
        goal.pose.orientation.x=qx
        goal.pose.orientation.y=qy
        goal.pose.orientation.z=qz
        goal.pose.orientation.w=qw

        planning_deadline=time.monotonic()+6.0
        planning_attempts=0
        active_states=(GoalStatus.PENDING,GoalStatus.ACTIVE,GoalStatus.PREEMPTING,GoalStatus.RECALLING)
        while True:
            self._require_vehicle_ready("continue planning navigation")
            try:
                response=self.make_plan_service(GetPlanRequest(start=start,goal=goal,tolerance=float(tolerance)))
                break
            except rospy.ServiceException as error:
                planning_attempts+=1
                action_state=self.client.get_state() if self.client is not None else None
                if action_state not in active_states:
                    raise RuntimeError(f"move_base failed to generate a path: {error}") from error
                if time.monotonic()>=planning_deadline:
                    raise RuntimeError(f"move_base failed to generate a path after {planning_attempts} attempts while waiting for its previous action to become inactive: {error}") from error
                if planning_attempts==1:
                    rospy.logwarn("move_base temporarily rejected /make_plan while finishing its previous action; retrying for up to 6 seconds.")
                rospy.sleep(0.1)
        path=response.plan
        if not path.poses:
            raise RuntimeError("move_base returned an empty path.")
        if path.header.frame_id and path.header.frame_id!=frame_id:
            raise ValueError(f"Planned path uses frame {path.header.frame_id!r}, expected {frame_id!r}.")
        if any(pose.header.frame_id and pose.header.frame_id!=frame_id for pose in path.poses):
            raise ValueError("The planned path contains poses in an unexpected coordinate frame.")
        return path

    def image(self,targpos:Position,yaw_margin: float=np.deg2rad(15))-> None:
        """Rotate towards the desired object"""
        try:
            state=self.get_state()
            target_yaw=np.arctan2(targpos.y-state.data["position"].y,targpos.x-state.data["position"].x)
            error=(target_yaw-state.data["orientation"].yaw+np.pi)%(2*np.pi)-np.pi
            if abs(error)<yaw_margin:
                return
            x,y,z=state.data["position"].x,state.data["position"].y,state.data["position"].z
            yaw=state.data["orientation"].yaw
            d=2.0  # metres
            c,s=np.cos(yaw),np.sin(yaw)

            front=Position(x=x+d*c,y=y+d*s,z=z)
            behind=Position(x=x-d*c,y=y-d*s,z=z)
            left=Position(x=x-d*s,y=y+d*c,z=z)
            right=Position(x=x+d*s,y=y-d*c,z=z)
            arr=[]
            if(error>0):
                arr=[left,behind,right,front]
            if(error<0):
                arr=[right,behind,left,front]
            i=0
            yaw_error=error
            rate=rospy.Rate(100)
            initial_yaw=state.data["orientation"].yaw
            while i<4 and abs(yaw_error)>yaw_margin:
                state=self.get_state()
                if getattr(self,"stopping",False):
                    raise rospy.ROSInterruptException("Scan interrupted by cleanup.")

                orientation=Orientation(roll=self.state.data["orientation"].roll,pitch=self.state.data["orientation"].pitch,yaw=initial_yaw+pi/2)
                orientation.yaw=orientation.yaw%(2*pi)
                bearing=np.arctan2(arr[i].y-state.data["position"].y,arr[i].x-state.data["position"].x)
                error=(bearing-state.data["orientation"].yaw+pi)%(2*pi)-pi
                yaw_error=(target_yaw-state.data["orientation"].yaw+np.pi)%(2*np.pi)-np.pi
                if abs(yaw_error)<=yaw_margin:
                    break
                if abs(error)<=yaw_margin:
                    i+=1
                    if i==4:
                        raise RuntimeError("Imaging rotation went past one full circle without reaching target(impossible).")
                target=PoseStamped()
                target.header.stamp = rospy.Time.now()
                target.pose.position.x=arr[i].x
                target.pose.position.y=arr[i].y
                target.pose.position.z=arr[i].z
                self.whereitisnt=target

                self.moveflag=True
                if rospy.is_shutdown():
                    raise rospy.ROSInterruptException("Rospy shut down")
                rate.sleep()
            self.moveflag=False
            sumofzeroreadings=0
            if self.current_yawrate is not None:
                flag=False
                while flag==False:
                    if(abs(self.current_yawrate)<0.01):
                        sumofzeroreadings+=1
                    else:
                        sumofzeroreadings=0
                    if(sumofzeroreadings>=100):
                        flag=True
                    rospy.sleep(0.01)
            else:
                raise RuntimeError("Yawrate Unavailible")
        finally:
            self.moveflag=False

        
    def scan(self, yaw_margin: float=np.deg2rad(45)) -> None:
        """Scan eight headings 45 degrees apart, waiting for a fresh detection at each heading."""
        marginmax=pi/4
        if not isfinite(yaw_margin):
            raise ValueError("yaw_margin must be finite.")
        if (yaw_margin<=0 or yaw_margin>marginmax):
            raise ValueError(f"yaw_margin must be between 0 and {marginmax} radians.")
        def callback(data):
            nonlocal current_yaw,last_pose_update
            q=data.pose.pose.orientation
            _,_,current_yaw=euler_from_quaternion([q.x,q.y,q.z,q.w])
            last_pose_update=time.monotonic()
        def detection_callback(data):
            nonlocal last_detection_stamp
            last_detection_stamp=data.header.stamp
        current_yaw=None
        last_pose_update=None
        last_detection_stamp=None
        subscriber=rospy.Subscriber("vehicle_pose", Odometry, callback)
        detection_subscriber=rospy.Subscriber("detections",ObservationFrame,detection_callback)
        rate=rospy.Rate(100)
        try:
            initial_pose_deadline=time.monotonic()+2.0
            while current_yaw is None and not rospy.is_shutdown():
                if time.monotonic()>initial_pose_deadline:
                    raise TimeoutError("Timed out waiting for the initial vehicle pose.")
                rospy.sleep(0.01)

            if rospy.is_shutdown():
                raise rospy.ROSInterruptException("Rospy shut down")
            if current_yaw is None:
                raise RuntimeError("Vehicle pose is unavailable.")

            state=self.get_state()
            x,y,z=state.data["position"].x,state.data["position"].y,state.data["position"].z
            yaw=state.data["orientation"].yaw
            d=2.0  # metres
            next_position=Position(x=x+d*np.cos(yaw+pi/2),y=y+d*np.sin(yaw+pi/2),z=z)

            i=0
            while i<8 and not rospy.is_shutdown():
                state=self.get_state()
                if getattr(self,"stopping",False):
                    raise rospy.ROSInterruptException("Scan interrupted by cleanup.")

                bearing=np.arctan2(next_position.y-state.data["position"].y,next_position.x-state.data["position"].x)
                error=(bearing-state.data["orientation"].yaw+pi)%(2*pi)-pi
                if abs(error)<=yaw_margin:
                    self.moveflag=False
                    heading_reached_at=rospy.Time.now()
                    while last_detection_stamp is None or last_detection_stamp<heading_reached_at:
                        if rospy.is_shutdown() or getattr(self,"stopping",False):
                            raise rospy.ROSInterruptException("Scan interrupted while waiting for detection.")
                        rospy.sleep(0.05)
                    i+=1
                    if i==8:
                        break
                    state=self.get_state()
                    next_position=Position(x=state.data["position"].x+d*np.cos(state.data["orientation"].yaw+pi/2),y=state.data["position"].y+d*np.sin(state.data["orientation"].yaw+pi/2),z=state.data["position"].z)

                target=PoseStamped()
                target.header.stamp = rospy.Time.now()
                target.pose.position.x=next_position.x
                target.pose.position.y=next_position.y
                target.pose.position.z=next_position.z
                self.whereitisnt=target

                self.moveflag=True
                if rospy.is_shutdown():
                    raise rospy.ROSInterruptException("Rospy shut down")
                rate.sleep()
            if rospy.is_shutdown():
                raise rospy.ROSInterruptException("Rospy shut down")

        finally:
            subscriber.unregister()
            detection_subscriber.unregister()
            self.moveflag=False

    def navigate(self, destination: Position, orientation: Orientation, path: Path, frame_id: str | None=None,arrival_tolerance:float | None=None) -> None:
        """Move along the assessed path, then turn to the requested yaw."""
        if self.client is None:
            raise RuntimeError("Sensor data must be initialized before navigation.")
        goal_values=np.array([destination.x,destination.y,destination.z,orientation.roll,orientation.pitch,orientation.yaw],dtype=np.float64)
        if not np.all(np.isfinite(goal_values)):
            raise ValueError("Navigation position and orientation must contain only finite values.")
        arrival_tolerance=self.final_wp_marg if arrival_tolerance is None else arrival_tolerance
        if not isfinite(arrival_tolerance) or arrival_tolerance<=0:
            raise ValueError("Arrival tolerance must be finite and positive.")
        frame_id=self.navigation_frame if frame_id is None else frame_id
        if not isinstance(frame_id,str) or not frame_id.strip():
            raise ValueError("Navigation frame must be a non-empty string.")
        self._require_vehicle_ready("navigate")
        if not isinstance(path,Path) or not path.poses:
            raise ValueError("An assessed, non-empty path is required for navigation.")
        if path.header.frame_id and path.header.frame_id!=frame_id:
            raise ValueError(f"Assessed path uses frame {path.header.frame_id!r}, expected {frame_id!r}.")
        if any(pose.header.frame_id and pose.header.frame_id!=frame_id for pose in path.poses):
            raise ValueError("The assessed path contains poses in an unexpected coordinate frame.")
        try:
            try:
                flag=False
                i=0
                if path.poses is not None and self.movement_publisher is not None:
                    rate = rospy.Rate(20)
                    tagret=path.poses[-1]
                    while flag==False and not rospy.is_shutdown() and not getattr(self,"stopping",False):
                        state = self.get_state()
                        pose=path.poses[i]
                        target=PoseStamped()
                        target.header.stamp = rospy.Time.now()
                        target.pose=deepcopy(pose.pose)

                        distance=np.sqrt((state.data["position"].x-tagret.pose.position.x)**2+(state.data["position"].y-tagret.pose.position.y)**2)
                        yaw_error = (orientation.yaw - self.state.data["orientation"].yaw + pi) % (2*pi) - pi

                        q = target.pose.orientation
                        _, _, target_yaw = euler_from_quaternion([q.x,q.y,q.z,q.w])

                        yaw_error_nextwp=(target_yaw - self.state.data["orientation"].yaw + pi) % (2*pi) - pi

                        dist2nextwp=np.sqrt((state.data["position"].x-target.pose.position.x)**2+(state.data["position"].y-target.pose.position.y)**2)

                        if i==len(path.poses)-1:
                            if distance<=arrival_tolerance:
                                flag=True
                            else:
                                scale=max(distance,0.20)/distance
                                target.pose.position.x=state.data["position"].x+(tagret.pose.position.x-state.data["position"].x)*scale
                                target.pose.position.y=state.data["position"].y+(tagret.pose.position.y-state.data["position"].y)*scale
                        self.whereitisnt=target
                        self.moveflag=not flag
                        if dist2nextwp<=self.intermediate_wp_marg:
                            i+=1
                        if i>len(path.poses)-1:
                            i=len(path.poses)-1
                        rate.sleep()
                else:
                    raise RuntimeError("Planned path is empty or movement publisher is unavailable.")
            except rospy.exceptions.ROSException as error:
                raise RuntimeError("Failed to publish the planned path to the movement topic.") from error
            if not flag or rospy.is_shutdown() or getattr(self,"stopping",False):
                raise rospy.ROSInterruptException("Navigation interrupted before the final turn.")
            self.moveflag=False
            position=self.get_state().data["position"]
            self.image(targpos=Position(x=position.x+2.0*np.cos(orientation.yaw),y=position.y+2.0*np.sin(orientation.yaw),z=position.z))
        finally:
            self.moveflag=False

    def hiddenpoints(self,object_id:str,state:State,distance_behind:float=0.5)->list[Position]:
        """Find points behind object on the same plane as the UGV."""
        position=np.array([state.data["position"].x,state.data["position"].y,state.data["position"].z],dtype=np.float64)
        if not isinstance(object_id,str) or not object_id.strip():
            raise ValueError("Object ID must be a non-empty string.")
        if not isinstance(distance_behind,(int,float)) or isinstance(distance_behind,bool) or not isfinite(distance_behind) or distance_behind<=0:
            raise ValueError("distance_behind must be finite and positive.")
        if not np.all(np.isfinite(position)):
            raise ValueError("Vehicle position must contain only finite values.")    
        matches=self.objectmodel.query_world(object_id)
        if not matches:
            raise KeyError(f"Object {object_id!r} does not exist in the current world model.")
        if len(matches)>1:
            raise KeyError("Only one object can be requested at a time.")
        
        object_values=np.array([matches[0].position.x,matches[0].position.y],dtype=np.float64)
        if not np.all(np.isfinite(object_values)):
            raise ValueError(f"Object {object_id!r}'s position is invalid.")
        dir_vector=object_values-position[:2]
        magnitude=float(np.hypot(dir_vector[0],dir_vector[1]))
        if magnitude<=np.finfo(np.float64).eps:
            raise ValueError("Vehicle and object positions overlap in the navigation plane, so a point behind the object cannot be defined.")

        yaw=np.arctan2(dir_vector[1],dir_vector[0])

        translation_vectors=[distance_behind*np.array([np.cos(yaw),np.sin(yaw)])]
        for theta in range(10,91,10):
            for invert in [1,-1]:
                translation_vectors.append(distance_behind*np.array([np.cos(yaw+(invert*np.deg2rad(theta))),np.sin(yaw+(invert*np.deg2rad(theta)))]))
        posnews=[]
        for vector in translation_vectors:
            posnews.append(object_values+vector)

        posnewer=[]
        if self.riskcal is None:
            raise RuntimeError("Hidden-point filtering requires the destination risk settings.")
        voxel_size=self.voxelmodel.get_voxel_size()
        radius=self.hiddenpoint_neighbourhood_radius
        minimum_z=float(position[2]+self.vehicle_min[2]+0.03)
        maximum_z=float(position[2]+self.vehicle_max[2])
        for pos in posnews:
            lower=np.array(self.voxelmodel.world_position_to_voxel(Position(x=float(pos[0]-radius),y=float(pos[1]-radius),z=minimum_z)),dtype=np.int64)
            upper=np.array(self.voxelmodel.world_position_to_voxel(Position(x=float(pos[0]+radius),y=float(pos[1]+radius),z=maximum_z)),dtype=np.int64)
            indices=np.indices(tuple(upper-lower+1)).reshape(3,-1).T+lower
            lower_center=self.voxelmodel.voxel_to_position((int(lower[0]),int(lower[1]),int(lower[2])))
            centers=np.array([lower_center.x,lower_center.y,lower_center.z])+(indices-lower)*voxel_size
            inside=(np.sum((centers[:,:2]-pos)**2,axis=1)<=radius**2)&(centers[:,2]>=minimum_z)&(centers[:,2]<=maximum_z)
            observations=self.voxelmodel.states_scores([(int(x),int(y),int(z)) for x,y,z in indices[inside]])
            occupied_count=sum(observation is not None and observation[0]==1 and observation[1]>=self.riskcal.occupied_confidence_threshold for observation in observations)
            if occupied_count>=self.maximum_nearby_occupied_voxels:
                continue
            posnewer.append(Position(x=float(pos[0]),y=float(pos[1]),z=float(state.data["position"].z)))

        return posnewer

    def hiddenpoint(self,object_id:str,state:State,distance_behind:float=0.5)->Position:
        """Find point behind object on the same plane as the UGV."""
        position=np.array([state.data["position"].x,state.data["position"].y,state.data["position"].z],dtype=np.float64)
        if not isinstance(object_id,str) or not object_id.strip():
            raise ValueError("Object ID must be a non-empty string.")
        if not isinstance(distance_behind,(int,float)) or isinstance(distance_behind,bool) or not isfinite(distance_behind) or distance_behind<=0:
            raise ValueError("distance_behind must be finite and positive.")
        if not np.all(np.isfinite(position)):
            raise ValueError("Vehicle position must contain only finite values.")    
        matches=self.objectmodel.query_world(object_id)
        if not matches:
            raise KeyError(f"Object {object_id!r} does not exist in the current world model.")
        if len(matches)>1:
            raise KeyError("Only one object can be requested at a time.")
        
        object_values=np.array([matches[0].position.x,matches[0].position.y],dtype=np.float64)
        if not np.all(np.isfinite(object_values)):
            raise ValueError(f"Object {object_id!r}'s position is invalid.")
        dir_vector=object_values-position[:2]
        magnitude=float(np.hypot(dir_vector[0],dir_vector[1]))
        if magnitude<=np.finfo(np.float64).eps:
            raise ValueError("Vehicle and object positions overlap in the navigation plane, so a point behind the object cannot be defined.")

        unit_vector=dir_vector/magnitude
        translation_vector=unit_vector*(magnitude+distance_behind)
        posnew=position[:2]+translation_vector
        return Position(x=float(posnew[0]),y=float(posnew[1]),z=float(state.data["position"].z))
