from __future__ import annotations

import time
import numpy as np
from typing import Protocol
from dataclasses import dataclass
from math import pi,isfinite
import actionlib
from actionlib_msgs.msg import GoalStatus
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from std_msgs.msg import Float64
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
from mavros_msgs.msg import State as MavrosState
from tf.transformations import euler_from_quaternion,quaternion_from_euler,quaternion_matrix
from . import worldmodel as wm
from . import risk
from .worldmodel import Position,Orientation
from sensor_msgs.msg import Imu
from .risk import PathVoxelAssessment, VantagePoint
import random
from .vatools import VARegistry,Function

@dataclass
class State():
    battery_level: float | None
    position: Position
    orientation: Orientation
    armed: bool = False
    mode: str = ""
    connected: bool = False

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
    def init_sensordata(self) -> None:
        """Begins republishing sensor data."""
        raise NotImplementedError("This method should be implemented by subclasses.")
    @property
    def navigation_frame(self) -> str:
        """Return the coordinate frame used for navigation goals."""
        raise NotImplementedError("This method should be implemented by subclasses.")
    def get_state(self) -> State:
        """Returns the current state of the vehicle as a State Object."""
        raise NotImplementedError("This method should be implemented by subclasses.")
    def require_navigation_ready(self,action_name:str) -> None:
        """Raise when the vehicle is not currently ready for autonomous navigation."""
        raise NotImplementedError("This method should be implemented by subclasses.")
    def get_collision_geometry(self) -> tuple[Odometry | None,np.ndarray,np.ndarray]:
        """Return the latest pose and body-frame collision bounds."""
        raise NotImplementedError("This method should be implemented by subclasses.")
    def plan_path(self, destination: Position, orientation: Orientation, frame_id: str | None=None, tolerance: float=0.0) -> Path:
        """Return a planned path without commanding the vehicle to move."""
        raise NotImplementedError("This method should be implemented by subclasses.")
    def navigate(self, destination: Position, orientation: Orientation, frame_id: str | None=None) -> None:
        """Navigate to a specified destination."""
        raise NotImplementedError("This method should be implemented by subclasses.")
    def scan(self) -> None:
        """Scan the environment and update the world model."""
        raise NotImplementedError("This method should be implemented by subclasses.")
    def vantagepoint(self,pathvoxel:PathVoxelAssessment,max_updates:int=100)->set[tuple[int,int,int]]:
        """Find a mostly known voxel ring around the unknown region."""
        raise NotImplementedError("This method should be implemented by subclasses.")
    def hiddenpoints(self,object_id:str,state:State,distance_behind:float=0.5)->list[Position]:
        """Vehicle specific inspection points for the inspect function."""
        raise NotImplementedError("This method should be implemented by subclasses.")
    def hiddenpoint(self,object_id:str,state:State,distance_behind:float=0.5)->Position:
        """Vehicle specific inspection point for the inspect function."""
        raise NotImplementedError("This method should be implemented by subclasses.")

class UGVAdapter(VehicleAdapter):
    def __init__(self, state: State, worldmodel: wm.ObjectWorldModel, risk:risk.RiskCal | None=None, **kwargs) -> None:
        self.state=state
        self.registry=VARegistry([Function(name="scan",arguments={"yaw_margin":("float",False)}),Function(name="get_state",arguments={}),Function(name="hiddenpoint",arguments={"object_id":("str",True),"state":("State",True),"distance_behind":("float",False)}),Function(name="image",arguments={"targpos":("Position",True),"yaw_margin":("float",False)})])
        self.worldmodel=worldmodel
        self.riskcal=risk
        self.state_lock=Lock()
        self.client=None
        self.make_plan_service=None
        navigation_frame=kwargs.get("navigation_frame","map")
        if not isinstance(navigation_frame,str) or not navigation_frame.strip():
            raise ValueError("Navigation frame must be a non-empty string.")
        self._navigation_frame:str=navigation_frame
        self.voxel_size = 0.05
        self.vehicle_pose=None
        self.last_pose_monotonic:float | None=None
        self.max_pose_age=0.5
        self.last_state_monotonic:float | None=None
        self.max_state_age=2.0
        self.lidar_points_world=None
        self.lidar_origin_world=None
        self.lidar_condition=Condition()
        self.pending_lidar=None
        self.stopping=False
        self.stopped=False
        self.vehicle_min = np.array([-0.15, -0.17, -0.25])
        self.vehicle_max = np.array([0.15, 0.17, 0.0])
        self.search_offset = np.array([0.30, 0.30, 0.05])
        self.roi_min = self.vehicle_min - self.search_offset
        self.roi_max = self.vehicle_max + self.search_offset
        self.ground_z = -0.05
        self.ground_margin = 0.06
        self.N_open = 5
        self.N_dense = 20
        self.lidar_to_body_translation = np.array([-0.011, -0.02329, 0.04412],dtype=np.float64)
        self.lidar_to_body_rotation = np.eye(3,dtype=np.float64)
        self.whereitisnt:PoseStamped|None=None
        self.intermediate_wp_marg=0.4
        self.final_wp_marg=0.15
        self.moveflag=False
        self.theta=None
        self.current_yawrate=None

    @property
    def navigation_frame(self)->str:
        return self._navigation_frame

    def init_sensordata(self) -> None:
        translation=rospy.get_param("/mapping/extrinsic_T",self.lidar_to_body_translation.tolist())
        rotation=rospy.get_param("/mapping/extrinsic_R",self.lidar_to_body_rotation.reshape(-1).tolist())
        self.lidar_to_body_translation=np.asarray(translation,dtype=np.float64)
        self.lidar_to_body_rotation=np.asarray(rotation,dtype=np.float64).reshape(3,3)
        if self.lidar_to_body_translation.shape!=(3,) or not np.all(np.isfinite(self.lidar_to_body_translation)) or not np.all(np.isfinite(self.lidar_to_body_rotation)):
            raise ValueError("FAST-LIO extrinsic translation and rotation must contain finite values.")
        self.client=actionlib.SimpleActionClient("/move_base",MoveBaseAction)

        self.voxel_publisher = rospy.Publisher("voxel_observations",SpatialObservationFrame,queue_size=1)
        self.movement_publisher = rospy.Publisher("/mavros/setpoint_position/local",PoseStamped,queue_size=1)
        self.pose_publisher = rospy.Publisher("vehicle_pose",Odometry,queue_size=1)
        self.camera_info_publisher = rospy.Publisher("camera_info", CameraInfo, queue_size=1)
        self.battery_publisher = rospy.Publisher("battery_level",BatteryState,queue_size=1)
        self.color_publisher = rospy.Publisher("camera_color",Image,queue_size=1)
        self.depth_publisher = rospy.Publisher("camera_depth",Image,queue_size=1)
        self.clutter_publisher = rospy.Publisher("clutter_level",Float64,queue_size=1)
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

    def _imu_callback(self,msg:Imu):
        self.current_yawrate=msg.angular_velocity.z

    def update_target(self):
        if self.last_pose_monotonic is None or self.px4_pose is None:
            return
        state=self.get_state()
        if self.whereitisnt is None:
            self.whereitisnt=PoseStamped()
            self.whereitisnt.header.stamp = rospy.Time.now()
            self.whereitisnt.pose.position.x=state.position.x
            self.whereitisnt.pose.position.y=state.position.y
            self.whereitisnt.pose.position.z=state.position.z

            q=quaternion_from_euler(state.orientation.roll,state.orientation.pitch,state.orientation.yaw)
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
        if self.riskcal is not None:
            center=self.riskcal.center(pathvoxel)
        else:
            raise ValueError("Risk calibration module is None.")
        voxellist:set[tuple[int,int,int]]={center}
        visited:set[tuple[int,int,int]]={center}
        for _ in range(max_updates):
            voxelnew:set[tuple[int,int,int]]=set()
            for voxel in voxellist:
                voxelnew.update((voxel[0]+dx,voxel[1]+dy,voxel[2]) for dx in (-1,0,1) for dy in (-1,0,1) if not(dx==0 and dy==0))
            voxelnew.difference_update(visited)
            if not voxelnew:
                return set()
            voxelstates=self.worldmodel.get_voxel_states(list(voxelnew))
            known_count=sum(state is not None for state in voxelstates)
            if known_count/len(voxelnew)>=0.15:
                return voxelnew
            visited.update(voxelnew)
            voxellist=voxelnew
        raise RuntimeError(f"Could not find a voxel ring with at least 15% known voxels after {max_updates} updates.")

    def _px4_callback(self,msg:PoseStamped):
        self.px4_pose=msg

    def _lidar_callback(self,cloud: PointCloud2,vehicle_pose: Odometry)->None:
        points_body=np.asarray(list(pc2.read_points(cloud,field_names=("x","y","z"),skip_nans=True)),dtype=np.float64).reshape(-1,3)
        points_body=points_body[np.isfinite(points_body).all(axis=1)]
        self.lidar_points_world,self.lidar_origin_world=self._lidar_to_world(points_body,vehicle_pose)

        clutter_points=points_body[np.all((points_body>=self.roi_min) & (points_body<self.roi_max),axis=1)]
        clutter_points=clutter_points[~np.all((clutter_points>=self.vehicle_min) & (clutter_points<=self.vehicle_max),axis=1)]
        clutter_points=clutter_points[np.abs(clutter_points[:,2]-self.ground_z)>=self.ground_margin]
        if len(clutter_points)==0:
            occupied_count=0
        else:
            clutter_voxel_indices=np.floor((clutter_points-self.roi_min)/self.voxel_size).astype(np.int32)
            occupied_count=len(np.unique(clutter_voxel_indices,axis=0))
        if self.N_dense<=self.N_open:
            rospy.logerr("N_dense must be greater than N_open.")
            return
        clutter_level=float(np.clip((occupied_count-self.N_open)/(self.N_dense-self.N_open),0.0,1.0))

        voxel_counts: dict[tuple[int,int,int],list[int]]={}
        selected = self.lidar_points_world.tolist()
        if len(selected)>300:
            selected = random.sample(selected, 300)

        for point in selected:
            free_voxels,occupied_voxel=self._raycast_voxels(self.lidar_origin_world,point,self.roi_min,self.voxel_size)
            for free_voxel in free_voxels:
                counts=voxel_counts.setdefault(free_voxel,[0,0])
                counts[0]+=1
            counts=voxel_counts.setdefault(occupied_voxel,[0,0])
            counts[1]+=1
        voxel_messages=[VoxelObservationMsg(x=voxel[0],y=voxel[1],z=voxel[2],free_observations=counts[0],occupied_observations=counts[1]) for voxel,counts in voxel_counts.items()]
        voxel_frame=SpatialObservationFrame()
        voxel_frame.header.stamp=cloud.header.stamp
        voxel_frame.header.frame_id=self.navigation_frame
        voxel_frame.voxels=voxel_messages
        voxel_frame.voxel_size=self.voxel_size
        voxel_frame.grid_origin.x=self.roi_min[0]
        voxel_frame.grid_origin.y=self.roi_min[1]
        voxel_frame.grid_origin.z=self.roi_min[2]
        voxel_frame.source_id="UGV LiDAR"

        self.voxel_publisher.publish(voxel_frame)
        self.clutter_publisher.publish(Float64(data=clutter_level))

    def _battery_callback(self,battery_msg):
        if rospy.is_shutdown():
            return
        percentage=float(battery_msg.percentage)
        valid_percentage=isfinite(percentage) and 0.0<=percentage<=1.0
        with self.state_lock:
            self.state.battery_level=percentage if valid_percentage else None
        if not valid_percentage:
            rospy.logwarn_throttle(30.0,"MAVROS battery percentage is unavailable or invalid: %r",percentage)
        self._publish_if_active(self.battery_publisher,battery_msg)

    def _mavros_state_callback(self,state_msg:MavrosState)->None:
        if rospy.is_shutdown():
            return
        with self.state_lock:
            self.state.connected=bool(state_msg.connected)
            self.state.armed=bool(state_msg.armed)
            self.state.mode=str(state_msg.mode)
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
            self.state.position=Position(x=vehicle_pose.pose.pose.position.x,y=vehicle_pose.pose.pose.position.y,z=vehicle_pose.pose.pose.position.z)
            self.state.orientation=Orientation(pitch=pitch,yaw=yaw,roll=roll)
            self.vehicle_pose=vehicle_pose
            self.last_pose_monotonic=time.monotonic()
        self._publish_if_active(self.pose_publisher,vehicle_pose)

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
            self.theta = (yawp-state.orientation.yaw+np.pi)%(2*np.pi)-np.pi
            self.c,self.s = np.cos(self.theta),np.sin(self.theta)
            xl=state.position.x
            yl=state.position.y
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

    def _raycast_voxels(self,origin_world: np.ndarray,endpoint_world: np.ndarray,grid_origin: np.ndarray,voxel_size: float)->tuple[list[tuple[int,int,int]],tuple[int,int,int]]:
        """Return free voxels traversed by a ray and its occupied endpoint voxel."""
        origin=np.asarray(origin_world,dtype=np.float64)
        endpoint=np.asarray(endpoint_world,dtype=np.float64)
        grid_origin=np.asarray(grid_origin,dtype=np.float64)

        if origin.shape!=(3,) or endpoint.shape!=(3,) or grid_origin.shape!=(3,):
            raise ValueError("origin_world, endpoint_world, and grid_origin must have shape (3,)")
        if not np.all(np.isfinite(origin)) or not np.all(np.isfinite(endpoint)) or not np.all(np.isfinite(grid_origin)):
            raise ValueError("Ray coordinates must be finite")
        if not np.isfinite(voxel_size) or voxel_size<=0:
            raise ValueError("voxel_size must be finite and positive")

        current=np.floor((origin-grid_origin)/voxel_size).astype(np.int64)
        target=np.floor((endpoint-grid_origin)/voxel_size).astype(np.int64)

        if np.array_equal(current,target):
            target_index=(int(target[0]),int(target[1]),int(target[2]))
            return [],target_index

        direction=endpoint-origin
        step=np.sign(direction).astype(np.int64)
        t_delta=np.full(3,np.inf,dtype=np.float64)
        t_max=np.full(3,np.inf,dtype=np.float64)

        for axis in range(3):
            if direction[axis]==0:
                continue

            t_delta[axis]=voxel_size/abs(direction[axis])

            if step[axis]>0:
                boundary=grid_origin[axis]+(current[axis]+1)*voxel_size
            else:
                boundary=grid_origin[axis]+current[axis]*voxel_size

            t_max[axis]=(boundary-origin[axis])/direction[axis]

        free_voxels=[]
        maximum_steps=int(np.abs(target-current).sum())+3

        while not np.array_equal(current,target):
            free_voxels.append((int(current[0]),int(current[1]),int(current[2])))

            next_crossing=float(np.min(t_max))
            crossed_axes=np.flatnonzero(np.isclose(t_max,next_crossing,rtol=0.0,atol=1e-12))

            for axis in crossed_axes:
                current[axis]+=step[axis]
                t_max[axis]+=t_delta[axis]

            if len(free_voxels)>maximum_steps:
                raise RuntimeError("Voxel ray-casting exceeded its expected step count")

        target_index=(int(target[0]),int(target[1]),int(target[2]))
        return free_voxels,target_index


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
        if not state.connected:
            raise NavigationReadinessError(f"Cannot {action_name}: MAVROS is not connected to the flight controller.")
        if not state.armed:
            raise NavigationReadinessError(f"Cannot {action_name}: vehicle is not armed.")
        if state.mode!="OFFBOARD":
            raise NavigationReadinessError(f"Cannot {action_name}: vehicle is in {state.mode or 'an unknown mode'}, not OFFBOARD.")
        if last_pose_monotonic is None or now-last_pose_monotonic>self.max_pose_age:
            raise NavigationReadinessError(f"Cannot {action_name}: vehicle localization is unavailable or stale.")
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
            target_yaw=np.arctan2(targpos.y-state.position.y,targpos.x-state.position.x)
            error=(target_yaw-state.orientation.yaw+np.pi)%(2*np.pi)-np.pi
            if abs(error)<yaw_margin:
                return
            x,y,z=state.position.x,state.position.y,state.position.z
            yaw=state.orientation.yaw
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
            initial_yaw=state.orientation.yaw
            while i<4 and abs(yaw_error)>yaw_margin:
                state=self.get_state()
                if getattr(self,"stopping",False):
                    raise rospy.ROSInterruptException("Scan interrupted by cleanup.")

                orientation=Orientation(roll=self.state.orientation.roll,pitch=self.state.orientation.pitch,yaw=initial_yaw+pi/2)
                orientation.yaw=orientation.yaw%(2*pi)
                bearing=np.arctan2(arr[i].y-state.position.y,arr[i].x-state.position.x)
                error=(bearing-state.orientation.yaw+pi)%(2*pi)-pi
                yaw_error=(target_yaw-state.orientation.yaw+np.pi)%(2*np.pi)-np.pi
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
            x,y,z=state.position.x,state.position.y,state.position.z
            yaw=state.orientation.yaw
            d=2.0  # metres
            next_position=Position(x=x+d*np.cos(yaw+pi/2),y=y+d*np.sin(yaw+pi/2),z=z)

            i=0
            while i<8 and not rospy.is_shutdown():
                state=self.get_state()
                if getattr(self,"stopping",False):
                    raise rospy.ROSInterruptException("Scan interrupted by cleanup.")

                bearing=np.arctan2(next_position.y-state.position.y,next_position.x-state.position.x)
                error=(bearing-state.orientation.yaw+pi)%(2*pi)-pi
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
                    next_position=Position(x=state.position.x+d*np.cos(state.orientation.yaw+pi/2),y=state.position.y+d*np.sin(state.orientation.yaw+pi/2),z=state.position.z)

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

    def navigate(self, destination: Position, orientation: Orientation, frame_id: str | None=None) -> None:
        """Moves the vehicle."""
        if self.client is None:
            raise RuntimeError("Sensor data must be initialized before navigation.")
        goal_values=np.array([destination.x,destination.y,destination.z,orientation.roll,orientation.pitch,orientation.yaw],dtype=np.float64)
        if not np.all(np.isfinite(goal_values)):
            raise ValueError("Navigation position and orientation must contain only finite values.")
        frame_id=self.navigation_frame if frame_id is None else frame_id
        if not isinstance(frame_id,str) or not frame_id.strip():
            raise ValueError("Navigation frame must be a non-empty string.")
        path=self.plan_path(destination,orientation,frame_id)
        try:
            try:
                flag=False
                i=0
                if path.poses is not None and self.movement_publisher is not None:
                    rate = rospy.Rate(20)
                    while flag==False and not rospy.is_shutdown() and not getattr(self,"stopping",False):
                        state = self.get_state()
                        pose=path.poses[i]
                        target=PoseStamped()
                        target.header.stamp = rospy.Time.now()
                        target.pose=deepcopy(pose.pose)
                        self.whereitisnt=target
                        self.moveflag=True

                        distance=np.sqrt((state.position.x-destination.x)**2+(state.position.y-destination.y)**2)
                        yaw_error = (orientation.yaw - self.state.orientation.yaw + pi) % (2*pi) - pi

                        q = target.pose.orientation
                        _, _, target_yaw = euler_from_quaternion([q.x,q.y,q.z,q.w])

                        yaw_error_nextwp=(target_yaw - self.state.orientation.yaw + pi) % (2*pi) - pi

                        dist2nextwp=np.sqrt((state.position.x-target.pose.position.x)**2+(state.position.y-target.pose.position.y)**2)

                        if i==len(path.poses)-1:
                            if distance<=self.final_wp_marg:
                                flag=True
                        if dist2nextwp<=self.intermediate_wp_marg:
                            i+=1
                        if i>len(path.poses)-1:
                            i=len(path.poses)-1
                        rate.sleep()
                else:
                    raise RuntimeError("Planned path is empty or movement publisher is unavailable.")
            except rospy.exceptions.ROSException as error:
                raise RuntimeError("Failed to publish the planned path to the movement topic.") from error
        finally:
            self.moveflag=False

    def hiddenpoints(self,object_id:str,state:State,distance_behind:float=0.5)->list[Position]:
        """Find points behind object on the same plane as the UGV."""
        position=np.array([state.position.x,state.position.y,state.position.z],dtype=np.float64)
        if not isinstance(object_id,str) or not object_id.strip():
            raise ValueError("Object ID must be a non-empty string.")
        if not isinstance(distance_behind,(int,float)) or isinstance(distance_behind,bool) or not isfinite(distance_behind) or distance_behind<=0:
            raise ValueError("distance_behind must be finite and positive.")
        if not np.all(np.isfinite(position)):
            raise ValueError("Vehicle position must contain only finite values.")    
        matches=self.worldmodel.query_world(object_id)
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
        
        for pos in posnews:
            posnewer.append(Position(x=float(pos[0]),y=float(pos[1]),z=float(state.position.z)))

        return posnewer

    def hiddenpoint(self,object_id:str,state:State,distance_behind:float=0.5)->Position:
        """Find point behind object on the same plane as the UGV."""
        position=np.array([state.position.x,state.position.y,state.position.z],dtype=np.float64)
        if not isinstance(object_id,str) or not object_id.strip():
            raise ValueError("Object ID must be a non-empty string.")
        if not isinstance(distance_behind,(int,float)) or isinstance(distance_behind,bool) or not isfinite(distance_behind) or distance_behind<=0:
            raise ValueError("distance_behind must be finite and positive.")
        if not np.all(np.isfinite(position)):
            raise ValueError("Vehicle position must contain only finite values.")    
        matches=self.worldmodel.query_world(object_id)
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
        return Position(x=float(posnew[0]),y=float(posnew[1]),z=float(state.position.z))
