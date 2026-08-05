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
from threading import Lock
from copy import deepcopy
from nav_msgs.msg import Odometry,Path
from nav_msgs.srv import GetPlan,GetPlanRequest
from geometry_msgs.msg import PoseStamped,TwistStamped
from alpaca_msgs.msg import SpatialObservationFrame, VoxelObservation as VoxelObservationMsg
from sensor_msgs.msg import Image, BatteryState,CameraInfo, PointCloud2
from tf.transformations import euler_from_quaternion,quaternion_from_euler,quaternion_matrix
from . import worldmodel as wm
from . import risk
from .worldmodel import Position,Orientation
from .risk import PathVoxelAssessment, VantagePoint

@dataclass
class State():
    battery_level: float
    position: Position
    orientation: Orientation

@dataclass
class VoxelObservation():
    location: tuple[int,int,int]
    occupied: int
    free: int


class VehicleAdapter(Protocol):
    state: State
    @property
    def navigation_frame(self) -> str:
        """Return the coordinate frame used for navigation goals."""
        raise NotImplementedError("This method should be implemented by subclasses.")
    def get_state(self) -> State:
        """Returns the current state of the vehicle as a State Object."""
        raise NotImplementedError("This method should be implemented by subclasses.")
    def observe(self) -> None:
        """Observe the environment and update the world model."""
        raise NotImplementedError("This method should be implemented by subclasses.")
    def inspect(self, object_id: str) -> None:
        """Inspect an object in the world model."""
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
    def sensordata(self) -> None:
        """Collect raw sensor data from the vehicle's sensors."""
        raise NotImplementedError("This method should be implemented by subclasses.")
    def vantagepoint(self,pathvoxel:PathVoxelAssessment,max_updates:int=100)->set[tuple[int,int,int]]:
        """Find a mostly known voxel ring around the unknown region."""
        raise NotImplementedError("This method should be implemented by subclasses.")

class UGVAdapter(VehicleAdapter):
    def __init__(self, state: State, risk:risk.RiskCal, worldmodel: wm.WorldModel, *args, **kwargs) -> None:
        self.state=state
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
        self.lidar_points_world=None
        self.lidar_origin_world=None
        self.vehicle_min = np.array([-0.15, -0.17, -0.25])
        self.vehicle_max = np.array([0.15, 0.17, 0.0])
        self.search_offset = np.array([0.30, 0.30, 0.05])
        self.roi_min = self.vehicle_min - self.search_offset
        self.roi_max = self.vehicle_max + self.search_offset
        self.ground_z = -0.25
        self.ground_margin = 0.06
        self.N_open = 5
        self.N_dense = 20
        self.lidar_to_body_translation = np.array([-0.011, -0.02329, 0.04412],dtype=np.float64)
        self.lidar_to_body_rotation = np.eye(3,dtype=np.float64)

    @property
    def navigation_frame(self)->str:
        return self._navigation_frame

    def init_sensordata(self) -> None:
        if not rospy.core.is_initialized():
            rospy.init_node("UGVAdapter",anonymous=True)
        translation=rospy.get_param("/mapping/extrinsic_T",self.lidar_to_body_translation.tolist())
        rotation=rospy.get_param("/mapping/extrinsic_R",self.lidar_to_body_rotation.reshape(-1).tolist())
        self.lidar_to_body_translation=np.asarray(translation,dtype=np.float64)
        self.lidar_to_body_rotation=np.asarray(rotation,dtype=np.float64).reshape(3,3)
        if self.lidar_to_body_translation.shape!=(3,) or not np.all(np.isfinite(self.lidar_to_body_translation)) or not np.all(np.isfinite(self.lidar_to_body_rotation)):
            raise ValueError("FAST-LIO extrinsic translation and rotation must contain finite values.")
        self.client=actionlib.SimpleActionClient("/move_base",MoveBaseAction)

        self.voxel_publisher = rospy.Publisher("voxel_observations",SpatialObservationFrame,queue_size=1)
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

        self.gnss_position_subscriber = message_filters.Subscriber("mavros/global_position/local",Odometry,queue_size=10)
        self.gnss_position_subscriber.registerCallback(self._pose_callback)
        self.color_subscriber = message_filters.Subscriber("/camera/color/image_raw", Image, queue_size=1)
        self.depth_subscriber = message_filters.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, queue_size=1)
        self.camera_info_subscriber = message_filters.Subscriber("/camera/aligned_depth_to_color/camera_info",CameraInfo,queue_size=1)
        self.battery_subscriber = rospy.Subscriber("/mavros/battery", BatteryState, queue_size=1,callback=self._battery_callback)
        self.lidar_subscriber = message_filters.Subscriber("/cloud_registered_body",PointCloud2,queue_size=5)

        self.camera_synchronizer = message_filters.ApproximateTimeSynchronizer([self.color_subscriber,self.depth_subscriber,self.camera_info_subscriber], queue_size=10, slop=0.05)
        self.camera_synchronizer.registerCallback(self._adapter_callback)
        self.lidar_synchronizer = message_filters.ApproximateTimeSynchronizer([self.lidar_subscriber,self.gnss_position_subscriber],queue_size=10,slop=0.03)
        self.lidar_synchronizer.registerCallback(self._lidar_callback)

    def vantagepoint(self,pathvoxel:PathVoxelAssessment,max_updates:int=100)->set[tuple[int,int,int]]:
        if not isinstance(max_updates,int) or isinstance(max_updates,bool) or max_updates<1:
            raise ValueError("max_updates must be a positive integer.")
        center=self.riskcal.center(pathvoxel)
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
            if known_count/len(voxelnew)>0.5:
                return voxelnew
            visited.update(voxelnew)
            voxellist=voxelnew
        raise RuntimeError(f"Could not find a voxel ring with more than 50% known voxels after {max_updates} updates.")

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
        for point in self.lidar_points_world:
            free_voxels,occupied_voxel=self._raycast_voxels(self.lidar_origin_world,point,self.roi_min,self.voxel_size)
            for free_voxel in free_voxels:
                counts=voxel_counts.setdefault(free_voxel,[0,0])
                counts[0]+=1
            counts=voxel_counts.setdefault(occupied_voxel,[0,0])
            counts[1]+=1
        voxel_messages=[VoxelObservationMsg(x=voxel[0],y=voxel[1],z=voxel[2],free_observations=counts[0],occupied_observations=counts[1]) for voxel,counts in voxel_counts.items()]
        voxel_frame=SpatialObservationFrame()
        voxel_frame.header.stamp=cloud.header.stamp
        voxel_frame.header.frame_id=vehicle_pose.header.frame_id
        voxel_frame.voxels=voxel_messages
        voxel_frame.voxel_size=self.voxel_size
        voxel_frame.grid_origin.x=self.roi_min[0]
        voxel_frame.grid_origin.y=self.roi_min[1]
        voxel_frame.grid_origin.z=self.roi_min[2]
        voxel_frame.source_id="UGV LiDAR"

        self.voxel_publisher.publish(voxel_frame)
        self.clutter_publisher.publish(Float64(data=clutter_level))

    def _battery_callback(self,battery_msg):
        with self.state_lock:
            self.state.battery_level=battery_msg.percentage
            self.battery_publisher.publish(battery_msg)

    def _adapter_callback(self,cameracolor,cameradepth,camerainfo):
        self.depth_publisher.publish(cameradepth)
        self.color_publisher.publish(cameracolor)
        self.camera_info_publisher.publish(camerainfo)

    def _pose_callback(self,gnss_position):
        q=gnss_position.pose.pose.orientation
        roll,pitch,yaw=euler_from_quaternion([q.x,q.y,q.z,q.w])
        with self.state_lock:
            self.state.position=Position(x=gnss_position.pose.pose.position.x,y=gnss_position.pose.pose.position.y,z=gnss_position.pose.pose.position.z)
            self.state.orientation=Orientation(pitch=pitch,yaw=yaw,roll=roll)
            self.vehicle_pose=deepcopy(gnss_position)
        self.pose_publisher.publish(gnss_position)

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

        try:
            response=self.make_plan_service(GetPlanRequest(start=start,goal=goal,tolerance=float(tolerance)))
        except rospy.ServiceException as error:
            raise RuntimeError(f"move_base failed to generate a path: {error}") from error
        path=response.plan
        if not path.poses:
            raise RuntimeError("move_base returned an empty path.")
        if path.header.frame_id and path.header.frame_id!=frame_id:
            raise ValueError(f"Planned path uses frame {path.header.frame_id!r}, expected {frame_id!r}.")
        if any(pose.header.frame_id and pose.header.frame_id!=frame_id for pose in path.poses):
            raise ValueError("The planned path contains poses in an unexpected coordinate frame.")
        return path

    def scan(self, yaw_margin: float=0.1) -> None:
        """Rotate 360 degrees while collecting sensor data(Data collection is not done in this script)"""
        marginmax=pi/6
        if not isfinite(yaw_margin):
            raise ValueError("yaw_margin must be finite.")
        if (yaw_margin<=0 or yaw_margin>marginmax):
            raise ValueError(f"yaw_margin must be between 0 and {marginmax} radians.")
        def callback(data):
            nonlocal current_yaw,last_pose_update
            q=data.pose.pose.orientation
            _,_,current_yaw=euler_from_quaternion([q.x,q.y,q.z,q.w])
            last_pose_update=time.monotonic()
        current_yaw=None
        last_pose_update=None
        subscriber=rospy.Subscriber("vehicle_pose", Odometry, callback)

        
        def normalize_angle(angle,initial_yaw):
            # This function measures current yaw as a difference between current yaw and initial yaw, and normalizes it to a range of 0 to 2π radians(π/2 is north).
            return (angle - initial_yaw + pi / 2) % (2 * pi)
        yaw_publisher = rospy.Publisher("/mavros/setpoint_velocity/cmd_vel",TwistStamped,queue_size=10,)
        try:
            message=TwistStamped()
            initial_pose_deadline=time.monotonic()+2.0
            while current_yaw is None and not rospy.is_shutdown():
                if time.monotonic()>initial_pose_deadline:
                    raise TimeoutError("Timed out waiting for the initial vehicle pose.")
                rospy.sleep(0.01)

            if rospy.is_shutdown():
                return
            initial_yaw=current_yaw

            flag=0
            initial_range=[(pi/2)+yaw_margin,(pi/2)-yaw_margin]
            scan_deadline=time.monotonic()+20.0

            while not rospy.is_shutdown() and (flag==0 or not(normalize_angle(current_yaw,initial_yaw)<initial_range[0] and normalize_angle(current_yaw,initial_yaw)>initial_range[1])):
                if time.monotonic()>scan_deadline:
                    raise TimeoutError("Scan rotation timed out.")
                if last_pose_update is None or time.monotonic()-last_pose_update>0.5:
                    raise RuntimeError("Vehicle pose stopped updating during scan.")
                yawrate_d=0.5 # set the desired yaw rate to 0.5 rad/s(aka. start rotating)
                if (not(normalize_angle(current_yaw,initial_yaw)<initial_range[0] and normalize_angle(current_yaw,initial_yaw)>initial_range[1])):
                    flag=1
                rospy.sleep(0.1)
                message.twist.angular.z=yawrate_d
                message.header.stamp = rospy.Time.now()
                yaw_publisher.publish(message)
                if rospy.is_shutdown():
                    return

        finally:
            try:
                if not rospy.is_shutdown():
                    stop_message=TwistStamped()
                    stop_message.twist.angular.z=0.0
                    stop_message.header.stamp = rospy.Time.now()
                    yaw_publisher.publish(stop_message)  # Stop rotation
            finally:     
                subscriber.unregister()

    def navigate(self, destination: Position, orientation: Orientation, frame_id: str | None=None) -> None:
        """Navigate Tool: Navigate to a specified destination using the move_base action server."""
        if self.client is None:
            raise RuntimeError("Sensor data must be initialized before navigation.")
        goal_values=np.array([destination.x,destination.y,destination.z,orientation.roll,orientation.pitch,orientation.yaw],dtype=np.float64)
        if not np.all(np.isfinite(goal_values)):
            raise ValueError("Navigation position and orientation must contain only finite values.")
        frame_id=self.navigation_frame if frame_id is None else frame_id
        if not isinstance(frame_id,str) or not frame_id.strip():
            raise ValueError("Navigation frame must be a non-empty string.")
        goal=MoveBaseGoal()
        goal.target_pose.header.frame_id=frame_id
        goal.target_pose.header.stamp=rospy.Time.now()
        goal.target_pose.pose.position.x=destination.x
        goal.target_pose.pose.position.y=destination.y
        goal.target_pose.pose.position.z=destination.z
        qx,qy,qz,qw=quaternion_from_euler(orientation.roll,orientation.pitch,orientation.yaw)
        goal.target_pose.pose.orientation.x=qx
        goal.target_pose.pose.orientation.y=qy
        goal.target_pose.pose.orientation.z=qz
        goal.target_pose.pose.orientation.w=qw
        self.client.send_goal(goal)
        if not self.client.wait_for_result(rospy.Duration(secs=120)):
            self.client.cancel_goal()
            raise TimeoutError("Navigation timed out.")
        status=self.client.get_state()
        if status!=GoalStatus.SUCCEEDED:
            raise RuntimeError(f"Navigation failed with action status {status}.")

    def inspect(self, object_id: str) -> None:
        # Implement UGV-specific inspection logic here
        pass

