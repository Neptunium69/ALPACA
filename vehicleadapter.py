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
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import Image, BatteryState,CameraInfo, PointCloud2
from tf.transformations import euler_from_quaternion,quaternion_from_euler
from .worldmodel import Position,Orientation

@dataclass
class State():
    battery_level: float
    position: Position
    orientation: Orientation


class VehicleAdapter(Protocol):
    state: State
    def get_state(self) -> State:
        """Returns the current state of the vehicle as a State Object."""
        raise NotImplementedError("This method should be implemented by subclasses.")
    def observe(self) -> None:
        """Observe the environment and update the world model."""
        raise NotImplementedError("This method should be implemented by subclasses.")
    def inspect(self, object_id: str) -> None:
        """Inspect an object in the world model."""
        raise NotImplementedError("This method should be implemented by subclasses.")
    def navigate(self, destination: Position, orientation: Orientation) -> None:
        """Navigate to a specified destination."""
        raise NotImplementedError("This method should be implemented by subclasses.")
    def scan(self) -> None:
        """Scan the environment and update the world model."""
        raise NotImplementedError("This method should be implemented by subclasses.")
    def sensordata(self) -> None:
        """Collect raw sensor data from the vehicle's sensors."""
        raise NotImplementedError("This method should be implemented by subclasses.")

class UGVAdapter(VehicleAdapter):
    def __init__(self, state: State, *args, **kwargs) -> None:
        self.state=state
        self.state_lock=Lock()
        self.client=None
        self.voxel_size = 0.05
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
        self.pose_publisher = rospy.Publisher("vehicle_pose",Odometry,queue_size=1)
        self.camera_info_publisher = rospy.Publisher("camera_info", CameraInfo, queue_size=1)
        self.battery_publisher = rospy.Publisher("battery_level",BatteryState,queue_size=1)
        self.color_publisher = rospy.Publisher("camera_color",Image,queue_size=1)
        self.depth_publisher = rospy.Publisher("camera_depth",Image,queue_size=1)
        self.clutter_publisher = rospy.Publisher("clutter_level",Float64,queue_size=1)
        if not self.client.wait_for_server(rospy.Duration(secs=5)):
            raise RuntimeError("The move_base action server is unavailable.")

        self.gnss_position_subscriber = rospy.Subscriber("mavros/global_position/local",Odometry,queue_size=1,callback=self._pose_callback)
        self.color_subscriber = message_filters.Subscriber("/camera/color/image_raw", Image, queue_size=1)
        self.depth_subscriber = message_filters.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, queue_size=1)
        self.camera_info_subscriber = message_filters.Subscriber("/camera/aligned_depth_to_color/camera_info",CameraInfo,queue_size=1)
        self.battery_subscriber = rospy.Subscriber("/mavros/battery", BatteryState, queue_size=1,callback=self._battery_callback)
        self.lidar_subscriber = rospy.Subscriber("/cloud_registered_body", PointCloud2, queue_size=1, callback=self._lidar_callback)

        self.camera_synchronizer = message_filters.ApproximateTimeSynchronizer([self.color_subscriber,self.depth_subscriber,self.camera_info_subscriber], queue_size=10, slop=0.05)
        self.camera_synchronizer.registerCallback(self._adapter_callback)

    def _lidar_callback(self, cloud: PointCloud2)-> None:
        points = np.asarray(list(pc2.read_points(cloud, field_names=("x", "y", "z"), skip_nans=True)), dtype=np.float32)

        if len(points) == 0:
            return

        points = points[np.isfinite(points).all(axis=1)]
        points = (self.lidar_to_body_rotation.T @ (points - self.lidar_to_body_translation).T).T

        inside_roi = np.all((points >= self.roi_min) & (points < self.roi_max), axis=1)
        points = points[inside_roi]

        inside_vehicle = np.all((points >= self.vehicle_min) & (points <= self.vehicle_max), axis=1)
        points = points[~inside_vehicle]

        is_ground = np.abs(points[:, 2] - self.ground_z) < self.ground_margin
        points = points[~is_ground]

        if len(points) == 0:
            occupied_count = 0
        else:
            voxel_indices = np.floor((points - self.roi_min) / self.voxel_size).astype(np.int32)
            occupied_voxels = np.unique(voxel_indices, axis=0)
            occupied_count = len(occupied_voxels)


        if self.N_dense<=self.N_open:
            rospy.logerr("N_dense must be greater than N_open.")
            return
        clutter_level = float(np.clip((occupied_count - self.N_open) / (self.N_dense - self.N_open), 0.0, 1.0))
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
        self.pose_publisher.publish(gnss_position)
        

    def get_state(self):
        with self.state_lock:
            return deepcopy(self.state)

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

    def navigate(self, destination: Position, orientation: Orientation) -> None:
        """Navigate Tool: Navigate to a specified destination using the move_base action server."""
        if self.client is None:
            raise RuntimeError("Sensor data must be initialized before navigation.")
        goal=MoveBaseGoal()
        goal.target_pose.header.frame_id="map"
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

