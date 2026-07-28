from typing import Protocol
from dataclasses import dataclass
from math import pi,isfinite
import rospy
import message_filters
from threading import Lock
from copy import deepcopy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import Image, BatteryState,CameraInfo
from tf.transformations import euler_from_quaternion
from .worldmodel import Position,Orientation

@dataclass
class State():
    battery_level: float
    position: Position
    orientation: Orientation


class VehicleAdapter(Protocol):
    def __init__(self, state: State, *args, **kwargs) -> None:
        self.state=state
    def get_state(self) -> State:
        """Returns the current state of the vehicle as a State Object."""
        raise NotImplementedError("This method should be implemented by subclasses.")
    def observe(self) -> None:
        """Observe the environment and update the world model."""
        raise NotImplementedError("This method should be implemented by subclasses.")
    def inspect(self, object_id: str) -> None:
        """Inspect an object in the world model."""
        raise NotImplementedError("This method should be implemented by subclasses.")
    def navigate(self, destination: Position) -> None:
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
        super().__init__(state, *args, **kwargs)
        self.state_lock=Lock()

    def init_sensordata(self) -> None:
        rospy.init_node("UGVAdapter",anonymous=True)
        self.pose_publisher = rospy.Publisher("vehicle_pose",Odometry,queue_size=1)
        self.camera_info_publisher = rospy.Publisher("camera_info", CameraInfo, queue_size=1)
        self.battery_publisher = rospy.Publisher("battery_level",BatteryState,queue_size=1)
        self.color_publisher = rospy.Publisher("camera_color",Image,queue_size=1)
        self.depth_publisher = rospy.Publisher("camera_depth",Image,queue_size=1)

        self.gnss_position_subscriber = rospy.Subscriber("mavros/global_position/local",Odometry,queue_size=1,callback=self._pose_callback)
        self.color_subscriber = message_filters.Subscriber("/camera/color/image_raw", Image, queue_size=1)
        self.depth_subscriber = message_filters.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, queue_size=1)
        self.camera_info_subscriber = message_filters.Subscriber("/camera/aligned_depth_to_color/camera_info",CameraInfo,queue_size=1)
        self.battery_subscriber = rospy.Subscriber("/mavros/battery", BatteryState, queue_size=1,callback=self._battery_callback)

        synchronizer = message_filters.ApproximateTimeSynchronizer([self.color_subscriber,self.depth_subscriber,self.camera_info_subscriber], queue_size=10, slop=0.05)
        synchronizer.registerCallback(self._adapter_callback)
        rospy.spin()
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

    

    def inspect(self, object_id: str) -> None:
        # Implement UGV-specific inspection logic here
        pass

    def navigate(self, destination: Position) -> None:
        # Implement UGV-specific navigation logic here
        pass

    def scan(self, yaw_margin: float=0.1) -> None:
        """Rotate 360 degrees while collecting sensor data(Data collection is not done in this script)"""
        marginmax=pi/6
        if not isfinite(yaw_margin):
            raise ValueError("yaw_margin must be finite.")
        if (yaw_margin<=0 or yaw_margin>marginmax):
            raise ValueError(f"yaw_margin must be between 0 and {marginmax} radians.")
        def callback(data):
                nonlocal current_yaw
                q=data.pose.pose.orientation
                _,_,current_yaw=euler_from_quaternion([q.x,q.y,q.z,q.w])
        current_yaw=None
        subscriber=rospy.Subscriber("vehicle_pose", Odometry, callback)

        
        def normalize_angle(angle,initial_yaw):
            # This function measures current yaw as a difference between current yaw and initial yaw, and normalizes it to a range of 0 to 2π radians(π/2 is north).
            return (angle - initial_yaw + pi / 2) % (2 * pi)
        yaw_publisher = rospy.Publisher("/mavros/setpoint_velocity/cmd_vel",TwistStamped,queue_size=10,)
        try:
            message=TwistStamped()
            while current_yaw is None and not rospy.is_shutdown():
                rospy.sleep(0.01)

            if rospy.is_shutdown():
                return
            initial_yaw=current_yaw

            flag=0
            initial_range=[(pi/2)+yaw_margin,(pi/2)-yaw_margin]

            while not rospy.is_shutdown() and (flag==0 or not(normalize_angle(current_yaw,initial_yaw)<initial_range[0] and normalize_angle(current_yaw,initial_yaw)>initial_range[1])):
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
