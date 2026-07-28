from email import message
from typing import Protocol
from dataclasses import dataclass
from math import pi,isfinite
from matplotlib.pylab import angle
import rospy
import message_filters
import json
from std_msgs.msg import Float64,String
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, BatteryState
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
        # Initialize UGV-specific attributes here

    def init_sensordata(self) -> None:
        rospy.init_node("UGVAdapter",anonymous=True)
        self.gnss_position_subscriber = message_filters.Subscriber("mavros/global_position/local",Odometry,queue_size=1)
        self.color_subscriber = message_filters.Subscriber("/camera/color/image_raw", Image, queue_size=1)
        self.depth_subscriber = message_filters.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, queue_size=1)
        self.battery_subscriber = rospy.Subscriber("/mavros/battery", BatteryState, queue_size=1,callback=self._battery_callback)
        self.pose_publisher = rospy.Publisher("vehicle_pose",Odometry,queue_size=1)
        self.battery_publisher = rospy.Publisher("battery_level",BatteryState,queue_size=1)
        self.color_publisher = rospy.Publisher("camera_color",Image,queue_size=1)
        self.depth_publisher = rospy.Publisher("camera_depth",Image,queue_size=1)

        synchronizer = message_filters.ApproximateTimeSynchronizer([self.gnss_position_subscriber,self.color_subscriber,self.depth_subscriber], queue_size=10, slop=0.05)
        synchronizer.registerCallback(self._adapter_callback)
        rospy.spin()
    def _battery_callback(self):
        self.state.battery_level=self.battery_subscriber.percentage
        self.battery_publisher.publish(self.battery_subscriber)

    def _adapter_callback(self,gnss_position,cameracolor,cameradepth):
        q=gnss_position.pose.pose.orientation
        roll,pitch,yaw=euler_from_quaternion([q.x,q.y,q.z,q.w])
        self.state.position=Position(x=gnss_position.pose.pose.position.x,y=gnss_position.pose.pose.position.y,z=gnss_position.pose.pose.position.z)
        self.state.orientation=Orientation(pitch=pitch,yaw=yaw,roll=roll)

        message=json.dumps(self.state)

        self.depth_publisher.publish(cameradepth)
        self.color_publisher.publish(cameracolor)
        self.pose_publisher.publish(gnss_position)
        

    def get_state(self):
        return self.state

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
        publisher = rospy.Publisher("/cmd_vel_yaw",Float64,queue_size=10,)
        try:
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
                message=Float64(yawrate_d)
                publisher.publish(message)
                if rospy.is_shutdown():
                    return

        finally:
            if not rospy.is_shutdown():
                publisher.publish(Float64(0.0))  # Stop rotation        
                subscriber.unregister()
