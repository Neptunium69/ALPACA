from email import message
from typing import Protocol
from dataclasses import dataclass
from math import pi,isfinite
from matplotlib.pylab import angle
import rospy
from std_msgs.msg import Float64
from .worldmodel import Position

@dataclass
class State():
    battery_level: float
    position: Position


class VehicleAdapter(Protocol):
    def __init__(self, state: State, *args, **kwargs) -> None:
        self.state=state
    def get_state(self) -> State:
        """Returns the current state of the vehicle as a State Object."""
        return self.state
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
        # Will be implemented in the future once UGV is set up properly.
        pass

class UGVAdapter(VehicleAdapter):
    def __init__(self, state: State, *args, **kwargs) -> None:
        super().__init__(state, *args, **kwargs)
        # Initialize UGV-specific attributes here

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
                current_yaw = data.data
        current_yaw=None
        subscriber=rospy.Subscriber("compass", Float64, callback)

        
        def normalize_angle(angle,initial_yaw):
            # This function measures current yaw as a difference between current yaw and initial yaw, and normalizes it to a range of 0 to 2π radians.
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

    def sensordata(self) -> None:
        # Implement UGV-specific sensor data collection logic here
        pass