# Timestamp and camera optical frame for this observation.
std_msgs/Header header

# Vehicle body pose in the world frame (T_WB).
nav_msgs/Odometry vehicle_pose

# Camera optical pose expressed in the vehicle body frame (T_BC).
geometry_msgs/TransformStamped camera_pose_in_body

# Calibration associated with the source image.
sensor_msgs/CameraInfo camera_info

# All detector outputs produced from the source image.
alpaca_msgs/ObjectDetection[] detections
