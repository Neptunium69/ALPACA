# Semantic detector output for one object in an ObservationFrame.
string class_name
float32 confidence

# Bounding box in the source image. Minimum coordinates are inclusive and
# maximum coordinates are exclusive.
float32 x_min
float32 y_min
float32 x_max
float32 y_max

# Selected depth measurement in the camera optical frame.
bool has_depth
float32 optical_depth
float32 depth_sample_u
float32 depth_sample_v

# Euclidean distance from the camera optical origin.
float32 camera_range

# Position in ObservationFrame.vehicle_pose.header.frame_id.
bool has_world_position
geometry_msgs/Point world_position
