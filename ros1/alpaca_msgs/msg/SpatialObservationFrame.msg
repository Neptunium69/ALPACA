# Timestamp and world frame for this observation.
std_msgs/Header header

# Sensor or adapter that produced the observation.
string source_id

# World-model voxel resolution in metres.
float32 voxel_size

# World position corresponding to voxel index (0,0,0).
geometry_msgs/Point grid_origin

# LiDAR origin in header.frame_id at header.stamp (not the grid origin).
geometry_msgs/Point sensor_position

# Sensor-neutral voxel observations.
alpaca_msgs/VoxelObservation[] voxels
