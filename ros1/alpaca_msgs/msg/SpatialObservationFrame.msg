# Timestamp and world frame for this observation.
std_msgs/Header header

# Sensor or adapter that produced the observation.
string source_id

# World-model voxel resolution in metres.
float32 voxel_size

# World position corresponding to voxel index (0,0,0).
geometry_msgs/Point grid_origin

# Sensor-neutral voxel observations.
alpaca_msgs/VoxelObservation[] voxels
