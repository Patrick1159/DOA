Header header
uint32[] box_ids

# centroid of 3D box
float32[] centroid_x
float32[] centroid_y
float32[] centroid_z

float32[] thickness
float32[] width
float32[] height

# 2D box in camera plan
uint32[] box_x # top-left point list
uint32[] box_y
uint32[] box_width
uint32[] box_height

# mask
sensor_msgs/Image[] mask # 仅保留box内的部分rgb信息
sensor_msgs/Image binary_mask # for pointcloud, only dynamic-obstacles zone is not 0

# Fearture
std_msgs/Float32MultiArray[] features

# History trajectory
nav_msgs/Path[] obstacles_history_pos