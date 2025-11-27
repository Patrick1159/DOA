# TrackerDetails.msg
# Message containing detailed information about tracked obstacles

Header header

# Tracker information
int32[] tracker_ids
geometry_msgs/Point[] positions
geometry_msgs/Vector3[] velocities
geometry_msgs/Vector3[] box_dimensions
float64[] last_update_times
float64[] ages