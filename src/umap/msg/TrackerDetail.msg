# TrackerDetail.msg
# Single tracked obstacle information

int32 tracker_id
geometry_msgs/Point position
geometry_msgs/Vector3 velocity
geometry_msgs/Vector3 box_dimensions
float64 last_update_time
float64 age

# Confidence: 1.0 = isDynamic confirmed (passed hysteresis),
#             0.5 = candidate (early-published, motion suspected),
#             0.0 = static / undecided.
float32 confidence
