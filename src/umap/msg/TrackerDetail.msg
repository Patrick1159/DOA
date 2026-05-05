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

# Diagonal of the Kalman Filter covariance matrix at the current post-update
# step (frame OUTPUT_FRAME, typically `odom`). Cross terms (sigma_xy etc.)
# are assumed ≈ 0 because the constant-velocity KF here keeps x/y/z fully
# decoupled (A, H, and initial P all diagonal; updates do not introduce
# cross-coupling).
#
# Downstream (MPC) propagates these into per-step uncertainty using the
# closed-form CV-KF formula
#   sigma_pos(k)^2  = sigma_xx + (k*dt)^2 * sigma_vxvx
# (and analogously for y / vy). When all four fields are 0 (or absent on
# legacy publishers), consumers should skip uncertainty inflation entirely.
float32 sigma2_xx
float32 sigma2_yy
float32 sigma2_vxvx
float32 sigma2_vyvy
