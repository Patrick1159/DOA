/// @file clean_coarse_match.cpp
/// @brief Coarse match ROS node entry point — subscribers, publishers, parameter loading, callbacks, main()

#include <umap/coarse_match.h>
#include <geometry_msgs/PointStamped.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <limits>

/// @brief Constructor for the trackerNode class.
/// @param nh
trackerNode::trackerNode(ros::NodeHandle& nh) : nh_(nh), tf_listener_(tf_buffer_) {
    // Subscribe to hardware bounding boxes
    boxes3D_sub_ = nh_.subscribe("/umap/3D_bounding_boxes", 10, &trackerNode::boxes3DCallback, this);
    boxes2D_sub_ = nh_.subscribe("/umap/2D_bounding_boxes", 10, &trackerNode::boxes2DCallback, this);

    // Publishers for visualization
    marker_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("tracker/cubes", 10);
    text_marker_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("tracker/text_markers", 10);
    path_pub_ = nh_.advertise<nav_msgs::Path>("tracker/obstacle_paths", 10);
    obstacles_pub_ = nh_.advertise<umap::Obstacles>("tracker/dynamic_obstacles", 10);

    // Params in match module
    nh_.param("tracker_node/DYNAMIC_IOU_THRESHOLD", params_.DYNAMIC_IOU_THRESHOLD, 0.3);
    nh_.param("tracker_node/DYNAMIC_DIST_THRESHOLD", params_.DYNAMIC_DIST_THRESHOLD, 0.5);
    nh_.param("tracker_node/IOU_WEIGHT", params_.iou_weight, 0.5);
    nh_.param("tracker_node/DIST_WEIGHT", params_.dist_weight, 0.5);

    // Params in isDynamic module 
    nh_.param("tracker_node/DYNAMIC_VELOCITY_THRESHOLD", params_.DYNAMIC_VELOCITY_THRESHOLD, 1.0);
    nh_.param("tracker_node/DYNAMIC_VELOCITY_MAX", params_.DYNAMIC_VELOCITY_MAX, 5.0);
    nh_.param("tracker_node/SUSTAINED_VELOCITY_THRESHOLD", params_.SUSTAINED_VELOCITY_THRESHOLD, 0.3);
    nh_.param("tracker_node/SUSTAINED_VELOCITY_COUNT", params_.SUSTAINED_VELOCITY_COUNT, 3);
    nh_.param("tracker_node/AVERAGE_DEPLACEMENT_THRESHOLD", params_.AVERAGE_DEPLACEMENT_THRESHOLD, 0.2);
    nh_.param("tracker_node/HISTORY_SIZE", params_.HISTORY_SIZE, 10); // control the length of history_pos and history_vel
    nh_.param("tracker_node/STATUS_HISTORY_SIZE", params_.STATUS_HISTORY_SIZE, 10); // control the length of status
    nh_.param("tracker_node/MIN_CONSECUTIVE_FRAMES_FOR_STATUS_CHANGE", params_.MIN_CONSECUTIVE_FRAMES_FOR_STATUS_CHANGE, 3); // control the length of status
    nh_.param("tracker_node/MAX_TRACKER_AGE", params_.MAX_TRACKER_AGE, 1.0); // seconds
    nh_.param("tracker_node/UNMATCHED_PUBLISH_TIMEOUT", params_.UNMATCHED_PUBLISH_TIMEOUT, 0.5); // seconds, independent of MAX_TRACKER_AGE
    nh_.param("tracker_node/EXPANSION_RATE", params_.EXPANSION_RATE, 0.3); // per second dimension expansion rate for dynamic trackers
    nh_.param("tracker_node/CANDIDATE_VELOCITY_THRESHOLD", params_.CANDIDATE_VELOCITY_THRESHOLD, 0.2); // candidate early-publish |v| threshold
    nh_.param("tracker_node/MIN_OBSTACLE_Z", params_.MIN_OBSTACLE_Z, 0.1); // ground filter: drop detections with output-frame z below this

    // Visualization param
    nh_.param("tracker_node/COARSE_VIS_FRAME", params_.COARSE_VIS_FRAME, std::string("camera_color_optical_frame"));
    // Output frame: frame_id stamped on obstacles_msg / paths consumed downstream.
    // Defaults to VIS_FRAME so old configs remain backward-compatible.
    nh_.param("tracker_node/COARSE_OUTPUT_FRAME", params_.COARSE_OUTPUT_FRAME, std::string(""));
    if (params_.COARSE_OUTPUT_FRAME.empty()) {
        params_.COARSE_OUTPUT_FRAME = params_.COARSE_VIS_FRAME;
    }
    ROS_INFO("[Tracker] Node initialized");
}

trackerNode::~trackerNode() {
    ROS_INFO("[Tracker] Node shutting down");
}

/// @brief  Main function for processing 3D boxes. Convert 3D bounding boxes to tracker measurements.
/// @param boxes3D_msg
void trackerNode::boxes3DCallback(const umap::BoundingBoxes3D& boxes3D_msg) {
    ROS_DEBUG_NAMED("Coarse tracker", "\n\n========== [Tracker] New Frame ==========");
    curr_boxes3D_raw_ = boxes3D_msg; // Store raw 3D boxes from message

    const std::string source_frame = boxes3D_msg.header.frame_id;
    const ros::Time stamp = boxes3D_msg.header.stamp;

    // 0. Extract 3D bounding boxes from the message and reproject each
    //    AABB (centroid + extents) from the sensor optical frame to
    //    COARSE_OUTPUT_FRAME (typically `odom`). KF state is stored in
    //    OUTPUT_FRAME. The reprojection takes the 8 corners of the source
    //    AABB and re-AABBs them in the output frame, so dimensions become
    //    semantically meaningful in the output frame (width = output-x
    //    extent, etc.) — eliminates rotation-dependent dimension confusion
    //    when the robot turns. Still an AABB (not a true OBB).
    //    Detections whose output-frame z falls below MIN_OBSTACLE_Z are
    //    dropped as ground noise.
    std::vector<Eigen::Vector3d> measurements;
    std::vector<Eigen::Vector3d> sizes;
    std::vector<sensor_msgs::Image> masks;
    std::vector<size_t> kept_indices;  // original boxes3D_msg index per kept measurement

    for (size_t i = 0; i < boxes3D_msg.box_ids.size(); i++) {
        Eigen::Vector3d centroid_src, extents_src, centroid_out, extents_out;
        centroid_src << boxes3D_msg.x[i], boxes3D_msg.y[i], boxes3D_msg.z[i];
        extents_src  << boxes3D_msg.width[i], boxes3D_msg.height[i], boxes3D_msg.thickness[i];

        if (!reprojectAabbToOutputFrame(centroid_src, extents_src, source_frame, stamp,
                                         centroid_out, extents_out)) {
            // TF lookup failed — skip this detection entirely. KF will only
            // predict() for matched trackers in this frame, preserving
            // output-frame state semantics. Already warned inside the helper.
            continue;
        }

        // Ground filter: drop detections sitting on/below the floor.
        // Output frame z=0 is assumed to be the ground plane (true for
        // `odom` in the standard ROS convention).
        if (centroid_out.z() < params_.MIN_OBSTACLE_Z) {
            ROS_DEBUG_NAMED("Coarse tracker",
                "[Coarse tracker] Box %zu skipped: out-frame z=%.3f < MIN_OBSTACLE_Z=%.3f",
                i, centroid_out.z(), params_.MIN_OBSTACLE_Z);
            continue;
        }

        measurements.push_back(centroid_out);
        sizes.push_back(extents_out);
        masks.push_back(boxes3D_msg.mask[i]);
        kept_indices.push_back(i);

        ROS_INFO_STREAM("[Coarse tracker] Box " << i
                        << " src(" << source_frame << "): centroid [" << centroid_src.transpose()
                        << "] extents [" << extents_src.transpose() << "]"
                        << " → out(" << params_.COARSE_OUTPUT_FRAME << "): centroid ["
                        << centroid_out.transpose() << "] extents [" << extents_out.transpose() << "]");
    }

    // Rebuild curr_boxes2D_raw_ alignment: downstream visualization indexes
    // into curr_sizes_/curr_masks_ by the matched detection index, which now
    // refers to the kept-index space. Reindex curr_boxes2D_raw_ accordingly.
    {
        umap::BoundingBoxes2D b2d_kept;
        b2d_kept.header = curr_boxes2D_raw_.header;
        for (size_t k : kept_indices) {
            if (k < curr_boxes2D_raw_.x.size()) {
                b2d_kept.x.push_back(curr_boxes2D_raw_.x[k]);
                b2d_kept.y.push_back(curr_boxes2D_raw_.y[k]);
                b2d_kept.width.push_back(curr_boxes2D_raw_.width[k]);
                b2d_kept.height.push_back(curr_boxes2D_raw_.height[k]);
            }
        }
        curr_boxes2D_raw_ = b2d_kept;
    }

    curr_measurements_ = measurements;
    curr_sizes_ = sizes;
    curr_masks_ = masks;

    // 1. Perform matching and updating
    matchAndUpdate(measurements, sizes);

    for (size_t i = 0; i < trackers_.size(); ++i) {
        auto state = trackers_[i].getState();
        ROS_DEBUG_STREAM_NAMED("Coarse tracker", "Tracker " << i
            << " | ID: " << state.id
            << " | Position: [" << state.position.transpose() << "]"
            << " | Velocity: [" << state.velocity.transpose() << "]");
    }

    // 2. Publish visualization
    publishVisualization();

    ROS_DEBUG_NAMED("Coarse tracker", "=================Frame end================");
}

/// @brief Reproject a source-frame AABB to OUTPUT_FRAME by transforming all
/// 8 corners and re-AABBing them. Dimensions in the output frame become
/// semantically meaningful in that frame. Returns false on TF lookup
/// failure (with a throttled WARN); caller MUST skip the affected detection
/// in that case (do not write a fallback value into KF state — that would
/// mix frames).
bool trackerNode::reprojectAabbToOutputFrame(const Eigen::Vector3d& centroid_src,
                                              const Eigen::Vector3d& extents_src,
                                              const std::string& source_frame,
                                              const ros::Time& stamp,
                                              Eigen::Vector3d& centroid_out,
                                              Eigen::Vector3d& extents_out) const {
    const std::string& target = params_.COARSE_OUTPUT_FRAME;

    // Fast path: same frame, no transform needed.
    if (source_frame.empty() || source_frame == target) {
        centroid_out = centroid_src;
        extents_out = extents_src;
        return true;
    }

    // Look up the transform once for all 8 corners; re-using a single
    // transform avoids 8 TF lookups and ensures all corners share the same
    // rotation/translation snapshot.
    geometry_msgs::TransformStamped tf_src_to_out;
    try {
        tf_src_to_out = tf_buffer_.lookupTransform(target, source_frame, stamp,
                                                    ros::Duration(0.05));
    } catch (const tf2::TransformException& ex) {
        ROS_WARN_THROTTLE(2.0,
            "[Coarse tracker] AABB reproject TF %s ← %s @ %.3f failed: %s — skipping",
            target.c_str(), source_frame.c_str(), stamp.toSec(), ex.what());
        return false;
    }

    const Eigen::Vector3d half = extents_src * 0.5;
    Eigen::Vector3d out_min(std::numeric_limits<double>::infinity(),
                             std::numeric_limits<double>::infinity(),
                             std::numeric_limits<double>::infinity());
    Eigen::Vector3d out_max = -out_min;

    for (int sx : {-1, 1}) {
        for (int sy : {-1, 1}) {
            for (int sz : {-1, 1}) {
                geometry_msgs::PointStamped corner_in, corner_out;
                corner_in.header.stamp = stamp;
                corner_in.header.frame_id = source_frame;
                corner_in.point.x = centroid_src.x() + sx * half.x();
                corner_in.point.y = centroid_src.y() + sy * half.y();
                corner_in.point.z = centroid_src.z() + sz * half.z();
                tf2::doTransform(corner_in, corner_out, tf_src_to_out);
                out_min.x() = std::min(out_min.x(), corner_out.point.x);
                out_min.y() = std::min(out_min.y(), corner_out.point.y);
                out_min.z() = std::min(out_min.z(), corner_out.point.z);
                out_max.x() = std::max(out_max.x(), corner_out.point.x);
                out_max.y() = std::max(out_max.y(), corner_out.point.y);
                out_max.z() = std::max(out_max.z(), corner_out.point.z);
            }
        }
    }

    centroid_out = 0.5 * (out_min + out_max);
    extents_out  = out_max - out_min;
    return true;
}

/// @brief Save 2D bounding boxes from the message.
/// @param tracker 
void trackerNode::boxes2DCallback(const umap::BoundingBoxes2D& boxes2D_msg) {
    curr_boxes2D_raw_ = boxes2D_msg;
}


int main(int argc, char** argv) {
    ros::init(argc, argv, "coarse_match_node");
    ros::NodeHandle nh;
    trackerNode tracker(nh);
    ROS_INFO("[Coarse Match] Node is running");

    ros::spin();

    return 0;
}