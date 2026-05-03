/// @file clean_coarse_match.cpp
/// @brief Coarse match ROS node entry point — subscribers, publishers, parameter loading, callbacks, main()

#include <umap/coarse_match.h>
#include <geometry_msgs/PointStamped.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

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

    // 0. Extract 3D bounding boxes from the message and transform each
    //    centroid from the sensor optical frame to COARSE_OUTPUT_FRAME
    //    (typically `odom`). KF state is stored in OUTPUT_FRAME.
    //    Sizes (width/height/thickness) are scalar dimensions, not points,
    //    so they are not transformed — only the orientation of the AABB
    //    in the output frame is implicit (we keep the same axis-aligned
    //    convention; a tighter implementation could convert to OBBs but
    //    that is out of scope for this refactor).
    std::vector<Eigen::Vector3d> measurements;
    std::vector<Eigen::Vector3d> sizes;
    std::vector<sensor_msgs::Image> masks;
    std::vector<size_t> kept_indices;  // original boxes3D_msg index per kept measurement

    for (size_t i = 0; i < boxes3D_msg.box_ids.size(); i++) {
        Eigen::Vector3d measurement_src, measurement_out, size;
        measurement_src << boxes3D_msg.x[i], boxes3D_msg.y[i], boxes3D_msg.z[i];

        if (!transformPointToOutputFrame(measurement_src, source_frame, stamp, measurement_out)) {
            // TF lookup failed — skip this detection entirely. KF will only
            // predict() for matched trackers in this frame, preserving
            // odom-frame state semantics. Already warned inside the helper.
            continue;
        }

        size << boxes3D_msg.width[i], boxes3D_msg.height[i], boxes3D_msg.thickness[i];
        measurements.push_back(measurement_out);
        sizes.push_back(size);
        masks.push_back(boxes3D_msg.mask[i]);
        kept_indices.push_back(i);

        ROS_INFO_STREAM("[Coarse tracker] Box " << i
                        << " src(" << source_frame << "): [" << measurement_src.transpose()
                        << "] → out(" << params_.COARSE_OUTPUT_FRAME << "): ["
                        << measurement_out.transpose() << "], size: [" << size.transpose() << "]");
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

/// @brief Transform a point from `source_frame` to COARSE_OUTPUT_FRAME at
/// `stamp`. Returns false on lookup failure (with a throttled WARN); caller
/// MUST skip the affected measurement in that case (do not write a fallback
/// value into KF state — that would mix frames).
bool trackerNode::transformPointToOutputFrame(const Eigen::Vector3d& in,
                                              const std::string& source_frame,
                                              const ros::Time& stamp,
                                              Eigen::Vector3d& out) const {
    const std::string& target = params_.COARSE_OUTPUT_FRAME;
    if (source_frame.empty() || source_frame == target) {
        out = in;
        return true;
    }
    geometry_msgs::PointStamped p_in, p_out;
    p_in.header.stamp = stamp;
    p_in.header.frame_id = source_frame;
    p_in.point.x = in.x();
    p_in.point.y = in.y();
    p_in.point.z = in.z();
    try {
        // Wait briefly for the transform — covers stamp-vs-now skew.
        tf_buffer_.transform(p_in, p_out, target, ros::Duration(0.05));
    } catch (const tf2::TransformException& ex) {
        ROS_WARN_THROTTLE(2.0,
            "[Coarse tracker] TF lookup %s ← %s @ %.3f failed: %s — skipping measurement",
            target.c_str(), source_frame.c_str(), stamp.toSec(), ex.what());
        return false;
    }
    out << p_out.point.x, p_out.point.y, p_out.point.z;
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