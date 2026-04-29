/// @file clean_coarse_match.cpp
/// @brief Coarse match ROS node entry point — subscribers, publishers, parameter loading, callbacks, main()

#include <umap/coarse_match.h>

/// @brief Constructor for the trackerNode class.
/// @param nh 
trackerNode::trackerNode(ros::NodeHandle& nh) : nh_(nh) {
    // Subscribe to hardware bounding boxes
    boxes3D_sub_ = nh_.subscribe("/umap/3D_bounding_boxes", 10, &trackerNode::boxes3DCallback, this);
    boxes2D_sub_ = nh_.subscribe("/umap/2D_bounding_boxes", 10, &trackerNode::boxes2DCallback, this);

    // Publishers for visualization
    marker_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("", 10);
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

    // 0. Extract 3D bounding boxes from the message
    std::vector<Eigen::Vector3d> measurements;
    std::vector<Eigen::Vector3d> sizes; // box dimensions
    std::vector<sensor_msgs::Image> masks; // Store masks for each bounding box

    for (size_t i = 0; i < boxes3D_msg.box_ids.size(); i++) {
        Eigen::Vector3d measurement, size;
        measurement << boxes3D_msg.x[i], boxes3D_msg.y[i], boxes3D_msg.z[i];
        measurements.push_back(measurement);
        
        size << boxes3D_msg.width[i], boxes3D_msg.height[i], boxes3D_msg.thickness[i]; // width, height, thickness
        sizes.push_back(size);

        sensor_msgs::Image mask = boxes3D_msg.mask[i];
        masks.push_back(mask);
        ROS_INFO_STREAM("[Coarse tracker] Box " << i << "position: [" << measurement.transpose() << "], size: [" << size.transpose() << "]");   
    }

    curr_measurements_ = measurements;
    curr_sizes_ = sizes;
    curr_masks_ = masks; // The i-th element corresponds to the mask of the i-th box

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