/// @file  clean_detailed_match.cpp
/// @brief Detailed (re-match) ROS node entry point — subscribers, publishers, parameter loading, obstacles callback
#include <umap/rematch.h>

using namespace std::chrono;

// Helper: duration between two steady_clock time_points in milliseconds
static inline float ms_between(const std::chrono::steady_clock::time_point& t0,
                                const std::chrono::steady_clock::time_point& t1) {
    return static_cast<float>(
        duration_cast<microseconds>(t1 - t0).count()) / 1000.0f;
}

rematchNode::rematchNode(ros::NodeHandle &nh) : nh_(nh)
{
    obstacles_sub_ = nh_.subscribe("/model_output/obstacles", 10, &rematchNode::obstacles2Tracker, this);
    marker_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("rematch/visualization/tracked_obstacles", 10);
    text_marker_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("rematch/visualization/text_markers", 10);
    path_pub_ = nh_.advertise<nav_msgs::Path>("rematch/obstacle_paths", 10);
    tracker_details_pub_ = nh_.advertise<umap::TrackerDetailArray>("/rematch/tracker_details", 10);
    statistics_pub_ = nh_.advertise<umap::RematchStatistics>("/rematch_statistics", 10);

    nh_.param("rematch_node/SIMILARITY_THRESHOLD", rematch_params_.SIMILARITY_THRESHOLD, 0.6);
    nh_.param("rematch_node/SIMILARITY_WEIGHT", rematch_params_.SIMILARITY_WEIGHT, 0.5);
    nh_.param("rematch_node/IOU_THRESHOLD", rematch_params_.IOU_THRESHOLD, 0.5);
    nh_.param("rematch_node/IOU_WEIGHT", rematch_params_.IOU_WEIGHT, 0.5);
    nh_.param("rematch_node/DISTANCE_THRESHOLD", rematch_params_.DISTANCE_THRESHOLD, 0.1);
    nh_.param("rematch_node/DISTANCE_WEIGHT", rematch_params_.DISTANCE_WEIGHT, 0.5);
    nh_.param("rematch_node/REMATCH_VIS_FRAME", rematch_params_.REMATCH_VIS_FRAME, std::string("camera_color_optical_frame"));
    // Output frame: frame_id stamped on tracker_details / obstacles consumed downstream.
    // Defaults to VIS_FRAME so old configs remain backward-compatible.
    nh_.param("rematch_node/REMATCH_OUTPUT_FRAME", rematch_params_.REMATCH_OUTPUT_FRAME, std::string(""));
    if (rematch_params_.REMATCH_OUTPUT_FRAME.empty()) {
        rematch_params_.REMATCH_OUTPUT_FRAME = rematch_params_.REMATCH_VIS_FRAME;
    }
    nh_.param("rematch_node/MAX_TRACKER_AGE", rematch_params_.MAX_TRACKER_AGE, 1.0);
    nh_.param("rematch_node/UNMATCHED_PUBLISH_TIMEOUT", rematch_params_.UNMATCHED_PUBLISH_TIMEOUT, 0.5);
    nh_.param("rematch_node/EXPANSION_RATE", rematch_params_.EXPANSION_RATE, 0.3);

    global_tracker_id_ = 0; // Initialize global tracker ID
    ROS_INFO("Rematch node initialized.");
}

rematchNode::~rematchNode()
{
    ROS_INFO("Rematch node shutting down.");
}

/// @brief Callback function. Convert Obstacle message to tracker measurements & features.
/// @param dync_obstacle_msg
void rematchNode::obstacles2Tracker(const umap::Obstacles &dync_obstacle_msg)
{
    // ── Profiling: record callback entry time ─────────────────────────────────
    tp_cb_start_ = std::chrono::steady_clock::now();

    curr_obstacles_raw_ = dync_obstacle_msg;

    ROS_INFO("[Rematch] msg sizes -- box_ids:%zu centroid_x:%zu centroid_y:%zu centroid_z:%zu width:%zu height:%zu thickness:%zu box_w:%zu box_h:%zu features:%zu", 
             dync_obstacle_msg.box_ids.size(), dync_obstacle_msg.centroid_x.size(), dync_obstacle_msg.centroid_y.size(), dync_obstacle_msg.centroid_z.size(),
             dync_obstacle_msg.width.size(), dync_obstacle_msg.height.size(), dync_obstacle_msg.thickness.size(),
             dync_obstacle_msg.box_width.size(), dync_obstacle_msg.box_height.size(), dync_obstacle_msg.features.size());

    // 1. Extract 3D bounding boxes from the message
    std::vector<Eigen::Vector3d> centroid_measurements;
    std::vector<Eigen::Vector3d> dimensions;
    std::vector<Eigen::Vector2d> HWboxes;
    std::vector<std::vector<float>> features;
    std::vector<rematchNode::ObstacleMeasureFeature> curr_obstacles;

    for (size_t i = 0; i < dync_obstacle_msg.box_ids.size(); i++)
    {
        bool index_ok = true;
        if (i >= dync_obstacle_msg.centroid_x.size() || i >= dync_obstacle_msg.centroid_y.size() || i >= dync_obstacle_msg.centroid_z.size()) {
            ROS_ERROR("[Rematch][IdxMismatch] centroid size too short for i=%zu: cx=%zu cy=%zu cz=%zu", i,
                      dync_obstacle_msg.centroid_x.size(), dync_obstacle_msg.centroid_y.size(), dync_obstacle_msg.centroid_z.size());
            index_ok = false;
        }
        if (i >= dync_obstacle_msg.width.size() || i >= dync_obstacle_msg.height.size() || i >= dync_obstacle_msg.thickness.size()) {
            ROS_ERROR("[Rematch][IdxMismatch] dimension size too short for i=%zu: w=%zu h=%zu t=%zu", i,
                      dync_obstacle_msg.width.size(), dync_obstacle_msg.height.size(), dync_obstacle_msg.thickness.size());
            index_ok = false;
        }
        if (i >= dync_obstacle_msg.box_width.size() || i >= dync_obstacle_msg.box_height.size()) {
            ROS_ERROR("[Rematch][IdxMismatch] box2D size too short for i=%zu: bw=%zu bh=%zu", i,
                      dync_obstacle_msg.box_width.size(), dync_obstacle_msg.box_height.size());
            index_ok = false;
        }
        if (i >= dync_obstacle_msg.features.size()) {
            ROS_ERROR("[Rematch][IdxMismatch] features vector shorter than box_ids: i=%zu features.size=%zu", i, dync_obstacle_msg.features.size());
            index_ok = false;
        } else {
            ROS_INFO("[Rematch] det %zu feature length: %zu", i, dync_obstacle_msg.features[i].data.size());
        }

        if (!index_ok) {
            ROS_WARN("[Rematch] Skip det %zu due to index mismatch", i);
            continue;
        }

        // Get centroids
        Eigen::Vector3d centroid_measurement;
        centroid_measurement << dync_obstacle_msg.centroid_x[i], dync_obstacle_msg.centroid_y[i], dync_obstacle_msg.centroid_z[i];
        centroid_measurements.push_back(centroid_measurement);

        // Get dimensions
        Eigen::Vector3d dimension;
        dimension << dync_obstacle_msg.width[i], dync_obstacle_msg.height[i], dync_obstacle_msg.thickness[i];
        dimensions.push_back(dimension);

        // Get HW box
        Eigen::Vector2d box2D;
        box2D << dync_obstacle_msg.box_width[i], dync_obstacle_msg.box_height[i];
        HWboxes.push_back(box2D);

        // Get features
        std::vector<float> feature;
        for (size_t j = 0; j < dync_obstacle_msg.features[i].data.size(); j++)
        {
            feature.push_back(dync_obstacle_msg.features[i].data[j]);
        }
        features.push_back(feature);

        // Create obstacle
        rematchNode::ObstacleMeasureFeature obstacle;
        obstacle.centroid_measurement = centroid_measurement;
        obstacle.dimension = dimension;
        obstacle.box2D = box2D;
        obstacle.feature = feature;
        // Forward upstream confidence; default 1.0 if the array is short
        // (back-compat with old upstream that did not populate the field).
        if (i < dync_obstacle_msg.confidence.size()) {
            obstacle.confidence = dync_obstacle_msg.confidence[i];
        }
        curr_obstacles.push_back(obstacle);

        ROS_INFO_STREAM("\033[1;35m[Rematch] Receive obstacle " << i 
            << " position: [" << centroid_measurement.transpose() 
            << "], size: [" << dimension.transpose() << "]\033[0m");
    }
    curr_centroidMeasurements_ = centroid_measurements;
    curr_sizes_ = dimensions;

    // ── Profiling: parsing done ───────────────────────────────────────────────
    tp_parse_done_ = std::chrono::steady_clock::now();
    stat_num_detections_ = static_cast<int>(curr_obstacles.size());
    stat_num_trackers_   = static_cast<int>(trackers_.size());

    try {
        matchAndUpdate(curr_obstacles);
    } catch (const std::exception& e) {
        ROS_ERROR_STREAM("Exception caught in ## matchAndUpdate ##: " << e.what());
    }


    try {
        publishVisualization();
    } catch (const std::exception& e) {
        ROS_ERROR_STREAM("Exception caught in ## publishVisualization ##: " << e.what());
    }

    // ── Profiling: visualization done ────────────────────────────────────────
    tp_vis_done_ = std::chrono::steady_clock::now();

    try {
        publishTrackerDetails(dync_obstacle_msg.header.stamp.toSec());
    } catch (const std::exception& e) {
        ROS_ERROR_STREAM("Exception caught in ## publishTrackerDetails ##: " << e.what());
    }

    // ── Profiling: assemble and publish statistics ────────────────────────────
    {
        auto tp_end = std::chrono::steady_clock::now();
        umap::RematchStatistics stats;
        stats.header.stamp    = dync_obstacle_msg.header.stamp;
        stats.header.frame_id = "rematch_profiler";

        stats.num_detections    = stat_num_detections_;
        stats.num_trackers      = stat_num_trackers_;
        stats.num_matched       = stat_num_matched_;
        stats.num_unmatched_det = stat_num_unmatched_;

        stats.t_parse_ms       = ms_between(tp_cb_start_,   tp_parse_done_);
        stats.t_cost_matrix_ms = ms_between(tp_parse_done_,  tp_cost_done_);
        stats.t_association_ms = ms_between(tp_cost_done_,   tp_assoc_done_);
        stats.t_update_ms      = ms_between(tp_assoc_done_,  tp_update_done_);
        stats.t_publish_vis_ms = ms_between(tp_update_done_, tp_vis_done_);
        stats.t_publish_det_ms = ms_between(tp_vis_done_,    tp_end);
        stats.t_total_ms       = ms_between(tp_cb_start_,    tp_end);

        publishStatistics(stats);
    }
}
