/// @file rematch.h

#include <umap/KalmanFilter.h>
#include <umap/ObstacleTracker.h>
#include <umap/umapProcess.h>
#include <umap/umapType.h>
#include <umap/TrackerDetailArray.h>
#include <umap/RematchStatistics.h>

#include <Eigen/Dense>
#include <vector>
#include <thread>
#include <chrono>
#include <iostream>
#include <ros/ros.h>
#include <visualization_msgs/MarkerArray.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <std_msgs/ColorRGBA.h>
#include <string>

class rematchNode{
public:
    rematchNode(ros::NodeHandle& nh);
    ~rematchNode();
    
    void obstacles2Tracker(const umap::Obstacles& dync_boxes3D_msg);

    void updateTrackerList(vector<onboardDetector::ObstacleTracker>& trackers, const vector<Eigen::Vector3d>& bbox_sizes, const vector<Eigen::Vector3d>& measurements, double current_time);

    struct ObstacleMeasureFeature {
        Eigen::Vector3d centroid_measurement;
        Eigen::Vector3d dimension;
        Eigen::Vector2d box2D;
        std::vector<float> feature;
    };
private:
    ros::NodeHandle& nh_;
    ros::Subscriber obstacles_sub_;
    ros::Subscriber features_sub_;
    ros::Publisher marker_pub_;
    ros::Publisher text_marker_pub_;
    ros::Publisher path_pub_;
    ros::Publisher tracker_details_pub_;
    ros::Publisher statistics_pub_;   ///< Publishes to /rematch_statistics
    
    rematchParams rematch_params_;

    bool is_initialized_ = false;

    vector<onboardDetector::ObstacleTracker> trackers_;
    vector<Eigen::Vector3d> curr_centroidMeasurements_; // x, y, z
    vector<Eigen::Vector3d> curr_sizes_; // width, height, thickness
    umap::Obstacles curr_obstacles_raw_; // raw 3D boxes from message


    std::vector<std::vector<float>> clip_features_; // Store clip features
    std::vector<int> clip_ids_; // Store 3D clip features

    std::map<int, int> id_mapping_; // Maps tracker ID(global ID) to measurement ID(current frame ID)
    std::map<int, std::vector<int>> invalid_match_mapping_;
    size_t global_tracker_id_;

    std::map<int, std::vector<geometry_msgs::PoseStamped>> tracker_paths_; // Store historical positions for each tracker
    const size_t max_path_length_ = 100; // Maximum number of historical positions to store
    std::set<int> last_tracker_ids_;

    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;

    void matchAndUpdate(const std::vector<rematchNode::ObstacleMeasureFeature>& curr_obstacles);
    std::vector<Eigen::Vector3d> getNextKFrameState(const onboardDetector::ObstacleTracker& tracker, int K);
    std::vector<std::pair<int, int>> association(Eigen::MatrixXd &cost_matrix);
    void publishVisualization();
    void publishTrackerDetails(double current_time);
    void publishStatistics(const umap::RematchStatistics& stats);

    Eigen::MatrixXd getCostMatrix(vector<onboardDetector::ObstacleTracker>& trackers, const vector<ObstacleMeasureFeature>& measurements);

    // ── Per-frame profiling timestamps (set inside obstacles2Tracker) ─────────
    using TimePoint = std::chrono::steady_clock::time_point;
    TimePoint tp_cb_start_;
    TimePoint tp_parse_done_;
    TimePoint tp_cost_done_;
    TimePoint tp_assoc_done_;
    TimePoint tp_update_done_;
    TimePoint tp_vis_done_;

    // Cached counts for statistics message
    int stat_num_detections_  = 0;
    int stat_num_trackers_    = 0;
    int stat_num_matched_     = 0;
    int stat_num_unmatched_   = 0;

    // Helper functions for visualization
    int findTrackerIndexById(int tracker_id) const;
    std::vector<geometry_msgs::Point> createBoundingBoxCorners(float width, float height, float depth) const;
    void addBoundingBoxEdges(visualization_msgs::Marker& marker, const std::vector<geometry_msgs::Point>& corners) const;
    double calculateAverageMovement(const std::vector<Eigen::Vector3d>& history) const;
    visualization_msgs::Marker createObstacleMarker(int tracker_id, int tracker_idx, int measurement_id, const ros::Time& timestamp) const;
    visualization_msgs::Marker createTextMarker(int tracker_id, int tracker_idx, const geometry_msgs::Point& position, const ros::Time& timestamp) const;
    void updateTrackerPath(int tracker_id, const geometry_msgs::Point& position, const std_msgs::Header& header);
    visualization_msgs::Marker createPathMarker(int tracker_id, const ros::Time& timestamp) const;
    void hsvToRgb(float h, float s, float v, float& r, float& g, float& b) const;
    void deleteRemovedTrackerMarkers(const std::set<int>& current_tracker_ids, visualization_msgs::MarkerArray& marker_array, visualization_msgs::MarkerArray& text_marker_array, visualization_msgs::MarkerArray& path_marker_array, const ros::Time& timestamp);
    void cleanupStalePaths();
    bool transformPointBetweenFrames(const Eigen::Vector3d& point_in,
        const std::string& source_frame,
        const std::string& target_frame,
        const ros::Time& stamp,
        Eigen::Vector3d& point_out) const;
    bool transformSizeBetweenFrames(const Eigen::Vector3d& size_in,
        const std::string& source_frame,
        const std::string& target_frame,
        const ros::Time& stamp,
        Eigen::Vector3d& size_out) const;
};

