/// @file rematch.h

#include <umap/KalmanFilter.h>
#include <umap/ObstacleTracker.h>
#include <umap/umapProcess.h>
#include <umap/umapType.h>
#include <umap/TrackerDetails.h>

#include <Eigen/Dense>
#include <vector>
#include <thread>
#include <chrono>
#include <iostream>
#include <ros/ros.h>
#include <visualization_msgs/MarkerArray.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>
#include <std_msgs/ColorRGBA.h>

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

    void matchAndUpdate(const std::vector<rematchNode::ObstacleMeasureFeature>& curr_obstacles);
    std::vector<Eigen::Vector3d> getNextKFrameState(const onboardDetector::ObstacleTracker& tracker, int K);
    std::vector<std::pair<int, int>> association(Eigen::MatrixXd &cost_matrix);
    void publishVisualization();
    void publishTrackerDetails(double current_time);

    Eigen::MatrixXd getCostMatrix(vector<onboardDetector::ObstacleTracker>& trackers, const vector<ObstacleMeasureFeature>& measurements);

    // Helper functions for visualization
    int findTrackerIndexById(int tracker_id) const;
    std::vector<geometry_msgs::Point> createBoundingBoxCorners(float width, float height, float depth) const;
    void addBoundingBoxEdges(visualization_msgs::Marker& marker, const std::vector<geometry_msgs::Point>& corners) const;
    double calculateAverageMovement(const std::vector<Eigen::Vector3d>& history) const;
    visualization_msgs::Marker createObstacleMarker(int tracker_id, int tracker_idx, int measurement_id, const ros::Time& timestamp) const;
    visualization_msgs::Marker createTextMarker(int tracker_id, int tracker_idx, const geometry_msgs::Point& position, const ros::Time& timestamp) const;
    void updateTrackerPath(int tracker_idx, const geometry_msgs::Point& position, const std_msgs::Header& header);
    void publishTrackerPath(int tracker_idx, const ros::Time& timestamp);
    void deleteRemovedTrackerMarkers(const std::set<int>& current_tracker_ids, visualization_msgs::MarkerArray& marker_array, visualization_msgs::MarkerArray& text_marker_array, const ros::Time& timestamp);
    void cleanupStalePaths();
};

