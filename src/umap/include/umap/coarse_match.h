/// @file tracker_demo.h
/// @brief: Header file for the demo_node that uses the ObstacleTracker class
#include <umap/KalmanFilter.h>
#include <umap/ObstacleTracker.h>
#include <umap/umapType.h>
#include <umap/params.h>
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
#include <array>


class trackerNode{
public:
    trackerNode(ros::NodeHandle& nh);
    ~trackerNode();
    trackerParams params_; // Parameters for the coarse tracker
    
    void boxes3DCallback(const umap::BoundingBoxes3D& boxes3D_msg);
    void boxes2DCallback(const umap::BoundingBoxes2D& boxes2D_msg);

    void updateTrackerList(vector<onboardDetector::ObstacleTracker>& trackers, 
        const vector<Eigen::Vector3d>& bbox_sizes, const vector<Eigen::Vector3d>& measurements, 
        double current_time); // Run at th begining of 

private:
    ros::NodeHandle& nh_;
    ros::Subscriber HW_bbox_sub_;
    ros::Subscriber boxes3D_sub_;
    ros::Subscriber boxes2D_sub_;
    ros::Subscriber features_sub_;

    ros::Publisher marker_pub_;
    ros::Publisher text_marker_pub_;
    ros::Publisher path_pub_;
    ros::Publisher dynamic_box2D_pub_; // Publisher for dynamic 2D bounding boxes
    ros::Publisher dynamic_box3D_pub_; // Publisher for dynamic 3D bounding boxes
    ros::Publisher obstacles_pub_; // Publisher for obstacles message

    bool is_initialized_ = false;
    vector<onboardDetector::ObstacleTracker> trackers_; // Global list of trackers
    vector<Eigen::Vector3d> curr_measurements_; // x, y, z
    vector<Eigen::Vector3d> curr_sizes_; // width, height, thickness
    vector<sensor_msgs::Image> curr_masks_; // Store masks for each bounding box
    umap::BoundingBoxes2D curr_boxes2D_raw_; // raw 2D boxes from message
    umap::BoundingBoxes3D curr_boxes3D_raw_; // raw 3D boxes from message


    std::map<int, int> id_mapping_; // TODO: 改为绝对索引，而非trackers_ 索引到 measurements索引之间的映射
    std::map<int, std::vector<int>> invalid_match_mapping_; // For storing invalid matches (tracker idx -> list of detection idxs)

    int nextAvailableTrackerId = 0; // For generating unique tracker IDs

    std::map<int, std::vector<geometry_msgs::PoseStamped>> tracker_paths_; // Store historical positions for each tracker
    const size_t max_path_length_ = 100; // Maximum number of historical positions to store

    // Last observed data per tracker (to continue publishing when unmatched)
    std::map<int, sensor_msgs::Image> last_mask_by_tracker_;
    std::map<int, std::array<int,4>> last_box2d_by_tracker_; // x,y,width,height
    std::map<int, Eigen::Vector3d> last_size3d_by_tracker_; // width, height, thickness

    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;

    Eigen::MatrixXd getCost_matrix(vector<onboardDetector::ObstacleTracker>& trackers, const vector<Eigen::Vector3d>& measurements, const vector<Eigen::Vector3d>& bbox_sizes, 
        const double iou_threshold, const double dist_threshold,
        const double iou_weight, const double dist_weight);
    std::vector<std::pair<int, int>> association(const Eigen::MatrixXd& cost_matrix);
    void matchAndUpdate(const std::vector<Eigen::Vector3d>& measurements, const std::vector<Eigen::Vector3d>& bbox_sizes);
    void publishVisualization();
    void publishDynamicBoxes2D(const ros::Time& stamp); // New function to publish dynamic boxes

    bool isDynamicTracker(onboardDetector::ObstacleTracker& tracker) const;
    void clipFeatureProcess(const umap::MultiImageFeatures& clip_feature_msg); // Callback for clip features
};
