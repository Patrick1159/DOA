
/// @file: ObstacleTracker.h
/// @brief: Header file for ObstacleTracker.cpp

#ifndef OBSTACLE_TRACKER_H
#define OBSTACLE_TRACKER_H

#include <umap/KalmanFilter.h>
#include <Eigen/Dense>
#include <vector>

namespace onboardDetector {

struct Obstacle {
    int id;
    Eigen::Vector3d position;
    Eigen::Vector3d velocity;
    Eigen::Vector3d dimensions; // Dimensions of the 3D box (length, width, height)
    vector<Eigen::Vector3d> history_pos; // Store the history of the obstacle position
    double timestamp;
};

class ObstacleTracker {
private:
    kalman_filter kf;
    bool initialized;
    double dt;  // time step

    double lastUpdateTime; // last time observed, for manage lifecycle
    
    void initializeKalmanFilter();
    void updateStateTransitionMatrix();

public:
    ObstacleTracker();
    void init(const Eigen::Vector3d& initial_pos, const Eigen::Vector3d& bounding_box_size, double timestamp, const int newtracker_id, const std::string& uuid);
    void initWithFeature(const Eigen::Vector3d& initial_pos, const Eigen::Vector3d& bounding_box_size, const std::vector<float> feature, double timestamp, const int newtracker_id);
    void update(const Eigen::Vector3d& measurement, const Eigen::Vector3d& dimensions, double timestamp, int history_size = 10);
    void updateWithFeature(const Eigen::Vector3d& measurement, const Eigen::Vector3d& dimensions, std::vector<float> feature, double timestamp, int history_size = 10);
    // Predict state when no measurement is available (does not change lastUpdateTime)
    void predictOnly();
    Obstacle getState() const; // Obstacle struct is {id + pos + vel + timestamp}, state of a tracker is the obstacle that represented
    bool isInitialized() const { return initialized; }
    double getLastUpdateTime() const { return lastUpdateTime; }

    // Tracker properties
    int track_id;
    std::string track_uuid;

    bool isDynamic;
    // Store the history of whether the tracker is dynamic
    std::vector<bool> isDynamicHistory;
    vector<Eigen::Vector3d> history_pos; // store the history of the tracker
    vector<Eigen::Vector3d> history_vel; // store the history of the tracker
    std::vector<float> obcstacleClipFeatures; // clip feature of obstacle tracker
    Eigen::Vector3d box_dimensions; // Dimensions of the 3D box (width, height, thickness)
    int update_count; // Number of updates received
};

}  // namespace onboardDetector
#endif
