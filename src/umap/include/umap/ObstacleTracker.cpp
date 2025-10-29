/**
 * @file ObstacleTracker.cpp
 * @brief Implementation of the ObstacleTracker class for tracking obstacles using a Kalman Filter.
 *
 * This file contains the implementation of the ObstacleTracker class, which is part of the 
 * onboardDetector namespace. The class is responsible for initializing and updating a Kalman 
 * Filter to track the position and velocity of obstacles in 3D space.
 *
 * Key functionalities:
 * - Initialization of the Kalman Filter with state, measurement, and noise covariance matrices.
 * - Updating the state transition matrix based on a fixed time step.
 * - Tracking obstacles by processing measurements and estimating their states.
 * - Providing the current state of an obstacle, including position and velocity.
 *
 * Dependencies:
 * - Eigen library for matrix operations.
 * - A Kalman Filter implementation (assumed to be encapsulated in the `kf` object).
 *
 * Namespace:
 * - onboardDetector
 */

#include "ObstacleTracker.h"

namespace onboardDetector {

ObstacleTracker::ObstacleTracker() : initialized(false), dt(0.1), track_id(0), lastUpdateTime(0.0) {}

void ObstacleTracker::initializeKalmanFilter() {
    const int state_size = 6;  // [x, y, z, vx, vy, vz]
    const int meas_size = 3;   // [x, y, z]

    // Initialize state vector
    Eigen::MatrixXd states = Eigen::MatrixXd::Zero(state_size, 1);
    
    // State transition matrix
    Eigen::MatrixXd A = Eigen::MatrixXd::Identity(state_size, state_size);
    // Will be updated in updateStateTransitionMatrix()
    
    // Measurement matrix
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(meas_size, state_size);
    H.block(0, 0, 3, 3) = Eigen::MatrixXd::Identity(3, 3);
    
    // Process noise covariance
    Eigen::MatrixXd Q = Eigen::MatrixXd::Identity(state_size, state_size) * 0.01;
    
    // Measurement noise covariance
    Eigen::MatrixXd R = Eigen::MatrixXd::Identity(meas_size, meas_size) * 0.1;
    
    // Initial state covariance
    Eigen::MatrixXd P = Eigen::MatrixXd::Identity(state_size, state_size) * 1.0;
    
    // Input matrix (not used in this implementation)
    Eigen::MatrixXd B = Eigen::MatrixXd::Zero(state_size, 1);

    kf.setup(states, A, B, H, P, Q, R);
}

/// @brief  Update the state transition matrix based on the time step
void ObstacleTracker::updateStateTransitionMatrix() {
    const int state_size = 6;
    Eigen::MatrixXd A = Eigen::MatrixXd::Identity(state_size, state_size);
    
    // Update position based on velocity
    A.block(0, 3, 3, 3) = Eigen::MatrixXd::Identity(3, 3) * dt;
    
    kf.setA(A);
}

void ObstacleTracker::init(const Eigen::Vector3d& initial_pos, const Eigen::Vector3d& bounding_box_size, double timestamp, 
        const int newtracker_id, const std::string& uuid) {
    initializeKalmanFilter();
    
    Eigen::MatrixXd states = Eigen::MatrixXd::Zero(6, 1);
    states.block(0, 0, 3, 1) = initial_pos;
    
    track_id = newtracker_id; // Set the ID for the new tracker
    track_uuid = uuid; // Set the UUID for the new tracker

    lastUpdateTime = timestamp;
    box_dimensions = bounding_box_size; // initialize bounding box size
    history_pos.push_back(initial_pos); // Store position history
    history_vel.push_back(Eigen::Vector3d::Zero()); // Initialize velocity history
    isDynamic = true; // Initialize as dynamic
    initialized = true;
}

void ObstacleTracker::initWithFeature(const Eigen::Vector3d& initial_pos, const Eigen::Vector3d& bounding_box_size, const std::vector<float> feature, double timestamp, const int newtracker_id) {
    initializeKalmanFilter();
    
    Eigen::MatrixXd states = Eigen::MatrixXd::Zero(6, 1);
    states.block(0, 0, 3, 1) = initial_pos;
    
    track_id = newtracker_id; // Set the ID for the new tracker

    obcstacleClipFeatures = feature; // initialize clip feature
    box_dimensions = bounding_box_size; // initialize bounding box size
    lastUpdateTime = timestamp;
    history_pos.push_back(initial_pos);
    initialized = true;
    update_count = 1; // Initialize update count
}


/// @brief Update the Kalman Filter with a new measurement
/// @param measurement 
/// @param timestamp 
void ObstacleTracker::update(const Eigen::Vector3d& measurement, const Eigen::Vector3d& dimensions, double timestamp, int history_size) {   
    if (!initialized) return;

    updateStateTransitionMatrix();

    Eigen::MatrixXd z(3, 1); // Measurement vector
    z << measurement.x(), measurement.y(), measurement.z();

    Eigen::MatrixXd u = Eigen::MatrixXd::Zero(1, 1);  // No control input
    kf.estimate(z, u);

    lastUpdateTime = timestamp;

    // Store position history
    history_pos.push_back(measurement); 
    if (history_pos.size() > history_size) {
        history_pos.erase(history_pos.begin()); // Keep only the last 10 measurements
    }
    
    // Extract velocity and store it in history_vel
    Eigen::Vector3d velocity(
        kf.output(3),
        kf.output(4),
        kf.output(5)
    );
    history_vel.push_back(velocity);
    if (history_vel.size() > history_size) {
        history_vel.erase(history_vel.begin()); // Keep only the last history_size velocities
    }

    box_dimensions << dimensions.x(), dimensions.y(), dimensions.z(); // width, height, thickness
}

/// @brief Update the Kalman Filter with a new measurement
/// @param measurement 
/// @param timestamp 
void ObstacleTracker::updateWithFeature(const Eigen::Vector3d& measurement, const Eigen::Vector3d& dimensions, const std::vector<float> feature, double timestamp, int history_size) {
    if (!initialized) return;

    updateStateTransitionMatrix();

    Eigen::MatrixXd z(3, 1); // Measurement vector
    z << measurement.x(), measurement.y(), measurement.z();

    Eigen::MatrixXd u = Eigen::MatrixXd::Zero(1, 1);  // No control input
    kf.estimate(z, u);

    // TODO: 将每个tracker的feature变成根据观测到的次数进行加权平均
    // Update the feature by averaging the old and new features
    // float alpha = 1.0f / static_cast<float>(update_count + 1); // Weight for the new feature
    // for (size_t i = 0; i < obcstacleClipFeatures.size(); ++i) {
    //     obcstacleClipFeatures[i] = obcstacleClipFeatures[i] * (1.0f - alpha) + feature[i] * alpha;
    // }
    for (size_t i = 0; i < obcstacleClipFeatures.size(); ++i) {
        obcstacleClipFeatures[i] = feature[i];
    }

    lastUpdateTime = timestamp; // Record last update time
    history_pos.push_back(measurement); // Store position history
    if (history_pos.size() > history_size) {
        history_pos.erase(history_pos.begin()); // Keep only the last 10 measurements
    }
    // Extract velocity and store it in history_vel
    Eigen::Vector3d velocity(
        kf.output(3),
        kf.output(4),
        kf.output(5)
    );
    history_vel.push_back(velocity);
    if (history_vel.size() > history_size) {
        history_vel.erase(history_vel.begin()); // Keep only the last history_size velocities
    }

    // TODO: 更新bounding box size！这是一切错误匹配的根源
    box_dimensions << dimensions.x(), dimensions.y(), dimensions.z(); // width, height, thickness

    update_count++;
}


void ObstacleTracker::predictOnly() {
    if (!initialized) return;

    updateStateTransitionMatrix();
    Eigen::MatrixXd u = Eigen::MatrixXd::Zero(1, 1); // No control input
    kf.predict(u);
    // Note: lastUpdateTime is intentionally not changed here; it reflects last observation time.
    // History arrays are not updated for prediction-only steps to keep them as observation history.
}


Obstacle ObstacleTracker::getState() const {
    /// @brief Get the current state of the obstacle, represented in onboardDetector::Obstacle

    Obstacle obstacle;
    obstacle.id = track_id;
    
    // Extract position
    obstacle.position = Eigen::Vector3d(
        kf.output(0),
        kf.output(1),
        kf.output(2)
    );
    
    // Extract velocity
    obstacle.velocity = Eigen::Vector3d(
        kf.output(3),
        kf.output(4),
        kf.output(5)
    );

    obstacle.dimensions = box_dimensions; // Dimensions of the 3D box (width, height, thickness)
    obstacle.history_pos = history_pos; // vector<Eigen::Vector3d>
    return obstacle;
}

}  // namespace onboardDetector
