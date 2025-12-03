/// @file  rematch.cpp
/// @brief Second filter, to rematch the dynamic obstacles
#include <umap/rematch.h>
#include <munkres/munkres.h>
#include <numeric>           // For std::inner_product

using namespace std::chrono;

rematchNode::rematchNode(ros::NodeHandle &nh) : nh_(nh)
{
    obstacles_sub_ = nh_.subscribe("/model_output/obstacles", 10, &rematchNode::obstacles2Tracker, this);
    marker_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("rematch/visualization/tracked_obstacles", 10);
    text_marker_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("rematch/visualizationtext_markers", 10);
    path_pub_ = nh_.advertise<nav_msgs::Path>("rematch/obstacle_paths", 10);
    tracker_details_pub_ = nh_.advertise<umap::TrackerDetails>("/rematch/tracker_details", 10);

    nh_.param("rematch_node/SIMILARITY_THRESHOLD", rematch_params_.SIMILARITY_THRESHOLD, 0.6);
    nh_.param("rematch_node/SIMILARITY_WEIGHT", rematch_params_.SIMILARITY_WEIGHT, 0.5);
    nh_.param("rematch_node/IOU_THRESHOLD", rematch_params_.IOU_THRESHOLD, 0.5);
    nh_.param("rematch_node/IOU_WEIGHT", rematch_params_.IOU_WEIGHT, 0.5);
    nh_.param("rematch_node/VELOCITY_DIFF_THRESHOLD", rematch_params_.VELOCITY_DIFF_THRESHOLD, 0.1);
    nh_.param("rematch_node/VELOCITY_DIFF_WEIGHT", rematch_params_.VELOCITY_DIFF_WEIGHT, 0.5);
    nh_.param("rematch_node/REMATCH_VIS_FRAME", rematch_params_.REMATCH_VIS_FRAME, std::string("camera_color_optical_frame"));
    nh_.param("rematch_node/MAX_TRACKER_AGE", rematch_params_.MAX_TRACKER_AGE, 0.0);

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
    curr_obstacles_raw_ = dync_obstacle_msg;

    // 1. Extract 3D bounding boxes from the message
    std::vector<Eigen::Vector3d> centroid_measurements;
    std::vector<Eigen::Vector3d> dimensions;
    std::vector<Eigen::Vector2d> HWboxes;
    std::vector<std::vector<float>> features;
    std::vector<rematchNode::ObstacleMeasureFeature> curr_obstacles;

    for (size_t i = 0; i < dync_obstacle_msg.box_ids.size(); i++)
    {
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
        curr_obstacles.push_back(obstacle);

        ROS_INFO_STREAM("\033[1;35m[Rematch] Receive obstacle " << i 
            << " position: [" << centroid_measurement.transpose() 
            << "], size: [" << dimension.transpose() << "]\033[0m");
    }
    curr_centroidMeasurements_ = centroid_measurements;
    curr_sizes_ = dimensions;

    matchAndUpdate(curr_obstacles);

    publishVisualization();
    publishTrackerDetails(ros::Time::now().toSec());
}

/// @brief Calculate the bounding box for a given centroid and box size.
auto calculate_bounds = [](const Eigen::Vector3d &centroid, const Eigen::Vector3d &box)
{
    std::array<std::pair<double, double>, 3> bounds;
    for (int i = 0; i < 3; ++i)
    {
        bounds[i] = {
            centroid[i] - box[i] / 2, // min
            centroid[i] + box[i] / 2  // max
        };
    }
    return bounds;
};

/// @brief Get cost matrix based on the Euclidean distance between trackers and measurements(for Munkres algorithm).
/// @param trackers
/// @param measurements
/// @return
Eigen::MatrixXd rematchNode::getCostMatrix(vector<onboardDetector::ObstacleTracker> &trackers, const vector<ObstacleMeasureFeature> &measurements)
{
    int n_trackers = trackers.size();
    int m_detections = measurements.size();
    invalid_match_mapping_.clear(); // Clear previous invalid matches
    Eigen::MatrixXd cost_matrix(n_trackers, m_detections);

    double SIMILARITY_THRESHOLD = rematch_params_.SIMILARITY_THRESHOLD;
    double SIMILARITY_WEIGHT = rematch_params_.SIMILARITY_WEIGHT;
    double IOU_THRESHOLD = rematch_params_.IOU_THRESHOLD;
    double IOU_WEIGHT = rematch_params_.IOU_WEIGHT;

    double VELOCITY_DIFF_THRESHOLD = rematch_params_.VELOCITY_DIFF_THRESHOLD;

    ROS_INFO("=== Match Evaluation Start ===");
    for (size_t i = 0; i < n_trackers; ++i)
    {
        for (size_t j = 0; j < m_detections; ++j)
        {
            // TODO: 加入自适应更改cost function的权重: 若lastUpdateTime距现在太久，则降低iou权重，加大特征的权重

            // 1. IoU cost
            Eigen::Vector3d center_box = trackers[i].getState().position;
            Eigen::Vector3d center_detection = measurements[j].centroid_measurement;
            Eigen::Vector3d size_tracker = trackers[i].box_dimensions;
            Eigen::Vector3d size_detection = measurements[j].dimension;

            // Calculate IoU between tracker and detection bounding boxes
            double iou = 0.0;

            // Calculate 3D boundaries for tracker and detection
            std::array<std::pair<double, double>, 3> tracker_bounds, detection_bounds;
            for (int k = 0; k < 3; ++k) {
                tracker_bounds[k] = {
                    center_box[k] - size_tracker[k] / 2.0,
                    center_box[k] + size_tracker[k] / 2.0
                };
                detection_bounds[k] = {
                    center_detection[k] - size_detection[k] / 2.0,
                    center_detection[k] + size_detection[k] / 2.0
                };
            }

            // Calculate intersection volume
            double intersection_volume = 1.0;
            for (int k = 0; k < 3; ++k) {
                double min_bound = std::max(tracker_bounds[k].first, detection_bounds[k].first);
                double max_bound = std::min(tracker_bounds[k].second, detection_bounds[k].second);
                double overlap = std::max(0.0, max_bound - min_bound);
                intersection_volume *= overlap;
            }

            // Calculate union volume
            double tracker_volume = size_tracker[0] * size_tracker[1] * size_tracker[2];
            double detection_volume = size_detection[0] * size_detection[1] * size_detection[2];
            double union_volume = tracker_volume + detection_volume - intersection_volume;

            // Compute IoU
            if (union_volume > 0.0) {
                iou = intersection_volume / union_volume;
            }

            // 2. Cosine similarity for CLIP features
            const auto &feature1 = trackers[i].obcstacleClipFeatures;
            const auto &feature2 = measurements[j].feature;
            double dot_product = std::inner_product(feature1.begin(), feature1.end(), feature2.begin(), 0.0);
            double norm1 = std::sqrt(std::inner_product(feature1.begin(), feature1.end(), feature1.begin(), 0.0));
            double norm2 = std::sqrt(std::inner_product(feature2.begin(), feature2.end(), feature2.begin(), 0.0));
            double cosine_similarity = dot_product / (norm1 * norm2);

            bool iou_valid = iou > IOU_THRESHOLD;
            bool feature_match = cosine_similarity > SIMILARITY_THRESHOLD;

            ROS_INFO_NAMED("Rematch", "Track %zu <-> Det %zu: iou=%.3f sim=%.3f, ", i, j, iou, cosine_similarity);

            if (!feature_match || !iou_valid){
                cost_matrix(i, j) = 999999.0;
                invalid_match_mapping_[i].push_back(j); // 记录无效匹配(tracker_idx, detection_idx)
                ROS_DEBUG_NAMED("Rematch", "Invalid match: T[%ld] vs D[%ld] - IoU: %.3f < threshold: %.3f", i, j, iou, IOU_THRESHOLD);
                if (!iou_valid)
                    ROS_DEBUG_NAMED("Rematch", "  [NO MATCH] Low IoU");
                if (!feature_match)
                    ROS_DEBUG_NAMED("Rematch", "  [NO MATCH] Low feature similarity");
            }
            else{
                double iou_cost = 1.0 - iou;
                double similarity_cost = 1.0 - cosine_similarity;
                cost_matrix(i, j) = SIMILARITY_WEIGHT * similarity_cost 
                                  + IOU_WEIGHT * iou_cost;
                ROS_INFO_NAMED("Rematch", "\033[1;32m  [MATCH SUCCESS] Tracker %zu <-> Detection %zu: IoU=%.3f, Similarity=%.3f, Cost=%.6f\033[0m",
                         i, j, iou, cosine_similarity, cost_matrix(i, j));
            }
        }
    }

    // Print the entire matrix
    std::stringstream ss;
    ss << "\033[1;33m" << "\n[REMATCH] Final Cost Matrix:\n" << cost_matrix << "\033[0m\n";
    ROS_INFO_STREAM(ss.str());
    ROS_INFO("==============================");
    ROS_INFO("=== Match Evaluation End ===");
    return cost_matrix;
}


/// @brief Find unmatched detection indices
/// @param matches Matched pairs from association
/// @param num_detections Total number of detections
/// @return Vector of unmatched detection indices
std::vector<int> findUnmatchedDetections(
    const std::vector<std::pair<int, int>> &matches,
    const int num_detections)
{

    std::vector<bool> matched(num_detections, false);
    std::vector<int> unmatched;

    // Mark matched detections
    for (const auto &match : matches)
    {
        matched[match.second] = true;
    }

    // Collect unmatched detections
    for (int i = 0; i < num_detections; ++i)
    {
        if (!matched[i])
        {
            unmatched.push_back(i);
        }
    }

    return unmatched;
}

/// @brief Update tracker lifecycle and remove stale trackers
/// @param trackers Vector of trackers
/// @param current_time Current timestamp
void updateTrackerLifecycles(std::vector<onboardDetector::ObstacleTracker> &trackers, double current_time, const double max_age) {
    if (trackers.empty() || max_age <= 0.0 ) {
        return;
    }
    
    auto it = trackers.begin();
    while (it != trackers.end())
    {
        double time_since_update = current_time - it->getLastUpdateTime();
        // ROS_INFO_STREAM("Tracker ID: " << it->track_id << " last time: " << it->getLastUpdateTime()
        //                               << "\t current time: " << current_time
        //                               << "\t age: " << time_since_update << "s");
        if (time_since_update > max_age)
        {
            // ROS_INFO_STREAM("Removing stale tracker ID: " << it->track_id << " (age: " << time_since_update << "s)");
            it = trackers.erase(it);
        }
        else
        {
            ++it;
        }
    }
}

/// @brief Associates trackers with detections using the Munkres algorithm.
/// @param cost_matrix
/// @return A vector of pairs representing the associations (tracker index, detection index).
std::vector<std::pair<int, int>> rematchNode::association(Eigen::MatrixXd &cost_matrix) {
    if (cost_matrix.rows() == 0 || cost_matrix.cols() == 0)
        return {};

    if (cost_matrix.rows() == 1 && cost_matrix.cols() == 1 && cost_matrix(0, 0) > 1) {
        ROS_ERROR("1 tracker: No match! Cost matrix is too large.");
        return {};
    }

    Matrix<double> munkres_matrix(cost_matrix.rows(), cost_matrix.cols());
    for (size_t i = 0; i < cost_matrix.rows(); ++i)
        for (size_t j = 0; j < cost_matrix.cols(); ++j)
            munkres_matrix(i, j) = cost_matrix(i, j);

    Munkres<double> m;
    m.solve(munkres_matrix);

    Eigen::MatrixXd result_matrix(cost_matrix.rows(), cost_matrix.cols());
    for (size_t i = 0; i < cost_matrix.rows(); ++i)
        for (size_t j = 0; j < cost_matrix.cols(); ++j)
            result_matrix(i, j) = munkres_matrix(i, j);

    // Debug output for result_matrix
    // ROS_INFO("Result Matrix after Munkres Algorithm:");
    // for (size_t i = 0; i < result_matrix.rows(); ++i) {
    //     std::stringstream row_stream;
    //     for (size_t j = 0; j < result_matrix.cols(); ++j) {
    //         row_stream << result_matrix(i, j) << " ";
    //     }
    //     ROS_INFO("%s", row_stream.str().c_str());
    // }

    // Output invalid_match_mapping_
    // ROS_INFO("[Association] Invalid matches mapping:");
    for (const auto& invalid_match : invalid_match_mapping_) {
        int tracker_idx = invalid_match.first;
        for (int detection_idx : invalid_match.second) {
            // ROS_INFO("Invalid Match - Tracker %d with Detection %d", tracker_idx, detection_idx);
            result_matrix(tracker_idx, detection_idx) = -1.00; // Mark as invalid
        }
    }
    // Debug output for result_matrix
    ROS_INFO("[Detailed tracker] Adjusted Result Matrix after Invalid check:");
    for (size_t i = 0; i < result_matrix.rows(); ++i) {
        std::stringstream row_stream;
        for (size_t j = 0; j < result_matrix.cols(); ++j) {
            row_stream << result_matrix(i, j) << " ";
        }
        ROS_INFO("%s", row_stream.str().c_str());
    }

    std::vector<std::pair<int, int>> assignments;
    std::vector<bool> tracker_matched(cost_matrix.rows(), false);
    std::vector<bool> detection_matched(cost_matrix.cols(), false);

    // Write & print assignments
    // std::cout << "Assignment results:" << std::endl;
    for(size_t i = 0; i < result_matrix.rows(); ++i) { // row index = tracker index
        for(size_t j = 0; j < result_matrix.cols(); ++j) { // column index = detection index
            if(result_matrix(i,j) == 0) {
                // std::cout << "Tracker " << i << " matched with Detection " << j << std::endl;
                assignments.emplace_back(i, j);
                tracker_matched[i] = true;
                detection_matched[j] = true;
            }
        }
    }

    // unmatched tracker
    // for(size_t i = 0; i < tracker_matched.size(); ++i) {
        // if(!tracker_matched[i]) {
            // std::cout << "Tracker " << i << " unmatched" << std::endl;
        // }
    // }
    // unmatched detection
    // for(size_t j = 0; j < detection_matched.size(); ++j) {
        // if(!detection_matched[j]) {
            // std::cout << "Detection " << j << " unmatched" << std::endl;
        // }
    // }
    return assignments;
}


/// @brief 主匹配函数Match and update trackers with new measurements.
/// @param curr_obstacles Vector of current obstacles
void rematchNode::matchAndUpdate(const std::vector<rematchNode::ObstacleMeasureFeature> &curr_obstacles)
{
    double current_time = ros::Time::now().toSec();
    id_mapping_.clear();

    // 0. 若tracker列表为空，则直接将观测的obstacles转为tracker加入
    if (trackers_.empty()) {
        for (size_t i = 0; i < curr_obstacles.size(); i++) {
            onboardDetector::ObstacleTracker new_tracker;
            new_tracker.initWithFeature(curr_obstacles[i].centroid_measurement, curr_obstacles[i].dimension, curr_obstacles[i].feature, current_time, global_tracker_id_);
            new_tracker.updateWithFeature(curr_obstacles[i].centroid_measurement, curr_obstacles[i].dimension, curr_obstacles[i].feature, current_time);
            
            trackers_.push_back(new_tracker);

            id_mapping_[global_tracker_id_] = i;
            ROS_DEBUG_NAMED("rematch", "current tracker id is : %ld ", global_tracker_id_);
            global_tracker_id_++;
        }
        return;
    }

    // 1. 计算cost
    Eigen::MatrixXd cost_matrix = getCostMatrix(trackers_, curr_obstacles);

    // 2. 执行匈牙利算法
    auto matched_pairs = association(cost_matrix);
    // Output matched pairs
    // ROS_ERROR("Matched Pairs:");
    // for (const auto& match : matched_pairs) {
    //     ROS_INFO("Tracker idx %d matched with Detection idx %d", match.first, match.second);
    // }
    // ROS_INFO("Current measurement num: %ld", curr_obstacles.size());

    // 3. 更新匹配的tracker
    for (const auto &match : matched_pairs) {
        int tracker_idx = match.first;
        int measurement_idx = match.second;

        if (tracker_idx < trackers_.size() && measurement_idx < curr_obstacles.size()) {
            trackers_[tracker_idx].updateWithFeature(curr_obstacles[measurement_idx].centroid_measurement, curr_obstacles[measurement_idx].dimension,curr_obstacles[measurement_idx].feature, current_time);
            ROS_DEBUG_NAMED("rematch", "Updating tracker %d with boxSize %f\t%f\t%f", tracker_idx, trackers_[tracker_idx].box_dimensions.x(), trackers_[tracker_idx].box_dimensions.y(), trackers_[tracker_idx].box_dimensions.z()); 
            // id_mapping_[tracker_idx] = measurement_idx;
            int tracker_id = trackers_[tracker_idx].track_id;
            id_mapping_[tracker_id] = measurement_idx;
        }else{
            ROS_ERROR("Tracker %d or measurement %d out of bounds", tracker_idx, measurement_idx);
        }
    }

    // 4. 处理未匹配的检测: 将其作为new_tracker加入
    // 4.1. 找到未匹配的detections
    std::vector<int> unmatched = findUnmatchedDetections(matched_pairs, curr_obstacles.size());
    // 4.2. 创建new_tracker并加入
    for (int idx : unmatched) {
        ROS_INFO_STREAM("Creating new tracker for unmatched detection " << idx);
        onboardDetector::ObstacleTracker new_tracker;

        new_tracker.initWithFeature(curr_obstacles[idx].centroid_measurement, curr_obstacles[idx].dimension, curr_obstacles[idx].feature, current_time, global_tracker_id_);
        new_tracker.updateWithFeature(curr_obstacles[idx].centroid_measurement, curr_obstacles[idx].dimension, curr_obstacles[idx].feature, current_time);
        trackers_.push_back(new_tracker);
        
        id_mapping_[global_tracker_id_] = idx;
        ROS_ERROR("[Detailed] Created new tracker with ID : %ld ", global_tracker_id_);
        global_tracker_id_++;
        
    }

    ROS_DEBUG_NAMED("rematch", "[Match & Update] Current ID Mapping:");
    for (const auto &mapping : id_mapping_)
        ROS_DEBUG_NAMED("rematch", "  Tracker ID %d -> Measurement ID %d", mapping.first, mapping.second);

    // 5. 打印dynamic tracker数据(history pos, vel)
    ROS_INFO("========== Dynamic Trackers Analysis ==========");
    ROS_INFO("Total active trackers: %zu", trackers_.size());
    
    // for (size_t i = 0; i < trackers_.size(); ++i) {
    //     const auto& tracker = trackers_[i];
    //     auto state = tracker.getState();
        
    //     ROS_INFO("--- Tracker ID: %d ---", tracker.track_id);
    //     ROS_INFO("  Current Position: [%.3f, %.3f, %.3f]", 
    //              state.position.x(), state.position.y(), state.position.z());
    //     ROS_INFO("  Current Velocity: [%.3f, %.3f, %.3f] (magnitude: %.3f m/s)", 
    //              state.velocity.x(), state.velocity.y(), state.velocity.z(),
    //              state.velocity.norm());
    //     ROS_INFO("  Box Dimensions: [%.3f, %.3f, %.3f]", 
    //              tracker.box_dimensions.x(), tracker.box_dimensions.y(), tracker.box_dimensions.z());
    //     ROS_INFO("  Last Update Time: %.3f", tracker.getLastUpdateTime());
    //     ROS_INFO("  Age: %.3f seconds", current_time - tracker.getLastUpdateTime());
        
    //     // Print position history if available
    //     // Note: You might need to add a method to ObstacleTracker to access position history
    //     // For now, we'll show predicted future positions based on current velocity
    //     auto future_positions = getNextKFrameState(tracker, 5);
    //     ROS_INFO("  Predicted next 5 positions:");
    //     for (size_t k = 0; k < future_positions.size(); ++k) {
    //         ROS_INFO("    Frame +%zu: [%.3f, %.3f, %.3f]", k+1, 
    //                  future_positions[k].x(), future_positions[k].y(), future_positions[k].z());
    //     }
    //     ROS_INFO("  --------------------------------");
    // }
    // ROS_INFO("===============================================");

    // 6. 更新tracker的生命周期
    updateTrackerLifecycles(trackers_, current_time, rematch_params_.MAX_TRACKER_AGE);

}

std::vector<Eigen::Vector3d> rematchNode::getNextKFrameState(const onboardDetector::ObstacleTracker& tracker, int K) {
    std::vector<Eigen::Vector3d> future_states;
    Eigen::Vector3d current_position = tracker.getState().position;
    Eigen::Vector3d current_velocity = tracker.getState().velocity;

    for (int k = 1; k <= K; ++k) {
        Eigen::Vector3d future_position = current_position + current_velocity * static_cast<double>(k);
        future_states.push_back(future_position);
    }
    return future_states;
}

/// @brief Publish tracker details for logging and analysis
/// @param current_time Current timestamp
void rematchNode::publishTrackerDetails(double current_time) {
    umap::TrackerDetails tracker_details_msg;
    
    // Set header
    tracker_details_msg.header.stamp = ros::Time::now();
    tracker_details_msg.header.frame_id = rematch_params_.REMATCH_VIS_FRAME;
    
    // Populate tracker data
    for (const auto& tracker : trackers_) {
        auto state = tracker.getState();
        
        tracker_details_msg.tracker_ids.push_back(tracker.track_id);
        
        geometry_msgs::Point position;
        position.x = state.position.x();
        position.y = state.position.y();
        position.z = state.position.z();
        tracker_details_msg.positions.push_back(position);
        
        geometry_msgs::Vector3 velocity;
        velocity.x = state.velocity.x();
        velocity.y = state.velocity.y();
        velocity.z = state.velocity.z();
        tracker_details_msg.velocities.push_back(velocity);
        
        geometry_msgs::Vector3 box_dim;
        box_dim.x = tracker.box_dimensions.x();
        box_dim.y = tracker.box_dimensions.y();
        box_dim.z = tracker.box_dimensions.z();
        tracker_details_msg.box_dimensions.push_back(box_dim);
        
        tracker_details_msg.last_update_times.push_back(tracker.getLastUpdateTime());
        tracker_details_msg.ages.push_back(current_time - tracker.getLastUpdateTime());
    }
    ROS_INFO("Publishing tracker details with %zu entries", tracker_details_msg.tracker_ids.size());
    tracker_details_pub_.publish(tracker_details_msg);
}