/// @file detailed_matcher.cpp
/// @brief Detailed (re-match) matching algorithm implementations — cost matrix (IoU+distance+CLIP similarity), Hungarian association, match-and-update, tracker details publishing

#include <umap/rematch.h>
#include <umap/match_utils.hpp>
#include <munkres/munkres.h>
#include <numeric>

/// @brief Get cost matrix based on the Euclidean distance between trackers and measurements(for Munkres algorithm).
/// @param trackers
/// @param measurements
/// @return
Eigen::MatrixXd rematchNode::getCostMatrix(std::vector<onboardDetector::ObstacleTracker> &trackers, const std::vector<ObstacleMeasureFeature> &measurements)
{
    int n_trackers = trackers.size();
    int m_detections = measurements.size();
    invalid_match_mapping_.clear(); // Clear previous invalid matches
    Eigen::MatrixXd cost_matrix(n_trackers, m_detections);

    double SIMILARITY_THRESHOLD = rematch_params_.SIMILARITY_THRESHOLD;
    double SIMILARITY_WEIGHT = rematch_params_.SIMILARITY_WEIGHT;
    double IOU_THRESHOLD = rematch_params_.IOU_THRESHOLD;
    double IOU_WEIGHT = rematch_params_.IOU_WEIGHT;

    double DISTANCE_THRESHOLD = rematch_params_.DISTANCE_THRESHOLD;
    double DISTANCE_WEIGHT = rematch_params_.DISTANCE_WEIGHT;

    ROS_INFO("=== Match Evaluation Start ===");
    for (size_t i = 0; i < n_trackers; ++i)
    {
        for (size_t j = 0; j < m_detections; ++j)
        {
            // TODO: 加入自适应更改cost function的权重: 若lastUpdateTime距现在太久，则降低iou权重，加大特征的权重

            // 1. IoU / distance cost
            Eigen::Vector3d center_box = trackers[i].getState().position;
            Eigen::Vector3d center_detection = measurements[j].centroid_measurement;
            Eigen::Vector3d size_tracker = trackers[i].box_dimensions;
            Eigen::Vector3d size_detection = measurements[j].dimension;

            // Calculate IoU between tracker and detection bounding boxes
            double iou = umap::match_utils::computeIoU3D(
                center_box, size_tracker, center_detection, size_detection);
            double distance = (center_box - center_detection).norm();

            // 2. Cosine similarity for CLIP features
            const auto &feature1 = trackers[i].obcstacleClipFeatures;
            const auto &feature2 = measurements[j].feature;
            
            double cosine_similarity = 0.0;
            if (!feature1.empty() && !feature2.empty() && feature1.size() == feature2.size()) {
                double dot_product = std::inner_product(feature1.begin(), feature1.end(), feature2.begin(), 0.0);
                double norm1 = std::sqrt(std::inner_product(feature1.begin(), feature1.end(), feature1.begin(), 0.0));
                double norm2 = std::sqrt(std::inner_product(feature2.begin(), feature2.end(), feature2.begin(), 0.0));
                
                if (norm1 > 1e-6 && norm2 > 1e-6) {
                    cosine_similarity = dot_product / (norm1 * norm2);
                }
            } else {
                ROS_WARN_THROTTLE(1.0, "Feature dimension mismatch or empty: T[%zu]=%zu, D[%zu]=%zu", 
                                  i, feature1.size(), j, feature2.size());
            }

            bool iou_valid = iou > IOU_THRESHOLD;
            bool distance_valid = distance < DISTANCE_THRESHOLD;
            if (IOU_THRESHOLD < 0.0) // disable iou check
                iou_valid = true;

            bool feature_match = cosine_similarity > SIMILARITY_THRESHOLD;

            ROS_INFO_NAMED("Rematch", "Track %zu <-> Det %zu: iou=%.3f sim=%.3f, ", i, j, iou, cosine_similarity);

            if (!feature_match || !iou_valid || !distance_valid){
                cost_matrix(i, j) = 999999.0;
                invalid_match_mapping_[i].push_back(j); // 记录无效匹配(tracker_idx, detection_idx)
                ROS_DEBUG_NAMED("Rematch", "Invalid match: T[%ld] vs D[%ld] - IoU: %.3f < threshold: %.3f", i, j, iou, IOU_THRESHOLD);
                if (!iou_valid)
                    ROS_INFO("Rematch  [NO MATCH] Low IoU");
                if (!distance_valid)
                    ROS_INFO("Rematch  [NO MATCH] Large distance");
                if (!feature_match)
                    ROS_INFO("Rematch  [NO MATCH] Low feature similarity");
            }
            else{
                double iou_cost = 1.0 - iou;
                double similarity_cost = 1.0 - cosine_similarity;
                double distance_cost = distance;

                cost_matrix(i, j) = SIMILARITY_WEIGHT * similarity_cost 
                                  + IOU_WEIGHT * iou_cost
                                  + DISTANCE_WEIGHT * distance_cost;
                ROS_INFO_NAMED("Rematch", "\033[1;32m  [MATCH SUCCESS] Tracker %zu <-> Detection %zu: IoU=%.3f, Similarity=%.3f, Distance=%.3f, Cost=%.6f\033[0m",
                         i, j, iou, cosine_similarity, distance, cost_matrix(i, j));
            }
        }
    }

    // Print the entire matrix
    std::stringstream ss;
    ss << "\033[1;33m" << "\n[REMATCH] Final Cost Matrix:\n" << cost_matrix << "\033[0m\n";
    ROS_DEBUG_STREAM(ss.str());
    ROS_INFO("==============================");
    ROS_INFO("=== Match Evaluation End ===");

    // ── Profiling: cost matrix computation done ───────────────────────────────
    tp_cost_done_ = std::chrono::steady_clock::now();

    return cost_matrix;
}


// findUnmatchedDetections / updateTrackerLifecycles moved to
// <umap/match_utils.hpp> for reuse between coarse and detailed match nodes.

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
    ROS_DEBUG("[Detailed tracker] Adjusted Result Matrix after Invalid check:");
    for (size_t i = 0; i < result_matrix.rows(); ++i) {
        std::stringstream row_stream;
        for (size_t j = 0; j < result_matrix.cols(); ++j) {
            row_stream << result_matrix(i, j) << " ";
        }
        ROS_DEBUG("%s", row_stream.str().c_str());
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

    // ── Profiling: Munkres association done ──────────────────────────────────
    tp_assoc_done_ = std::chrono::steady_clock::now();

    return assignments;
}


/// @brief 主匹配函数Match and update trackers with new measurements.
/// @param curr_obstacles Vector of current obstacles
void rematchNode::matchAndUpdate(const std::vector<rematchNode::ObstacleMeasureFeature> &curr_obstacles)
{
    // Preserve the source measurement timestamp through the asynchronous
    // appearance path; wall-clock completion time is not a measurement time.
    const double current_time = curr_obstacles_raw_.header.stamp.toSec();
    id_mapping_.clear();

    // 0. 若tracker列表为空，则直接将观测的obstacles转为tracker加入
    if (trackers_.empty()) {
        for (size_t i = 0; i < curr_obstacles.size(); i++) {
            onboardDetector::ObstacleTracker new_tracker;
            new_tracker.initWithFeature(curr_obstacles[i].centroid_measurement, curr_obstacles[i].dimension, curr_obstacles[i].feature, current_time, global_tracker_id_);
            // Do not immediately update with the same measurement/timestamp:
            // velocity warm-start needs two distinct observations.
            new_tracker.last_confidence = curr_obstacles[i].confidence;
            
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
            trackers_[tracker_idx].was_dimension_expanded = false; // Reset flag when getting new measurements
            trackers_[tracker_idx].last_confidence = curr_obstacles[measurement_idx].confidence;
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
    std::vector<int> unmatched = umap::match_utils::findUnmatchedDetections(matched_pairs, curr_obstacles.size());
    // 4.2. 创建new_tracker并加入
    for (int idx : unmatched) {
        ROS_INFO_STREAM("Creating new tracker for unmatched detection " << idx);
        onboardDetector::ObstacleTracker new_tracker;

        new_tracker.initWithFeature(curr_obstacles[idx].centroid_measurement, curr_obstacles[idx].dimension, curr_obstacles[idx].feature, current_time, global_tracker_id_);
        // Do not immediately update with the same measurement/timestamp:
        // velocity warm-start needs two distinct observations.
        new_tracker.last_confidence = curr_obstacles[idx].confidence;
        trackers_.push_back(new_tracker);
        
        id_mapping_[global_tracker_id_] = idx;
        ROS_INFO_STREAM("\033[1;32m[Detailed] Created new tracker with ID : " << global_tracker_id_ << "\033[0m");
        global_tracker_id_++;
        
    }

    ROS_DEBUG_NAMED("rematch", "[Match & Update] Current ID Mapping:");
    for (const auto &mapping : id_mapping_)
        ROS_DEBUG_NAMED("rematch", "  Tracker ID %d -> Measurement ID %d", mapping.first, mapping.second);

    // 5. 打印dynamic tracker数据(history pos, vel) — DEBUG only.
    // The "future positions" computed via getNextKFrameState are NOT consumed
    // downstream (TrackerDetail.msg has no future_positions field, MPC does
    // its own constant-velocity extrapolation). Keep the print as a debug
    // aid but stop spamming INFO every frame.
    ROS_DEBUG_NAMED("rematch", "========== Dynamic Trackers Analysis ==========");
    ROS_DEBUG_NAMED("rematch", "Total active trackers: %zu", trackers_.size());

    for (size_t i = 0; i < trackers_.size(); ++i) {
        const auto& tracker = trackers_[i];
        auto state = tracker.getState();

        ROS_DEBUG_NAMED("rematch", "--- Tracker ID: %d ---", tracker.track_id);
        ROS_DEBUG_NAMED("rematch", "  Current Position: [%.3f, %.3f, %.3f]",
                 state.position.x(), state.position.y(), state.position.z());
        ROS_DEBUG_NAMED("rematch", "  Current Velocity: [%.3f, %.3f, %.3f] (magnitude: %.3f m/s)",
                 state.velocity.x(), state.velocity.y(), state.velocity.z(),
                 state.velocity.norm());
        ROS_DEBUG_NAMED("rematch", "  Box Dimensions: [%.3f, %.3f, %.3f]",
                 tracker.box_dimensions.x(), tracker.box_dimensions.y(), tracker.box_dimensions.z());
        ROS_DEBUG_NAMED("rematch", "  Last Update Time: %.3f", tracker.getLastUpdateTime());
        ROS_DEBUG_NAMED("rematch", "  Age: %.3f seconds", current_time - tracker.getLastUpdateTime());

        // Print predicted future positions (constant-velocity, KF state-based).
        // Currently only used for debug visibility; not exported to downstream.
        auto future_positions = getNextKFrameState(tracker, 5);
        ROS_DEBUG_NAMED("rematch", "  Predicted next 5 positions:");
        for (size_t k = 0; k < future_positions.size(); ++k) {
            ROS_DEBUG_NAMED("rematch", "    Frame +%zu: [%.3f, %.3f, %.3f]", k+1,
                     future_positions[k].x(), future_positions[k].y(), future_positions[k].z());
        }
        ROS_DEBUG_NAMED("rematch", "  --------------------------------");
    }
    ROS_DEBUG_NAMED("rematch", "===============================================");

    // 6. Predict for unmatched dynamic trackers so their KF state is up-to-date
    if (!trackers_.empty()) {
        std::vector<bool> is_tracker_matched(trackers_.size(), false);
        for (const auto& match : matched_pairs) {
            if (match.first >= 0 && match.first < static_cast<int>(trackers_.size())) {
                is_tracker_matched[match.first] = true;
            }
        }
        for (size_t i = 0; i < trackers_.size(); ++i) {
            if (is_tracker_matched[i]) continue;
            // Only continue prediction/publishing for trackers that haven't timed out
            double time_since_update = current_time - trackers_[i].getLastUpdateTime();
            if (time_since_update <= rematch_params_.UNMATCHED_PUBLISH_TIMEOUT) {
                // Use dimension expansion prediction for dynamic trackers
                trackers_[i].predictOnlyWithDimensionExpansion(current_time, rematch_params_.EXPANSION_RATE);
                trackers_[i].was_dimension_expanded = true; // Mark as dimension expanded
                
                ROS_DEBUG_NAMED("Detailed rematch", "[Dimension Expansion] Tracker ID %d: time_since_update=%.3fs, dimensions=[%.3f, %.3f, %.3f]", 
                               trackers_[i].track_id, time_since_update, 
                               trackers_[i].box_dimensions.x(), trackers_[i].box_dimensions.y(), trackers_[i].box_dimensions.z());
            }
        }
    }

    // 7. 更新tracker的生命周期
    umap::match_utils::removeStaleTrackers(
        trackers_, current_time, rematch_params_.MAX_TRACKER_AGE,
        [](int /*dead_id*/) { /* detailed node has no per-tracker caches to prune */ });

    // ── Profiling: matchAndUpdate done ────────────────────────────────────────
    tp_update_done_ = std::chrono::steady_clock::now();
    stat_num_matched_   = static_cast<int>(matched_pairs.size());
    stat_num_unmatched_ = static_cast<int>(unmatched.size());
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
    umap::TrackerDetailArray tracker_details_msg;
    
    // Set header
    tracker_details_msg.header.stamp = ros::Time(current_time);
    tracker_details_msg.header.frame_id = rematch_params_.REMATCH_OUTPUT_FRAME;
    
    // Populate tracker data
    for (const auto& tracker : trackers_) {
        auto state = tracker.getState();
        
        umap::TrackerDetail detail;
        detail.tracker_id = tracker.track_id;
        
        detail.position.x = state.position.x();
        detail.position.y = state.position.y();
        detail.position.z = state.position.z();
        
        detail.velocity.x = state.velocity.x();
        detail.velocity.y = state.velocity.y();
        detail.velocity.z = state.velocity.z();
        
        detail.box_dimensions.x = tracker.box_dimensions.x();
        detail.box_dimensions.y = tracker.box_dimensions.y();
        detail.box_dimensions.z = tracker.box_dimensions.z();
        
        detail.last_update_time = tracker.getLastUpdateTime();
        detail.age = current_time - tracker.getLastUpdateTime();
        detail.confidence = tracker.last_confidence;

        // KF covariance diagonal — used by MPC to inflate the swept-capsule
        // radius along the prediction horizon. Cross terms ~= 0 in the CV
        // KF used here (see TrackerDetail.msg for justification).
        detail.sigma2_xx    = static_cast<float>(tracker.getCovariance(0, 0));
        detail.sigma2_yy    = static_cast<float>(tracker.getCovariance(1, 1));
        detail.sigma2_vxvx  = static_cast<float>(tracker.getCovariance(3, 3));
        detail.sigma2_vyvy  = static_cast<float>(tracker.getCovariance(4, 4));

        tracker_details_msg.trackers.push_back(detail);
    }
    ROS_DEBUG_NAMED("rematch", "Publishing tracker details with %zu entries", tracker_details_msg.trackers.size());
    tracker_details_pub_.publish(tracker_details_msg);
}

/// @brief Publish per-frame profiling statistics to /rematch_statistics
/// @param stats Fully populated RematchStatistics message
void rematchNode::publishStatistics(const umap::RematchStatistics& stats)
{
    statistics_pub_.publish(stats);
    ROS_DEBUG("[Rematch profiler] total=%.2f ms  parse=%.2f  cost=%.2f  assoc=%.2f  update=%.2f  vis=%.2f  det=%.2f",
              stats.t_total_ms, stats.t_parse_ms, stats.t_cost_matrix_ms,
              stats.t_association_ms, stats.t_update_ms,
              stats.t_publish_vis_ms, stats.t_publish_det_ms);
}
