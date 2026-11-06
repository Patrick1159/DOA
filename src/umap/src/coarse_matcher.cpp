/// @file coarse_matcher.cpp
/// @brief Coarse matching algorithm implementations — cost matrix, Hungarian association, match-and-update, dynamic classification

#include <umap/coarse_match.h>
#include <umap/match_utils.hpp>
#include <umap/cost_guard.h>
#include <umap/utils.h>
#include <munkres/munkres.h>

/// @brief Check if the tracker is dynamic.
/// @param tracker: ObstacleTracker object
bool trackerNode::isDynamicTracker(onboardDetector::ObstacleTracker& tracker) const {
    auto state = tracker.getState();
    double velocity = state.velocity.norm();
    double avg_movement = 0.0;

    // 1. Average movement over history
    if (!tracker.history_pos.empty()) {
        for (const auto& pos : tracker.history_pos) {
            avg_movement += pos.norm();
        }
        avg_movement /= tracker.history_pos.size();
    }

    // 2. Check sustained velocity over n frames
    const double SUSTAINED_SPEED_THRESHOLD = params_.SUSTAINED_VELOCITY_THRESHOLD;
    const int MIN_CONSECUTIVE_FRAMES = params_.SUSTAINED_VELOCITY_COUNT;
    bool is_velocity_sustained = false;

    if (tracker.history_vel.size() >= MIN_CONSECUTIVE_FRAMES) {
        int consecutive_count = 0;
        for (size_t i = tracker.history_vel.size() - MIN_CONSECUTIVE_FRAMES; i < tracker.history_vel.size(); ++i) {
            double vel_norm = tracker.history_vel[i].norm();
            if (vel_norm > SUSTAINED_SPEED_THRESHOLD) {
                consecutive_count++;
            } else {
                consecutive_count = 0;
            }
        }
        is_velocity_sustained = (consecutive_count >= MIN_CONSECUTIVE_FRAMES);
    }

    // 综合判断动态性
    bool isDynamic = (velocity >= params_.DYNAMIC_VELOCITY_THRESHOLD) &&
                     (velocity <= params_.DYNAMIC_VELOCITY_MAX) &&
                     (avg_movement >= params_.AVERAGE_DEPLACEMENT_THRESHOLD) &&
                     (is_velocity_sustained); // 加入持续性检查

    // Append the new dynamic status to the history
    tracker.isDynamicHistory.push_back(isDynamic);

    // Queue: Ensure the history size does not exceed the limit
    const size_t STATUS_HISTORY_SIZE = params_.STATUS_HISTORY_SIZE; // 10
    if (tracker.isDynamicHistory.size() > STATUS_HISTORY_SIZE) {
        tracker.isDynamicHistory.erase(tracker.isDynamicHistory.begin());
    }



    // TODO: 上面下面看下哪个好
    // 1. 根据最后n帧是否连续来判断
    // Check the last few frames for consistent dynamic status
    int consecutive_dynamic_count = 0;
    int consecutive_static_count = 0;

    for (size_t i = tracker.isDynamicHistory.size() - std::min(tracker.isDynamicHistory.size(), static_cast<size_t>(params_.MIN_CONSECUTIVE_FRAMES_FOR_STATUS_CHANGE)); 
         i < tracker.isDynamicHistory.size(); ++i) {
        if (tracker.isDynamicHistory[i]) {
            consecutive_dynamic_count++;
            consecutive_static_count = 0; // Reset static count
        } else {
            consecutive_static_count++;
            consecutive_dynamic_count = 0; // Reset dynamic count
        }

        // If either dynamic or static status is consistent for the required number of frames, finalize it
        if (consecutive_dynamic_count >= params_.MIN_CONSECUTIVE_FRAMES_FOR_STATUS_CHANGE) {
            tracker.isDynamic = true;
            break;
        } else if (consecutive_static_count >= params_.MIN_CONSECUTIVE_FRAMES_FOR_STATUS_CHANGE) {
            tracker.isDynamic = false;
            break;
        }
    }
    // Logging for debugging
    // ROS_INFO("[Tracker] ID: %d - Velocity: %.2f m/s, Avg Movement: %.2f, Sustained: %d, isDynamic: %d",
    //          tracker.track_id, velocity, avg_movement, is_velocity_sustained, isDynamic);

    // for (size_t i = 0; i < tracker.isDynamicHistory.size(); ++i) {
    //     ROS_INFO("[Tracker] isDynamicHistory[%zu]: %s", i, tracker.isDynamicHistory[i] ? "true" : "false");
    // }
    // // Output velocity history for the tracker
    // ROS_INFO("[Tracker] Velocity History for Tracker ID: %d", tracker.track_id);
    // for (size_t i = 0; i < tracker.history_vel.size(); ++i) {
    //     ROS_INFO("[Tracker] history_vel[%zu]: [%.2f, %.2f, %.2f] (norm: %.2f)", 
    //              i, 
    //              tracker.history_vel[i].x(), tracker.history_vel[i].y(), tracker.history_vel[i].z(),
    //              tracker.history_vel[i].norm());
    // }
    
    return isDynamic;
}

/// @brief True if centroid and size are all finite (never feed anything else to the KF).
static inline bool isFiniteMeasurement(const Eigen::Vector3d& centroid, const Eigen::Vector3d& size) {
    return umap::match_utils::allFinite(centroid) && umap::match_utils::allFinite(size);
}

/// @brief Get Munkres cost matrix between trackers and measurements.
/// @param trackers
/// @param measurements
Eigen::MatrixXd trackerNode::getCost_matrix(std::vector<onboardDetector::ObstacleTracker>& trackers,
        const std::vector<Eigen::Vector3d>& measurements, const std::vector<Eigen::Vector3d>& bbox_sizes,
        const double iou_threshold, const double dist_threshold,
        const double iou_weight, const double dist_weight) {
    int n_trackers = trackers.size();
    int m_detections = measurements.size();
    int n_bbox_sizes = bbox_sizes.size();
    invalid_match_mapping_.clear(); // Clear previous invalid matches
    Eigen::MatrixXd cost_matrix(n_trackers, m_detections);

    if (n_bbox_sizes == m_detections) {
        // Print measurements information
        ROS_DEBUG_NAMED("Coarse tracker", "=== Current Measurements ===");
        for (int j = 0; j < m_detections; ++j) {
            ROS_DEBUG_NAMED("Coarse tracker", "Detection[%d] position: (%.3f, %.3f, %.3f) size(WHT): (%.3f, %.3f, %.3f)",
                j,
                measurements[j].x(), measurements[j].y(), measurements[j].z(),
                bbox_sizes[j].x(), bbox_sizes[j].y(), bbox_sizes[j].z());
        }

        // Print tracker information
        ROS_DEBUG_NAMED("Coarse tracker", "=== Current Trackers ===");
        for (int i = 0; i < n_trackers; ++i) {
            auto state = trackers[i].getState();
            ROS_DEBUG_NAMED("Coarse tracker", "Tracker[%d] position: (%.3f, %.3f, %.3f) velocity: (%.3f, %.3f, %.3f)",
                i,
                state.position.x(), state.position.y(), state.position.z(),
                state.velocity.x(), state.velocity.y(), state.velocity.z());
            ROS_DEBUG_NAMED("Coarse tracker", "Tracker[%d] dimensions: (%.3f, %.3f, %.3f)", 
                i, 
                trackers[i].box_dimensions.x(), trackers[i].box_dimensions.y(), trackers[i].box_dimensions.z());
        }

        // 1. Calculate costs
        ROS_DEBUG_NAMED("Coarse tracker", "=== Cost Matrix [%d x %d] ===", n_trackers, m_detections);

        for (int i = 0; i < n_trackers; ++i) {
            for (int j = 0; j < m_detections; ++j) {
                Eigen::Vector3d center_of_tracker = trackers[i].getState().position;
                Eigen::Vector3d center_of_detection = measurements[j];
                Eigen::Vector3d size_tracker = trackers[i].box_dimensions;
                Eigen::Vector3d size_detection = bbox_sizes[j];
                
                // 1. IoU cost (volumetric IoU between tracker and detection AABBs)
                double iou = umap::match_utils::computeIoU3D(
                    center_of_tracker, size_tracker,
                    center_of_detection, size_detection);
                double iou_cost = 1.0 - iou;
                
                // 2. Distance cost (Euclidean)
                double distance = (center_of_tracker - center_of_detection).norm();
                
                bool gated = false;
                if (iou < iou_threshold) {
                    iou_cost = 999999.0;
                    gated = true;
                    invalid_match_mapping_[i].push_back(j); // 记录无效匹配(tracker_idx, detection_idx)
                    // ROS_INFO("[Coarse tracker] Invalid match: T[%d] vs D[%d] - IoU: %.3f < threshold: %.3f", i, j, iou, iou_threshold);
                }else if (distance > dist_threshold) {
                    iou_cost = 999999.0;
                    gated = true;
                    invalid_match_mapping_[i].push_back(j);
                    // ROS_INFO("[Coarse tracker] Invalid match: T[%d] vs D[%d] - Distance: %.3f > threshold: %.3f", i, j, distance, dist_threshold);
                }

                cost_matrix(i, j) = iou_weight * iou_cost + dist_weight * distance;

                // Non-finite guard: NaN/inf (e.g. from a NaN tracker state or
                // measurement) is neither < iou_threshold nor > dist_threshold,
                // so it would slip through the gates and hang Munkres::solve.
                // Treat it as a gated (invalid) pair.
                if (!std::isfinite(cost_matrix(i, j))) {
                    ROS_WARN_THROTTLE(1.0, "[Coarse tracker] Non-finite cost for T[%d] vs D[%d] "
                                           "(tracker pos [%f %f %f], detection [%f %f %f]) -> gated",
                                      i, j,
                                      center_of_tracker.x(), center_of_tracker.y(), center_of_tracker.z(),
                                      center_of_detection.x(), center_of_detection.y(), center_of_detection.z());
                    cost_matrix(i, j) = umap::match_utils::kCostGate;
                    if (!gated) invalid_match_mapping_[i].push_back(j);
                }

                // ROS_INFO("[Coarse tracker] T[%d] vs D[%d]: dist=%.3f cost=%.3f iou=%.3f", 
                //     i, j, (center_of_tracker - center_of_detection).norm(), cost_matrix(i, j), iou);
            }
        }
        
        // Print the entire matrix
        // ROS_INFO("=== Final Cost Matrix ===");
        // for (int i = 0; i < cost_matrix.rows(); ++i) {
        //     std::stringstream row_stream;
        //     for (int j = 0; j < cost_matrix.cols(); ++j) {
        //         row_stream << std::fixed << std::setprecision(3) << cost_matrix(i, j) << " ";
        //     }
        //     ROS_INFO("Row[%d]: %s", i, row_stream.str().c_str());
        // }
        // ROS_INFO("==============================");
    }

    return cost_matrix;
}


// findUnmatchedDetections / updateTrackerLifecycles moved to
// <umap/match_utils.hpp> for reuse between coarse and detailed match nodes.

/// @brief Associates trackers with detections using the Munkres algorithm.
/// @param cost_matrix 
/// @return A vector of pairs representing the associations (tracker index, detection index)返回的是trackers_中的索引！.
std::vector<std::pair<int, int>> trackerNode::association(const Eigen::MatrixXd& cost_matrix) {

    if (cost_matrix.rows() == 0 || cost_matrix.cols() == 0) {
        return {};
    }
    if (cost_matrix.rows() == 1 && cost_matrix.cols() == 1 && !(cost_matrix(0, 0) <= 1)) {
        // No match! Too different (or non-finite, which the old `> 1` test let through)
        return {};
    }
    
    Matrix<double> munkres_matrix(cost_matrix.rows(), cost_matrix.cols());

    // Last line of defence: Munkres<double>::solve never returns on NaN
    // input. getCost_matrix() already gates non-finite entries; if any is
    // still present, replace it here and never call solve() with it.
    Eigen::MatrixXd safe_cost = cost_matrix;
    const int n_replaced = umap::match_utils::sanitizeCostMatrix(safe_cost);
    if (n_replaced > 0) {
        ROS_ERROR_THROTTLE(1.0, "[Coarse tracker] association(): %d non-finite cost entries replaced by gate "
                                "before Munkres::solve", n_replaced);
    }

    // Convert Eigen matrix to Munkres matrix format
    for(size_t i = 0; i < safe_cost.rows(); ++i) {
        for(size_t j = 0; j < safe_cost.cols(); ++j) {
            munkres_matrix(i,j) = safe_cost(i,j);
        }
    }

    Munkres<double> m;
    m.solve(munkres_matrix);

    // Convert back to Eigen matrix
    Eigen::MatrixXd result_matrix(cost_matrix.rows(), cost_matrix.cols());
    for(size_t i = 0; i < cost_matrix.rows(); ++i) {
        for(size_t j = 0; j < cost_matrix.cols(); ++j) {
            result_matrix(i,j) = munkres_matrix(i,j);
        }
    }
    // Debug output for result_matrix
    // ROS_INFO("Result Matrix after Munkres Algorithm:");
    // for (size_t i = 0; i < result_matrix.rows(); ++i) {
    //     std::stringstream row_stream;
    //     for (size_t j = 0; j < result_matrix.cols(); ++j) {
    //         row_stream << result_matrix(i, j) << " ";
    //     }
    //     ROS_INFO("%s", row_stream.str().c_str());
    // }

    // Postprocessing: Mark invalid matches
    // ROS_INFO("[Association] Invalid matches mapping:");
    for (const auto& invalid_match : invalid_match_mapping_) {
        int tracker_idx = invalid_match.first;
        for (int detection_idx : invalid_match.second) {
            // ROS_INFO("Invalid Match - Tracker %d with Detection %d", tracker_idx, detection_idx);
            result_matrix(tracker_idx, detection_idx) = -1.00; // Mark as invalid
        }
    }

    // Debug output for result_matrix
    // ROS_INFO("\033[34m[Coarse Match] Adjusted Result Matrix after Invalid check:\033[0m");
    for (size_t i = 0; i < result_matrix.rows(); ++i) {
        std::stringstream row_stream;
        for (size_t j = 0; j < result_matrix.cols(); ++j) {
            row_stream << result_matrix(i, j) << " ";
        }
        // ROS_INFO("%s", row_stream.str().c_str());
    }

    std::vector<std::pair<int, int>> assignments; // to return
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
    //     if(!detection_matched[j]) {
    //         std::cout << "Detection " << j << " unmatched" << std::endl;
    //     }
    // }
    return assignments;
}

/// @brief Match and update trackers with new measurements.
/// @param measurements Vector of new measurements
/// @param bbox_sizes Vector of bounding box sizes(width height thickness)
void trackerNode::matchAndUpdate(const std::vector<Eigen::Vector3d>& measurements, const std::vector<Eigen::Vector3d>& bbox_sizes) {
    // Tracker propagation must use the measurement time, never completion
    // time.  This keeps replay, live operation, and offline evaluation on the
    // same sensor time axis and avoids CPU/GPU load changing the KF delta-t.
    const double current_time = curr_boxes3D_raw_.header.stamp.toSec();

    // updateTrackerLifecycles(trackers_, current_time);

    // Clear previous mapping
    id_mapping_.clear();

    ROS_DEBUG_NAMED("Coarse tracker", "[Tracker] Processing - Measurements: %zu, BBox sizes: %zu", measurements.size(), bbox_sizes.size());

    // If no trackers exist yet, initialize them
    if (trackers_.empty()) {
        for (size_t i = 0; i < measurements.size(); i++) {
            if (!isFiniteMeasurement(measurements[i], bbox_sizes[i])) {
                ROS_ERROR_THROTTLE(1.0, "[Coarse tracker] refusing to init tracker from non-finite measurement %zu", i);
                continue;
            }
            onboardDetector::ObstacleTracker new_tracker;
            std::string uuid = umap::generate_uuid(); // TODO: to finish for uuid
            new_tracker.init(measurements[i], bbox_sizes[i], current_time, nextAvailableTrackerId, uuid);
            ROS_INFO("[Coarse tracker] Initializing first tracker with ID: %d and UUID: %s", nextAvailableTrackerId, uuid.c_str());

            ROS_DEBUG_NAMED("Coarse tracker", "[MATCH & UPDATE DEBUG] Initializing new tracker with values:");
            ROS_DEBUG_NAMED("Coarse tracker", "Measurement: [%.3f, %.3f, %.3f]", measurements[i].x(), measurements[i].y(), measurements[i].z());
            ROS_DEBUG_NAMED("Coarse tracker", "Bounding Box Size: [%.3f, %.3f, %.3f]", bbox_sizes[i].x(), bbox_sizes[i].y(), bbox_sizes[i].z());
            ROS_DEBUG_NAMED("Coarse tracker", "Current Time: %.3f", current_time);
            ROS_DEBUG_NAMED("Coarse tracker", "Tracker ID: %d", nextAvailableTrackerId);
            trackers_.push_back(new_tracker);
            
            // Add to mapping
            id_mapping_[nextAvailableTrackerId] = i; // tracker_id vs measurement idx
            ROS_DEBUG_NAMED("Coarse tracker", "This tracker id is : %d ", nextAvailableTrackerId);
            nextAvailableTrackerId++;
        }
        return;
    }

    // Calculate cost matrix
    Eigen::MatrixXd cost_matrix = getCost_matrix(trackers_, measurements, bbox_sizes, 
        params_.DYNAMIC_IOU_THRESHOLD, params_.DYNAMIC_DIST_THRESHOLD, params_.iou_weight, params_.dist_weight);

    // Perform data association
    auto matched_pairs = association(cost_matrix); // 这个matched_pairs是trackers_的索引与measurements的索引的对应关系
    // Output matched pairs
    ROS_INFO("[Coarse tracker] Matched Pairs:");
    for (const auto& match : matched_pairs) {
        ROS_INFO("Tracker idx %d matched with Detection idx %d", match.first, match.second);
    }
    ROS_DEBUG_NAMED("Coarse tracker", "Current measurement num: %ld", measurements.size());

    // Update matched trackers and mapping
    for (const auto& match : matched_pairs) {
        int tracker_idx = match.first;
        int measurement_idx = match.second;

        if (tracker_idx < trackers_.size() && measurement_idx < measurements.size()) {
            if (!isFiniteMeasurement(measurements[measurement_idx], bbox_sizes[measurement_idx])) {
                ROS_ERROR_THROTTLE(1.0, "[Coarse tracker] refusing to feed non-finite measurement %d to tracker %d",
                                   measurement_idx, tracker_idx);
                continue;
            }
            // Update tracker with new measurement
            trackers_[tracker_idx].update(measurements[measurement_idx],bbox_sizes[measurement_idx], current_time, params_.HISTORY_SIZE);
            trackers_[tracker_idx].was_dimension_expanded = false; // Reset flag when getting new measurements
            // Update mapping: use tracker_id (not tracker_idx) to be consistent with initialization
            int tracker_id = trackers_[tracker_idx].track_id;
            id_mapping_[tracker_id] = measurement_idx;

            // Update last observed measurement caches for this tracker
            if (measurement_idx < curr_masks_.size()) {
                last_mask_by_tracker_[tracker_id] = curr_masks_[measurement_idx];
            }
            if (measurement_idx < curr_boxes2D_raw_.x.size()) {
                last_box2d_by_tracker_[tracker_id] = {
                    curr_boxes2D_raw_.x[measurement_idx],
                    curr_boxes2D_raw_.y[measurement_idx],
                    curr_boxes2D_raw_.width[measurement_idx],
                    curr_boxes2D_raw_.height[measurement_idx]
                };
            }
            if (measurement_idx < bbox_sizes.size()) {
                last_size3d_by_tracker_[tracker_id] = bbox_sizes[measurement_idx];
            }
        }else{
            ROS_ERROR("During match & update, index out of range: tracker_idx=%d, measurement_idx=%d", tracker_idx, measurement_idx);
        }
    }

    // Handle unmatched detections
    std::vector<int> unmatched = umap::match_utils::findUnmatchedDetections(matched_pairs, measurements.size());
    for (int idx : unmatched) {
        if (!isFiniteMeasurement(measurements[idx], bbox_sizes[idx])) {
            ROS_ERROR_THROTTLE(1.0, "[Coarse tracker] refusing to init tracker from non-finite measurement %d", idx);
            continue;
        }
        onboardDetector::ObstacleTracker new_tracker;
        std::string uuid = umap::generate_uuid();
        new_tracker.init(measurements[idx], bbox_sizes[idx], current_time, nextAvailableTrackerId, uuid);
        trackers_.push_back(new_tracker);
        
        // Add new mapping
        id_mapping_[nextAvailableTrackerId] = idx;
        ROS_INFO_STREAM("[Coarse tracker]" << "Creating new tracker for unmatched detection " << idx 
            << " with tracker id: " << nextAvailableTrackerId);

        // Initialize last observed caches for the new tracker
        if (idx < curr_masks_.size()) {
            last_mask_by_tracker_[nextAvailableTrackerId] = curr_masks_[idx];
        }
        if (idx < curr_boxes2D_raw_.x.size()) {
            last_box2d_by_tracker_[nextAvailableTrackerId] = {
                curr_boxes2D_raw_.x[idx],
                curr_boxes2D_raw_.y[idx],
                curr_boxes2D_raw_.width[idx],
                curr_boxes2D_raw_.height[idx]
            };
        }
        last_size3d_by_tracker_[nextAvailableTrackerId] = bbox_sizes[idx];
        nextAvailableTrackerId++;
    }

    // Update dynamic status for every tracker exactly once per frame, and
    // predict unmatched dynamic trackers for dimension expansion.
    // Visualization reads `tracker.isDynamic` without calling isDynamicTracker()
    // again, to avoid corrupting isDynamicHistory with duplicate samples.
    std::vector<bool> is_tracker_matched(trackers_.size(), false);
    for (const auto& match : matched_pairs) {
        if (match.first >= 0 && match.first < static_cast<int>(trackers_.size())) {
            is_tracker_matched[match.first] = true;
        }
    }
    for (size_t i = 0; i < trackers_.size(); ++i) {
        (void)isDynamicTracker(trackers_[i]);
        if (is_tracker_matched[i]) continue;
        double time_since_update = current_time - trackers_[i].getLastUpdateTime();
        if (time_since_update <= params_.UNMATCHED_PUBLISH_TIMEOUT && trackers_[i].isDynamic) {
            trackers_[i].predictOnlyWithDimensionExpansion(current_time, params_.EXPANSION_RATE);
            trackers_[i].was_dimension_expanded = true;

            ROS_DEBUG_NAMED("Coarse tracker", "[Dimension Expansion] Tracker ID %d: time_since_update=%.3fs, dimensions=[%.3f, %.3f, %.3f]",
                           trackers_[i].track_id, time_since_update,
                           trackers_[i].box_dimensions.x(), trackers_[i].box_dimensions.y(), trackers_[i].box_dimensions.z());
        }
    }

    // Print current mapping
    ROS_DEBUG_NAMED("Coarse tracker", "[Tracker] ID Mapping Status:");
    for (const auto& mapping : id_mapping_) {
        ROS_DEBUG_NAMED("Coarse tracker", "[Tracker] Tracker ID %d -> Measurement ID %d", mapping.first, mapping.second);
    }

    // Update tracker lifecycles (also prune per-tracker auxiliary caches for
    // every tracker that gets removed).
    umap::match_utils::removeStaleTrackers(
        trackers_, current_time, params_.MAX_TRACKER_AGE,
        [this](int dead_id) {
            ROS_WARN_STREAM("[Coarse tracker] Removing stale tracker ID: " << dead_id);
            last_mask_by_tracker_.erase(dead_id);
            last_box2d_by_tracker_.erase(dead_id);
            last_size3d_by_tracker_.erase(dead_id);
            tracker_paths_.erase(dead_id);
        });

}
