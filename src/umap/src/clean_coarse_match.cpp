/// @brief First filter, to filter out the dynamic obstacles

#include <umap/coarse_match.h> 
#include <umap/utils.h>
#include <munkres/munkres.h> // Include Munkres header

using namespace std::chrono;

std::map<int, std::vector<int>> invalid_match_mapping_; // For storing invalid matches(明显不匹配的tracker idx - detection idxs对)

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

    // Visualization param
    nh_.param("tracker_node/COARSE_VIS_FRAME", params_.COARSE_VIS_FRAME, std::string("camera_color_optical_frame"));
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

    // // 指数平滑速度历史
    // std::vector<double> smoothed_velocities;
    // if (!tracker.history_vel.empty()) {
    //     smoothed_velocities.push_back(tracker.history_vel[0].norm());
    //     const double ALPHA = 0.2; // 指数移动平均平滑因子
    //     for (size_t i = 1; i < tracker.history_vel.size(); ++i) {
    //         double vel_norm = tracker.history_vel[i].norm();
    //         smoothed_velocities.push_back(ALPHA * vel_norm + (1 - ALPHA) * smoothed_velocities.back());
    //     }
    // }

    // // 基于平滑速度的持续性检查
    // bool is_smoothed_velocity_sustained = false;
    // if (smoothed_velocities.size() >= MIN_CONSECUTIVE_FRAMES) {
    //     int consecutive_count = 0;
    //     for (size_t i = smoothed_velocities.size() - MIN_CONSECUTIVE_FRAMES; i < smoothed_velocities.size(); ++i) {
    //         if (smoothed_velocities[i] > SPEED_THRESHOLD) {
    //             consecutive_count++;
    //         } else {
    //             consecutive_count = 0;
    //         }
    //     }
    //     is_smoothed_velocity_sustained = (consecutive_count >= MIN_CONSECUTIVE_FRAMES);
    // }

    // // 3. 计算速度差
    // for (auto it = tracker.history_vel.begin(); it != tracker.history_vel.end(); ++it) {
    //     if (it != tracker.history_vel.begin()) {
    //         auto prev_it = std::prev(it);
    //         double velocity_diff = (*it - *prev_it).norm();
    //         // ROS_INFO_STREAM("[Tracker] Velocity difference: " << velocity_diff);
    //     }
    // }

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


    // Determine the final dynamic status based on consecutive frames
    // // 1. 用STATUS_HISTORY投票确定
    // int dynamic_count = 0;
    // for (size_t i = 0; i < tracker.isDynamicHistory.size(); ++i) {
    //     if (tracker.isDynamicHistory[i]) {
    //         dynamic_count++;
    //     }
    // }
    // // If the dynamic status is consistent for the required number of frames, finalize it
    // if (dynamic_count >= params_.MIN_CONSECUTIVE_FRAMES_FOR_STATUS_CHANGE) {
    //     tracker.isDynamic = true;
    // } else if (tracker.isDynamicHistory.size() >= params_.MIN_CONSECUTIVE_FRAMES_FOR_STATUS_CHANGE &&
    //            dynamic_count == 0) {
    //     tracker.isDynamic = false;
    // }

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

/// @brief Get Munkres cost matrix between trackers and measurements.
/// @param trackers
/// @param measurements
Eigen::MatrixXd trackerNode::getCost_matrix(vector<onboardDetector::ObstacleTracker>& trackers,
        const vector<Eigen::Vector3d>& measurements, const vector<Eigen::Vector3d>& bbox_sizes,
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
                
                // 1. IoU cost
                double iou = 0.0;

                // Calculate 3D boundaries for tracker and detection
                std::array<std::pair<double, double>, 3> tracker_bounds, detection_bounds;
                for (int k = 0; k < 3; ++k) {
                    tracker_bounds[k] = {
                        center_of_tracker[k] - size_tracker[k] / 2.0,
                        center_of_tracker[k] + size_tracker[k] / 2.0
                    };
                    detection_bounds[k] = {
                        center_of_detection[k] - size_detection[k] / 2.0,
                        center_of_detection[k] + size_detection[k] / 2.0
                    };
                }

                // Calculate intersection volume
                // ROS_INFO_STREAM("\033[37m[Coarse IOU] Tracker " << i << " vs Detection " << j << "\033[0m");
                // ROS_INFO_STREAM("  Tracker bounds: X[" << tracker_bounds[0].first << ", " << tracker_bounds[0].second << "], "
                                // << "Y[" << tracker_bounds[1].first << ", " << tracker_bounds[1].second << "], "
                                // << "Z[" << tracker_bounds[2].first << ", " << tracker_bounds[2].second << "]");
                // ROS_INFO_STREAM("  Detection bounds: X[" << detection_bounds[0].first << ", " << detection_bounds[0].second << "], "
                                // << "Y[" << detection_bounds[1].first << ", " << detection_bounds[1].second << "], "
                                // << "Z[" << detection_bounds[2].first << ", " << detection_bounds[2].second << "]");
                double intersection_volume = 1.0;
                bool has_overlap = true;
                for (int k = 0; k < 3; ++k) {
                    double min_bound = std::max(tracker_bounds[k].first, detection_bounds[k].first);
                    double max_bound = std::min(tracker_bounds[k].second, detection_bounds[k].second);
                    double overlap = std::max(0.0, max_bound - min_bound);
                    if (overlap == 0.0) {
                        has_overlap = false;
                        break;
                    }
                    intersection_volume *= overlap;
                }
                if (!has_overlap) {
                    intersection_volume = 0.0;
                }

                // Calculate union volume
                double tracker_volume = size_tracker[0] * size_tracker[1] * size_tracker[2];
                double detection_volume = size_detection[0] * size_detection[1] * size_detection[2];
                double union_volume = tracker_volume + detection_volume - intersection_volume;

                // Compute IoU
                if (union_volume > 0.0) {
                    iou = intersection_volume / union_volume;
                }
                double iou_cost = 1.0 - iou;
                
                // 2. Distance cost (Euclidean)
                double distance = (center_of_tracker - center_of_detection).norm();
                
                if (iou < iou_threshold) {
                    iou_cost = 999999.0;
                    invalid_match_mapping_[i].push_back(j); // 记录无效匹配(tracker_idx, detection_idx)
                    // ROS_INFO("[Coarse tracker] Invalid match: T[%d] vs D[%d] - IoU: %.3f < threshold: %.3f", i, j, iou, iou_threshold);
                }else if (distance > dist_threshold) {
                    iou_cost = 999999.0;
                    invalid_match_mapping_[i].push_back(j);
                    // ROS_INFO("[Coarse tracker] Invalid match: T[%d] vs D[%d] - Distance: %.3f > threshold: %.3f", i, j, distance, dist_threshold);
                }

                cost_matrix(i, j) = iou_weight * iou_cost + dist_weight * distance;

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


/// @brief Find unmatched detection indices
/// @param matches Matched pairs from association
/// @param num_detections Total number of detections
/// @return Vector of unmatched detection indices in measurements
std::vector<int> findUnmatchedDetections(const std::vector<std::pair<int, int>>& matches, const int num_detections) {
    std::vector<bool> matched(num_detections, false);
    std::vector<int> unmatched;
    
    // Mark matched detections
    for(const auto& match : matches) {
        matched[match.second] = true;
    }
    
    // Collect unmatched detections
    for(int i = 0; i < num_detections; ++i) {
        if(!matched[i]) {
            unmatched.push_back(i);
            ROS_INFO_STREAM("Unmatched detection at measurement index: " << i);
        }
    }
    return unmatched;
}

/// @brief Update tracker lifecycle and remove stale trackers
/// @param trackers Vector of trackers
/// @param current_time Current timestamp
void updateTrackerLifecycles(std::vector<onboardDetector::ObstacleTracker>& trackers, double current_time, const double max_age) {
    if (trackers.empty() || max_age <= 0.0 ) {
        return;
    }

    auto it = trackers.begin();
    while (it != trackers.end()) {
        double time_since_update = current_time - it->getLastUpdateTime();
        if (time_since_update > max_age) {
            ROS_ERROR_STREAM("Removing stale tracker ID: " << it->track_id 
                           << " (age: " << time_since_update << "s)");
            it = trackers.erase(it);
        } else {
            ++it;
        }
    }
}

/// @brief Associates trackers with detections using the Munkres algorithm.
/// @param cost_matrix 
/// @return A vector of pairs representing the associations (tracker index, detection index)返回的是trackers_中的索引！.
std::vector<std::pair<int, int>> trackerNode::association(const Eigen::MatrixXd& cost_matrix) {

    if (cost_matrix.rows() == 0 || cost_matrix.cols() == 0) {
        return {};
    }
    if (cost_matrix.rows() == 1 && cost_matrix.cols() == 1 && cost_matrix(0, 0) > 1) {
        // No match! Too different
        return {};
    }
    
    Matrix<double> munkres_matrix(cost_matrix.rows(), cost_matrix.cols());
    
    // Convert Eigen matrix to Munkres matrix format
    for(size_t i = 0; i < cost_matrix.rows(); ++i) {
        for(size_t j = 0; j < cost_matrix.cols(); ++j) {
            munkres_matrix(i,j) = cost_matrix(i,j);
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
    double current_time = ros::Time::now().toSec(); // system time
    // double current_time = curr_boxes3D_raw_.header.stamp.toSec(); // sensor time

    // updateTrackerLifecycles(trackers_, current_time);

    // Clear previous mapping
    id_mapping_.clear();

    ROS_DEBUG_NAMED("Coarse tracker", "[Tracker] Processing - Measurements: %zu, BBox sizes: %zu", measurements.size(), bbox_sizes.size());

    // If no trackers exist yet, initialize them
    if (trackers_.empty()) {
        for (size_t i = 0; i < measurements.size(); i++) {
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
    std::vector<int> unmatched = findUnmatchedDetections(matched_pairs, measurements.size());
    for (int idx : unmatched) {
        onboardDetector::ObstacleTracker new_tracker;
        std::string uuid = umap::generate_uuid();
        new_tracker.init(measurements[idx], bbox_sizes[idx], current_time, nextAvailableTrackerId, uuid);
        new_tracker.update(measurements[idx], bbox_sizes[idx], current_time, params_.HISTORY_SIZE);
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

    // Predict for unmatched dynamic trackers so their KF state is up-to-date
    if (!trackers_.empty()) {
        std::vector<bool> is_tracker_matched(trackers_.size(), false);
        for (const auto& match : matched_pairs) {
            if (match.first >= 0 && match.first < static_cast<int>(trackers_.size())) {
                is_tracker_matched[match.first] = true;
            }
        }
        for (size_t i = 0; i < trackers_.size(); ++i) {
            if (is_tracker_matched[i]) continue;
            // Only continue prediction/publishing for dynamic trackers that haven't timed out
            double time_since_update = current_time - trackers_[i].getLastUpdateTime();
            if (time_since_update <= params_.UNMATCHED_PUBLISH_TIMEOUT) {
                // Update dynamic status and then predict
                (void)isDynamicTracker(trackers_[i]);
                if (trackers_[i].isDynamic) {
                    // Use dimension expansion prediction for dynamic trackers
                    trackers_[i].predictOnlyWithDimensionExpansion(current_time, params_.EXPANSION_RATE); // 30% expansion per second
                    trackers_[i].was_dimension_expanded = true; // Mark as dimension expanded
                    
                    ROS_DEBUG_NAMED("Coarse tracker", "[Dimension Expansion] Tracker ID %d: time_since_update=%.3fs, dimensions=[%.3f, %.3f, %.3f]", 
                                   trackers_[i].track_id, time_since_update, 
                                   trackers_[i].box_dimensions.x(), trackers_[i].box_dimensions.y(), trackers_[i].box_dimensions.z());
                }
            }
        }
    }

    // Print current mapping
    ROS_DEBUG_NAMED("Coarse tracker", "[Tracker] ID Mapping Status:");
    for (const auto& mapping : id_mapping_) {
        ROS_DEBUG_NAMED("Coarse tracker", "[Tracker] Tracker ID %d -> Measurement ID %d", mapping.first, mapping.second);
    }

    // Update tracker lifecycles
    updateTrackerLifecycles(trackers_, current_time, params_.MAX_TRACKER_AGE);

}

int main(int argc, char** argv) {
    ros::init(argc, argv, "coarse_match_node");
    ros::NodeHandle nh;
    trackerNode tracker(nh);
    ROS_INFO("[Coarse Match] Node is running");

    ros::spin();

    return 0;
}