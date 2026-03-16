#include <umap/rematch.h>

namespace {
    // Visualization constants (non-member variables only)
    const char* MARKER_NS_OBSTACLES = "matched_obstacles";
    const char* MARKER_NS_TEXT = "tracked_obstacles_text";
    const char* MARKER_NS_PATHS = "tracker_paths";
    
    const float LINE_WIDTH = 0.01f;
    const float TEXT_HEIGHT_OFFSET = 0.5f;
    const float TEXT_SCALE = 0.1f;
    
    // Color for matched obstacles (cyan)
    const float COLOR_R = 0.2f;
    const float COLOR_G = 0.7f;
    const float COLOR_B = 0.6f;
    
    // Color for text (green)
    const float TEXT_COLOR_R = 0.0f;
    const float TEXT_COLOR_G = 1.0f;
    const float TEXT_COLOR_B = 0.0f;

    const float MARKER_LIFETIME_SEC = 0.2f;
}

// Helper function: Find tracker array index by tracker ID
int rematchNode::findTrackerIndexById(int tracker_id) const {
    for (size_t i = 0; i < trackers_.size(); ++i) {
        if (trackers_[i].track_id == tracker_id) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

// Helper function: Create bounding box corners
std::vector<geometry_msgs::Point> rematchNode::createBoundingBoxCorners(
    float width, float height, float depth) const {
    
    std::vector<geometry_msgs::Point> corners(8);
    float dx = width / 2.0f;
    float dy = height / 2.0f;
    float dz = depth / 2.0f;
    
    // Bottom face
    corners[0].x = -dx; corners[0].y = -dy; corners[0].z = -dz;
    corners[1].x =  dx; corners[1].y = -dy; corners[1].z = -dz;
    corners[2].x =  dx; corners[2].y =  dy; corners[2].z = -dz;
    corners[3].x = -dx; corners[3].y =  dy; corners[3].z = -dz;
    
    // Top face
    corners[4].x = -dx; corners[4].y = -dy; corners[4].z = dz;
    corners[5].x =  dx; corners[5].y = -dy; corners[5].z = dz;
    corners[6].x =  dx; corners[6].y =  dy; corners[6].z = dz;
    corners[7].x = -dx; corners[7].y =  dy; corners[7].z = dz;
    
    return corners;
}

// Helper function: Add box edges to marker
void rematchNode::addBoundingBoxEdges(
    visualization_msgs::Marker& marker,
    const std::vector<geometry_msgs::Point>& corners) const {
    
    // Bottom face edges
    marker.points.push_back(corners[0]); marker.points.push_back(corners[1]);
    marker.points.push_back(corners[1]); marker.points.push_back(corners[2]);
    marker.points.push_back(corners[2]); marker.points.push_back(corners[3]);
    marker.points.push_back(corners[3]); marker.points.push_back(corners[0]);
    
    // Top face edges
    marker.points.push_back(corners[4]); marker.points.push_back(corners[5]);
    marker.points.push_back(corners[5]); marker.points.push_back(corners[6]);
    marker.points.push_back(corners[6]); marker.points.push_back(corners[7]);
    marker.points.push_back(corners[7]); marker.points.push_back(corners[4]);
    
    // Vertical edges
    marker.points.push_back(corners[0]); marker.points.push_back(corners[4]);
    marker.points.push_back(corners[1]); marker.points.push_back(corners[5]);
    marker.points.push_back(corners[2]); marker.points.push_back(corners[6]);
    marker.points.push_back(corners[3]); marker.points.push_back(corners[7]);
}

// Helper function: Calculate average movement from history
double rematchNode::calculateAverageMovement(const std::vector<Eigen::Vector3d>& history) const {
    if (history.size() < 2) {
        return 0.0;
    }
    
    double total_movement = 0.0;
    for (size_t i = 1; i < history.size(); ++i) {
        total_movement += (history[i] - history[i - 1]).norm();
    }
    return total_movement / static_cast<double>(history.size() - 1);
}

// Helper function: Create obstacle bounding box marker
visualization_msgs::Marker rematchNode::createObstacleMarker(
    int tracker_id, 
    int tracker_idx,
    int measurement_id,
    const ros::Time& timestamp) const {
    
    visualization_msgs::Marker marker;
    marker.header.frame_id = rematch_params_.REMATCH_VIS_FRAME;
    marker.header.stamp = timestamp;
    marker.ns = MARKER_NS_OBSTACLES;
    marker.id = tracker_id;
    marker.type = visualization_msgs::Marker::LINE_LIST;
    marker.action = visualization_msgs::Marker::ADD;
    marker.lifetime = ros::Duration(MARKER_LIFETIME_SEC);
    
    auto state = trackers_[tracker_idx].getState();

    // 1. position: TRACKING_FRAME -> REMATCH_VIS_FRAME
    Eigen::Vector3d vis_position = state.position;
    transformPointBetweenFrames(state.position,
                                rematch_params_.TRACKING_FRAME,
                                rematch_params_.REMATCH_VIS_FRAME,
                                timestamp,
                                vis_position);
    marker.pose.position.x = vis_position.x();
    marker.pose.position.y = vis_position.y();
    marker.pose.position.z = vis_position.z();

    marker.pose.orientation.x = 0.0;
    marker.pose.orientation.y = 0.0;
    marker.pose.orientation.z = 0.0;
    marker.pose.orientation.w = 1.0;

    // 2. size: TRACKING_FRAME -> REMATCH_VIS_FRAME
    Eigen::Vector3d vis_dimensions = trackers_[tracker_idx].box_dimensions;
    transformSizeBetweenFrames(trackers_[tracker_idx].box_dimensions,
                               rematch_params_.TRACKING_FRAME,
                               rematch_params_.REMATCH_VIS_FRAME,
                               timestamp,
                               vis_dimensions);

    float extent_x = vis_dimensions[0];
    float extent_y = vis_dimensions[1];
    float extent_z = vis_dimensions[2];

    auto corners = createBoundingBoxCorners(extent_x, extent_y, extent_z);
    addBoundingBoxEdges(marker, corners);

    marker.scale.x = LINE_WIDTH;
    marker.color.r = COLOR_R;
    marker.color.g = COLOR_G;
    marker.color.b = COLOR_B;
    marker.color.a = 1.0;

    return marker;
}

// Helper function: Create text marker with tracker information
visualization_msgs::Marker rematchNode::createTextMarker(
    int tracker_id,
    int tracker_idx,
    const geometry_msgs::Point& position,
    const ros::Time& timestamp) const {
    
    visualization_msgs::Marker text_marker;
    text_marker.header.frame_id = rematch_params_.REMATCH_VIS_FRAME;
    text_marker.header.stamp = timestamp;
    text_marker.ns = MARKER_NS_TEXT;
    text_marker.id = tracker_id;
    text_marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
    text_marker.action = visualization_msgs::Marker::ADD;
    text_marker.lifetime = ros::Duration(MARKER_LIFETIME_SEC);
    
    // Position above the obstacle
    text_marker.pose.position.x = position.x;
    text_marker.pose.position.y = position.y;
    text_marker.pose.position.z = position.z + TEXT_HEIGHT_OFFSET;
    text_marker.scale.z = TEXT_SCALE;
    
    // Build text content
    auto state = trackers_[tracker_idx].getState();
    double avg_movement = calculateAverageMovement(trackers_[tracker_idx].history_pos);
    
    std::stringstream ss;
    ss << "ID:" << tracker_id
       << "\nv:" << std::fixed << std::setprecision(2) << state.velocity.norm() << "m/s"
       << "\navg_mv:" << std::setprecision(2) << avg_movement << "m";
    text_marker.text = ss.str();
    
    // Set text color
    text_marker.color.r = TEXT_COLOR_R;
    text_marker.color.g = TEXT_COLOR_G;
    text_marker.color.b = TEXT_COLOR_B;
    text_marker.color.a = 1.0;
    
    return text_marker;
}

// Helper function: Update tracker path
void rematchNode::updateTrackerPath(
    int tracker_id,
    const geometry_msgs::Point& position,
    const std_msgs::Header& header) {
    
    geometry_msgs::PoseStamped pose;
    pose.header = header;
    pose.pose.position = position;
    
    // Initialize path if it doesn't exist
    if (tracker_paths_.find(tracker_id) == tracker_paths_.end()) {
        tracker_paths_[tracker_id] = std::vector<geometry_msgs::PoseStamped>();
        ROS_DEBUG_NAMED("visualization", "[Path] Initializing new path for tracker ID: %d", tracker_id);
    }
    
    // Add new pose to path
    tracker_paths_[tracker_id].push_back(pose);
    
    // Maintain maximum path length
    if (tracker_paths_[tracker_id].size() > max_path_length_) {
        tracker_paths_[tracker_id].erase(tracker_paths_[tracker_id].begin());
    }
    
    ROS_DEBUG_NAMED("visualization", "[Path] Updated path for tracker ID %d, path length: %zu, position: [%.3f, %.3f, %.3f]", 
                    tracker_id, tracker_paths_[tracker_id].size(), position.x, position.y, position.z);
}

// Helper function: Create path marker for a single tracker
visualization_msgs::Marker rematchNode::createPathMarker(
    int tracker_id,
    const ros::Time& timestamp) const {
    
    visualization_msgs::Marker path_marker;
    path_marker.header.frame_id = rematch_params_.REMATCH_VIS_FRAME;
    path_marker.header.stamp = timestamp;
    path_marker.ns = MARKER_NS_PATHS;
    path_marker.id = tracker_id;
    path_marker.type = visualization_msgs::Marker::LINE_STRIP;
    path_marker.action = visualization_msgs::Marker::ADD;
    path_marker.lifetime = ros::Duration(MARKER_LIFETIME_SEC);
    
    // Check if path exists
    auto it = tracker_paths_.find(tracker_id);
    if (it == tracker_paths_.end() || it->second.empty()) {
        ROS_DEBUG_NAMED("visualization", "[Path] No path found for tracker ID: %d", tracker_id);
        path_marker.action = visualization_msgs::Marker::DELETE;
        return path_marker;
    }
    
    // Add all points in the path
    for (const auto& pose : it->second) {
        path_marker.points.push_back(pose.pose.position);
    }
    
    // Set line properties
    path_marker.scale.x = LINE_WIDTH * 2.0f;  // Thicker line for path
    
    // Use color based on tracker ID for distinction
    float hue = static_cast<float>(tracker_id % 360);
    float r, g, b;
    hsvToRgb(hue, 0.8f, 0.9f, r, g, b);
    
    path_marker.color.r = r;
    path_marker.color.g = g;
    path_marker.color.b = b;
    path_marker.color.a = 0.8;
    
    ROS_DEBUG_NAMED("visualization", "[Path] Created path marker for tracker ID %d with %zu points", 
                    tracker_id, path_marker.points.size());
    
    return path_marker;
}

// Helper function: HSV to RGB conversion for color generation
void rematchNode::hsvToRgb(float h, float s, float v, float& r, float& g, float& b) const {
    float c = v * s;
    float x = c * (1.0f - std::abs(std::fmod(h / 60.0f, 2.0f) - 1.0f));
    float m = v - c;
    
    if (h < 60) {
        r = c; g = x; b = 0;
    } else if (h < 120) {
        r = x; g = c; b = 0;
    } else if (h < 180) {
        r = 0; g = c; b = x;
    } else if (h < 240) {
        r = 0; g = x; b = c;
    } else if (h < 300) {
        r = x; g = 0; b = c;
    } else {
        r = c; g = 0; b = x;
    }
    
    r += m; g += m; b += m;
}

// Helper function: Delete markers for removed trackers
void rematchNode::deleteRemovedTrackerMarkers(
    const std::set<int>& current_tracker_ids,
    visualization_msgs::MarkerArray& marker_array,
    visualization_msgs::MarkerArray& text_marker_array,
    visualization_msgs::MarkerArray& path_marker_array,
    const ros::Time& timestamp) {
    
    for (int old_id : last_tracker_ids_) {
        if (current_tracker_ids.find(old_id) == current_tracker_ids.end()) {
            auto makeDeleteMarker = [&](const std::string& ns) {
                visualization_msgs::Marker m;
                m.header.frame_id = rematch_params_.REMATCH_VIS_FRAME;
                m.header.stamp = timestamp;
                m.ns = ns;
                m.id = old_id;
                m.action = visualization_msgs::Marker::DELETE;
                return m;
            };
            marker_array.markers.push_back(makeDeleteMarker(MARKER_NS_OBSTACLES));
            text_marker_array.markers.push_back(makeDeleteMarker(MARKER_NS_TEXT));
            path_marker_array.markers.push_back(makeDeleteMarker(MARKER_NS_PATHS));

            tracker_paths_.erase(old_id);

            ROS_DEBUG_NAMED("visualization", "[Visualization] Deleting markers for tracker ID: %d", old_id);
        }
    }
}

// Helper function: Clean up stale tracker paths
void rematchNode::cleanupStalePaths() {
    for (auto it = tracker_paths_.begin(); it != tracker_paths_.end();) {
        int tracker_id = it->first;
        if (id_mapping_.find(tracker_id) == id_mapping_.end()) {
            ROS_DEBUG_NAMED("visualization", "[Path] Cleaning up stale path for tracker ID: %d", tracker_id);
            it = tracker_paths_.erase(it);
        } else {
            ++it;
        }
    }
}

// Main visualization function
void rematchNode::publishVisualization()
{
    const std::string vis_frame = rematch_params_.REMATCH_VIS_FRAME;
    const std::string output_frame = rematch_params_.REMATCH_OUTPUT_FRAME;
    std::set<int> current_tracker_ids;
    visualization_msgs::MarkerArray marker_array;
    visualization_msgs::MarkerArray text_marker_array;
    visualization_msgs::MarkerArray path_marker_array;
    ros::Time current_time = curr_obstacles_raw_.header.stamp;

    for (const auto& mapping : id_mapping_) {
        int tracker_id = mapping.first;
        int measurement_id = mapping.second;
        
        int tracker_idx = findTrackerIndexById(tracker_id);
        if (tracker_idx < 0) continue;
        
        current_tracker_ids.insert(tracker_id);
        
        auto marker = createObstacleMarker(tracker_id, tracker_idx, measurement_id, current_time);
        marker_array.markers.push_back(marker);
        
        auto text_marker = createTextMarker(tracker_id, tracker_idx, marker.pose.position, current_time);
        text_marker_array.markers.push_back(text_marker);
        
        updateTrackerPath(tracker_id, marker.pose.position, marker.header);

        // 发布 nav_msgs::Path
        nav_msgs::Path tracker_path_msg;
        tracker_path_msg.header.stamp = current_time;
        tracker_path_msg.header.frame_id = output_frame;
        for (const auto& history_pose_vis : tracker_paths_[tracker_id]) {
            Eigen::Vector3d vis_point(history_pose_vis.pose.position.x,
                                      history_pose_vis.pose.position.y,
                                      history_pose_vis.pose.position.z);
            Eigen::Vector3d output_point = vis_point;
            transformPointBetweenFrames(vis_point, vis_frame, output_frame, current_time, output_point);

            geometry_msgs::PoseStamped pose_stamped;
            pose_stamped.header = tracker_path_msg.header;
            pose_stamped.pose.position.x = output_point.x();
            pose_stamped.pose.position.y = output_point.y();
            pose_stamped.pose.position.z = output_point.z();
            pose_stamped.pose.orientation.w = 1.0;
            tracker_path_msg.poses.push_back(pose_stamped);
        }
        path_pub_.publish(tracker_path_msg);
        
        auto path_marker = createPathMarker(tracker_id, current_time);
        path_marker_array.markers.push_back(path_marker);
    }

    // 处理全空情况 —— 只负责补一个占位 marker 防止空数组，不再包含 DELETEALL
    if (marker_array.markers.empty() &&
        text_marker_array.markers.empty() &&
        path_marker_array.markers.empty() &&
        last_tracker_ids_.empty()) {
        // 真正什么都没有，直接返回
        return;
    }

    // ✅ 先执行 DELETE，此时 last_tracker_ids_ 还未更新
    deleteRemovedTrackerMarkers(
        current_tracker_ids,
        marker_array, text_marker_array, path_marker_array,
        current_time);

    // ✅ 再更新 last_tracker_ids_
    last_tracker_ids_ = current_tracker_ids;

    // 发布（即使 markers 只含 DELETE 指令也要发布）
    if (!marker_array.markers.empty())
        marker_pub_.publish(marker_array);
    if (!text_marker_array.markers.empty())
        text_marker_pub_.publish(text_marker_array);
    if (!path_marker_array.markers.empty())
        path_marker_pub_.publish(path_marker_array);
}

int main(int argc, char **argv)
{

    // Initialize ROS node
    ros::init(argc, argv, "rematch_node");
    ros::NodeHandle nh;

    // Create tracker node instance
    rematchNode rematch(nh);

    ROS_INFO("Rematch node is running...");

    // ROS spin
    ros::spin();

    return 0;
}