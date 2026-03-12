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
    
    // Set position from tracker state
    auto state = trackers_[tracker_idx].getState();
    marker.pose.position.x = state.position.x();
    marker.pose.position.y = state.position.y();
    marker.pose.position.z = state.position.z();
    
    // Add bounding box if measurement data is available
    if (measurement_id < static_cast<int>(curr_sizes_.size())) {
        float width = curr_sizes_[measurement_id][0];
        float height = curr_sizes_[measurement_id][1];
        float depth = curr_sizes_[measurement_id][2];
        
        auto corners = createBoundingBoxCorners(width, height, depth);
        addBoundingBoxEdges(marker, corners);
    }
    
    // Set visual properties
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
            // Delete obstacle marker
            visualization_msgs::Marker delete_marker;
            delete_marker.header.frame_id = rematch_params_.REMATCH_VIS_FRAME;
            delete_marker.header.stamp = timestamp;
            delete_marker.ns = MARKER_NS_OBSTACLES;
            delete_marker.id = old_id;
            delete_marker.action = visualization_msgs::Marker::DELETE;
            marker_array.markers.push_back(delete_marker);
            
            // Delete text marker
            visualization_msgs::Marker delete_text_marker = delete_marker;
            delete_text_marker.ns = MARKER_NS_TEXT;
            text_marker_array.markers.push_back(delete_text_marker);
            
            // **NEW**: Delete path marker
            visualization_msgs::Marker delete_path_marker = delete_marker;
            delete_path_marker.ns = MARKER_NS_PATHS;
            path_marker_array.markers.push_back(delete_path_marker);
            
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
    std::set<int> current_tracker_ids;
    visualization_msgs::MarkerArray marker_array;
    visualization_msgs::MarkerArray text_marker_array;
    visualization_msgs::MarkerArray path_marker_array;  // **NEW**: Separate array for paths
    ros::Time current_time = curr_obstacles_raw_.header.stamp;

    ROS_DEBUG_NAMED("visualization", "[Visualization] Publishing visualization for %zu trackers", trackers_.size());

    // Process each matched tracker
    for (const auto& mapping : id_mapping_) {
        int tracker_id = mapping.first;
        int measurement_id = mapping.second;
        
        // Find the tracker index in trackers_ list
        int tracker_idx = findTrackerIndexById(tracker_id);
        if (tracker_idx < 0) {
            ROS_DEBUG_NAMED("visualization", "[Visualization] Tracker ID %d not found, skipping", tracker_id);
            continue;
        }
        
        current_tracker_ids.insert(tracker_id);
        
        // Create and add obstacle bounding box marker
        auto marker = createObstacleMarker(tracker_id, tracker_idx, measurement_id, current_time);
        marker_array.markers.push_back(marker);
        
        // Create and add text marker
        auto text_marker = createTextMarker(tracker_id, tracker_idx, marker.pose.position, current_time);
        text_marker_array.markers.push_back(text_marker);
        
        // Update tracker path (store history)
        updateTrackerPath(tracker_id, marker.pose.position, marker.header);
        
        // **NEW**: Create and add path marker
        auto path_marker = createPathMarker(tracker_id, current_time);
        path_marker_array.markers.push_back(path_marker);
        
        ROS_DEBUG_NAMED("visualization", "[Visualization] Processing tracker ID %d (idx %d) at position [%.3f, %.3f, %.3f]",
                       tracker_id, tracker_idx, marker.pose.position.x, marker.pose.position.y, marker.pose.position.z);
    }

    // Handle empty marker array case
    if (marker_array.markers.empty()) {
        visualization_msgs::Marker clear_marker;
        clear_marker.header.frame_id = rematch_params_.REMATCH_VIS_FRAME;
        clear_marker.header.stamp = current_time;
        clear_marker.ns = MARKER_NS_OBSTACLES;
        clear_marker.action = visualization_msgs::Marker::DELETEALL;
        marker_array.markers.push_back(clear_marker);
        
        clear_marker.ns = MARKER_NS_TEXT;
        text_marker_array.markers.push_back(clear_marker);
        
        clear_marker.ns = MARKER_NS_PATHS;
        path_marker_array.markers.push_back(clear_marker);
    }
    
    // Delete markers for trackers that are no longer active
    deleteRemovedTrackerMarkers(current_tracker_ids, marker_array, text_marker_array, path_marker_array, current_time);
    last_tracker_ids_ = current_tracker_ids;
    
    // Clean up stale tracker paths
    cleanupStalePaths();

    // Publish marker arrays
    marker_pub_.publish(marker_array);
    text_marker_pub_.publish(text_marker_array);
    marker_pub_.publish(path_marker_array);  // **NEW**: Publish all path markers at once
    
    ROS_DEBUG_NAMED("visualization", "[Visualization] Published %zu markers, %zu text markers, and %zu path markers", 
                    marker_array.markers.size(), text_marker_array.markers.size(), path_marker_array.markers.size());
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