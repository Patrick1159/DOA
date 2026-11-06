#include <umap/coarse_match.h>

/// @brief Publish visualization information, including 3D cubes, text markers, and dynamic obstacle information (Obstacles message type).
void trackerNode::publishVisualization() {
    const std::string kFrameId = params_.COARSE_VIS_FRAME;        // RViz markers (cube/text/path)
    const std::string kOutputFrame = params_.COARSE_OUTPUT_FRAME; // obstacles_msg consumed downstream
    ros::Time current_time = curr_boxes3D_raw_.header.stamp; // keep stamp equal to image stamp

    // Track which tracker IDs produced cube markers this frame; used below to
    // emit per-ID DELETE markers for trackers that disappeared since last frame.
    std::set<int> current_published_ids;

    // Helper: resolve mapping.first (track_id) to tracker vector index and tracker_id
    auto resolveMapping = [this](int mapping_first, int& tracker_index_out, int& tracker_id_out) -> bool {
        int tracker_idx = mapping_first;      // mapping.first is a track_id
        // Check if tracker_idx matches any tracker's track_id
        bool found = false;
        for (const auto& tracker : trackers_) {
            if (tracker.track_id == tracker_idx) {
                found = true;
                break;
            }
        }
        if (!found) {
            ROS_WARN("[Visualization] Tracker ID %d not found in trackers_ list, skipping...", tracker_idx);
            return false;
        }
        // 交换回来正确的顺序
        tracker_id_out = tracker_idx; // tracker_id is the track_id
        for (size_t i = 0; i < trackers_.size(); ++i) {
            if (trackers_[i].track_id == tracker_idx) {
                tracker_index_out = static_cast<int>(i);
                break;
            }
        }
        ROS_DEBUG_NAMED("Coarse tracker", "[LOOP FOR TRACKERS] Current tracker idx(mappping.first): %d <--> tracker_id: %d", tracker_index_out, tracker_id_out);
        return true;
    };

    // 1) Publish 3D cube markers and paths
    auto publishCubesAndPaths = [this, &kFrameId, &current_time, &resolveMapping, &current_published_ids]() {
        visualization_msgs::MarkerArray marker_array;

        for (const auto& mapping : id_mapping_) {
            int tracker_idx = mapping.first;
            int measurement_id = mapping.second;
            ROS_DEBUG_NAMED("Coarse tracker", "[Visualization]Current mapping: %d to %d", tracker_idx, measurement_id);

            int tracker_index = -1, tracker_id = -1;
            if (!resolveMapping(tracker_idx, tracker_index, tracker_id)) continue;

            // 0. Check if tracker is dynamic (state updated in matchAndUpdate)
            bool is_dynamic = trackers_[tracker_index].isDynamic;
            // if (!is_dynamic) continue; // Skip static obstacles
            // if (!trackers_[tracker_index].isDynamic) continue; // Skip obstacles that are not consistently dynamic

            // 1. CUBE marker
            visualization_msgs::Marker marker;
            marker.header.frame_id = kFrameId;
            marker.header.stamp = current_time;
            marker.ns = "tracked_obstacles";
            marker.id = tracker_id;
            marker.type = visualization_msgs::Marker::CUBE;
            marker.action = visualization_msgs::Marker::ADD;

            auto state = trackers_[tracker_index].getState();

            // KF state is in COARSE_OUTPUT_FRAME (kFrameId, typically `odom`).
            // Direct copy — no axis swap. The transformation from the sensor
            // optical frame happened once at the boxes3DCallback entry.
            marker.pose.position.x = state.position.x();
            marker.pose.position.y = state.position.y();
            marker.pose.position.z = state.position.z();

            // Set marker size. The dimensions come from the sensor optical
            // frame (width=x_extent, height=y_extent, thickness=z_extent).
            // kFrameId is COARSE_VIS_FRAME, the same optical frame the state
            // and curr_sizes_ are already expressed in — direct copy, no
            // axis swap. The TF chain rotates the whole marker (position and
            // axis-aligned scale) into the Fixed Frame for display.
            if (measurement_id < curr_sizes_.size()) {
                marker.scale.x = curr_sizes_[measurement_id][0]; // optical x (width)
                marker.scale.y = curr_sizes_[measurement_id][1]; // optical y (height)
                marker.scale.z = curr_sizes_[measurement_id][2]; // optical z (thickness)
            }

            // Add velocity information to marker description
            std::stringstream ss;
            ss << "ID:" << tracker_id << " v:" << std::fixed << std::setprecision(2) 
               << state.velocity.norm() << "m/s";
            marker.text = ss.str();

            // Set color to blue for dynamic obstacles
            marker.color.r = 0.0;
            marker.color.g = 0.0;
            marker.color.b = 1.0;
            marker.color.a = 0.8;

            marker_array.markers.push_back(marker);
            current_published_ids.insert(tracker_id);

            // Path info (unchanged logic)
            geometry_msgs::PoseStamped pose;
            pose.header = marker.header;
            pose.pose.position = marker.pose.position;

            // Create or update path for this tracker (keyed by stable tracker_id, not vector index)
            if (tracker_paths_.find(tracker_id) == tracker_paths_.end()) {
                tracker_paths_[tracker_id] = std::vector<geometry_msgs::PoseStamped>();
            }

            // Add new position to path
            tracker_paths_[tracker_id].push_back(pose);

            // Limit path length
            if (tracker_paths_[tracker_id].size() > max_path_length_) {
                tracker_paths_[tracker_id].erase(tracker_paths_[tracker_id].begin());
            }

            // Create and publish path
            nav_msgs::Path path;
            path.header.frame_id = kFrameId;
            path.header.stamp = current_time;
            path.poses = tracker_paths_[tracker_id];

            // Add path visualization marker
            visualization_msgs::Marker path_marker;
            path_marker.header = marker.header;
            path_marker.ns = "tracker_paths";
            path_marker.id = tracker_id;
            path_marker.type = visualization_msgs::Marker::LINE_STRIP;
            path_marker.action = visualization_msgs::Marker::ADD;
            path_marker.scale.x = 0.03; // line width
            path_marker.color = marker.color;
            path_marker.color.a = 0.5; // more transparent than the cube

            // Convert path poses to points
            for (const auto& path_pose : tracker_paths_[tracker_id]) {
                geometry_msgs::Point point;
                point = path_pose.pose.position;
                path_marker.points.push_back(point);
            }
            
            // marker_array.markers.push_back(path_marker);
            path_pub_.publish(path);
        } // end of for loop

        // Emit per-ID DELETE markers for trackers that were published last frame
        // but not this frame (cube + text share the same ID space).
        for (int old_id : last_published_ids_) {
            if (current_published_ids.find(old_id) != current_published_ids.end()) continue;
            visualization_msgs::Marker del_cube;
            del_cube.header.frame_id = kFrameId;
            del_cube.header.stamp = current_time;
            del_cube.ns = "tracked_obstacles";
            del_cube.id = old_id;
            del_cube.action = visualization_msgs::Marker::DELETE;
            marker_array.markers.push_back(del_cube);
        }

        // If no markers at all were produced this frame, fall back to DELETEALL
        // to clean up any stragglers.
        if (marker_array.markers.empty()) {
            visualization_msgs::Marker clear_marker;
            clear_marker.header.frame_id = kFrameId;
            clear_marker.header.stamp = current_time;
            clear_marker.ns = "tracked_obstacles";
            clear_marker.action = visualization_msgs::Marker::DELETEALL;
            marker_array.markers.push_back(clear_marker);
        }

        // Clean up paths for removed trackers: key is tracker_id, so compare
        // against the set of still-alive tracker IDs (NOT id_mapping_, which
        // only contains matched-this-frame trackers and would prematurely
        // drop paths of unmatched-but-alive trackers).
        std::set<int> alive_ids;
        for (const auto& t : trackers_) alive_ids.insert(t.track_id);
        for (auto it = tracker_paths_.begin(); it != tracker_paths_.end();) {
            if (alive_ids.find(it->first) == alive_ids.end()) {
                it = tracker_paths_.erase(it);
            } else {
                ++it;
            }
        }

        // Publish marker array
        marker_pub_.publish(marker_array);
    };

    // 2) Publish text markers
    auto publishTextMarkers = [this, &kFrameId, &current_time, &resolveMapping, &current_published_ids]() {
        visualization_msgs::MarkerArray text_marker_array;

        for (const auto& mapping : id_mapping_) {
            int tracker_idx = mapping.first;
            int measurement_id = mapping.second;
            ROS_DEBUG_NAMED("Coarse tracker", "Current mapping: %d to %d", tracker_idx, measurement_id);

            int tracker_index = -1, tracker_id = -1;
            if (!resolveMapping(tracker_idx, tracker_index, tracker_id)) continue;

            // 0. Check if tracker is dynamic (state updated in matchAndUpdate)
            if (!trackers_[tracker_index].isDynamic) continue; // Skip obstacles that are not consistently dynamic

            auto state = trackers_[tracker_index].getState();

            // 2. Text marker
            visualization_msgs::Marker text_marker;
            text_marker.header.frame_id = kFrameId;
            text_marker.header.stamp = current_time;
            text_marker.ns = "tracked_obstacles_text";
            text_marker.id = tracker_id;
            text_marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
            text_marker.action = visualization_msgs::Marker::ADD;

            // position equals cube pose with a small +z offset (odom: z=up)
            text_marker.pose.position.x = state.position.x();
            text_marker.pose.position.y = state.position.y();
            text_marker.pose.position.z = state.position.z() + 0.5; // Offset above cube
            text_marker.scale.z = 0.1; // Font size

            // Calculate average movement
            double avg_movement = 0.0;
            const auto& history = trackers_[tracker_index].history_pos;
            if (!history.empty()) {
                for (const auto& movement : history) {
                    avg_movement += movement.norm();
                }
                avg_movement /= history.size();
            }

            // Update text to include average movement
            std::stringstream ss_text;
            ss_text << "ID:" << tracker_id 
                    << "\nv:" << std::fixed << std::setprecision(2) << state.velocity.norm() << "m/s"
                    << "\navg_mv:" << std::setprecision(2) << avg_movement << "m";
            text_marker.text = ss_text.str();

            text_marker.color.r = 1.0;
            text_marker.color.g = 1.0;
            text_marker.color.b = 1.0;
            text_marker.color.a = 1.0;

            text_marker_array.markers.push_back(text_marker);
        }

        // Emit per-ID DELETE markers for text markers of trackers that
        // disappeared. (Sending DELETE for an ID with no prior text marker is
        // harmless — RViz will ignore it.)
        for (int old_id : last_published_ids_) {
            if (current_published_ids.find(old_id) != current_published_ids.end()) continue;
            visualization_msgs::Marker del_text;
            del_text.header.frame_id = kFrameId;
            del_text.header.stamp = current_time;
            del_text.ns = "tracked_obstacles_text";
            del_text.id = old_id;
            del_text.action = visualization_msgs::Marker::DELETE;
            text_marker_array.markers.push_back(del_text);
        }

        if (text_marker_array.markers.empty()) {
            visualization_msgs::Marker clear_marker;
            clear_marker.header.frame_id = kFrameId;
            clear_marker.header.stamp = current_time;
            clear_marker.ns = "tracked_obstacles_text";
            clear_marker.action = visualization_msgs::Marker::DELETEALL;
            text_marker_array.markers.push_back(clear_marker);
        }

        // Publish text marker array
        text_marker_pub_.publish(text_marker_array);
    };

    // 3) Publish Obstacles message
    auto publishObstaclesMsg = [this, &kFrameId, &kOutputFrame, &current_time, &resolveMapping]() {
        umap::Obstacles obstacles_msg;

        // 3.1) First, publish matched dynamic trackers (existing behavior)
        for (const auto& mapping : id_mapping_) {
            int tracker_idx = mapping.first;
            int measurement_id = mapping.second;
            ROS_DEBUG_NAMED("Coarse tracker", "Current mapping: %d to %d", tracker_idx, measurement_id);

            int tracker_index = -1, tracker_id = -1;
            if (!resolveMapping(tracker_idx, tracker_index, tracker_id)) continue;

            // 0. Check if tracker is dynamic (state updated in matchAndUpdate)
            if (!trackers_[tracker_index].isDynamic) continue; // Skip obstacles that are not consistently dynamic

            auto state = trackers_[tracker_index].getState();

            // Construct obstacles_msg using current measurement.
            // KF state is already in OUTPUT_FRAME (kOutputFrame, typically
            // odom). Direct copy. Dimensions stay in their semantic fields
            // (msg.width = optical x-extent, height = optical y-extent,
            // thickness = optical z-extent).
            obstacles_msg.box_ids.push_back(tracker_id);
            obstacles_msg.centroid_x.push_back(state.position.x());
            obstacles_msg.centroid_y.push_back(state.position.y());
            obstacles_msg.centroid_z.push_back(state.position.z());
            obstacles_msg.width.push_back(curr_sizes_[measurement_id][0]);
            obstacles_msg.height.push_back(curr_sizes_[measurement_id][1]);
            obstacles_msg.thickness.push_back(curr_sizes_[measurement_id][2]);
            obstacles_msg.mask.push_back(curr_masks_[measurement_id]); // Add mask to obstacles_msg
            obstacles_msg.box_x.push_back(curr_boxes2D_raw_.x[measurement_id]); // x
            obstacles_msg.box_y.push_back(curr_boxes2D_raw_.y[measurement_id]); // y
            obstacles_msg.box_width.push_back(curr_boxes2D_raw_.width[measurement_id]); // width
            obstacles_msg.box_height.push_back(curr_boxes2D_raw_.height[measurement_id]); // height

            // Set observed field - true for matched trackers (they have current measurements)
            obstacles_msg.observed.push_back(true);
            obstacles_msg.confidence.push_back(1.0f);  // confirmed dynamic

            // add obstacles path to obstacles_msg
            nav_msgs::Path his_path;
            his_path.header.frame_id = kFrameId;
            his_path.header.stamp = current_time;
            for(const auto& pos : state.history_pos)
            {
                geometry_msgs::PoseStamped pose_stamped;

                pose_stamped.header.frame_id = kFrameId;
                pose_stamped.header.stamp = current_time;
                pose_stamped.pose.position.x = pos.x();
                pose_stamped.pose.position.y = pos.y();
                pose_stamped.pose.position.z = pos.z();

                // default identity orientation
                pose_stamped.pose.orientation.x = 0.0;
                pose_stamped.pose.orientation.y = 0.0;
                pose_stamped.pose.orientation.z = 0.0;
                pose_stamped.pose.orientation.w = 1.0;
                
                his_path.poses.push_back(pose_stamped);
            }
            obstacles_msg.obstacles_history_pos.push_back(his_path);
        }

        // 3.2) Then, also publish unmatched dynamic trackers (predictive publishing up to MAX_TRACKER_AGE)
        for (size_t i = 0; i < trackers_.size(); ++i) {
            const auto& tracker = trackers_[i];
            int tracker_id = tracker.track_id;
            // Skip if matched this frame
            if (id_mapping_.find(tracker_id) != id_mapping_.end()) continue;

            double time_since_update = current_time.toSec() - tracker.getLastUpdateTime();
            if (time_since_update > params_.UNMATCHED_PUBLISH_TIMEOUT) continue; // stop after unmatched publish timeout

            // Check dynamic status (state already refreshed in matchAndUpdate)
            if (!tracker.isDynamic) continue;

            auto state = tracker.getState();

            // Use last observed measurement data for size/mask/2D box
            auto mask_it = last_mask_by_tracker_.find(tracker_id);
            auto box2d_it = last_box2d_by_tracker_.find(tracker_id);
            auto size3d_it = last_size3d_by_tracker_.find(tracker_id);

            if (box2d_it == last_box2d_by_tracker_.end() || size3d_it == last_size3d_by_tracker_.end() || mask_it == last_mask_by_tracker_.end()) {
                ROS_DEBUG_NAMED("Coarse tracker", "Missing cached data for unmatched tracker %d, skipping predictive publish", tracker_id);
                continue;
            }

            obstacles_msg.box_ids.push_back(tracker_id);
            obstacles_msg.centroid_x.push_back(state.position.x());
            obstacles_msg.centroid_y.push_back(state.position.y());
            obstacles_msg.centroid_z.push_back(state.position.z());
            // Dimensions stay in their semantic fields (last observed sz from cache).
            const Eigen::Vector3d& sz = size3d_it->second; // [width, height, thickness]
            obstacles_msg.width.push_back(sz[0]);
            obstacles_msg.height.push_back(sz[1]);
            obstacles_msg.thickness.push_back(sz[2]);
            obstacles_msg.mask.push_back(mask_it->second);
            const auto& b2d = box2d_it->second;
            obstacles_msg.box_x.push_back(b2d[0]);
            obstacles_msg.box_y.push_back(b2d[1]);
            obstacles_msg.box_width.push_back(b2d[2]);
            obstacles_msg.box_height.push_back(b2d[3]);

            // Set observed field - false for unmatched trackers that used dimension expansion
            obstacles_msg.observed.push_back(!tracker.was_dimension_expanded);
            obstacles_msg.confidence.push_back(1.0f);  // confirmed dynamic, predictive publish

            // add obstacles path to obstacles_msg (use existing history_pos)
            nav_msgs::Path his_path;
            his_path.header.frame_id = kFrameId;
            his_path.header.stamp = current_time;
            for(const auto& pos : state.history_pos)
            {
                geometry_msgs::PoseStamped pose_stamped;

                pose_stamped.header.frame_id = kFrameId;
                pose_stamped.header.stamp = current_time;
                pose_stamped.pose.position.x = pos.x();
                pose_stamped.pose.position.y = pos.y();
                pose_stamped.pose.position.z = pos.z();

                // default identity orientation
                pose_stamped.pose.orientation.x = 0.0;
                pose_stamped.pose.orientation.y = 0.0;
                pose_stamped.pose.orientation.z = 0.0;
                pose_stamped.pose.orientation.w = 1.0;
                
                his_path.poses.push_back(pose_stamped);
            }
            obstacles_msg.obstacles_history_pos.push_back(his_path);
        }

        // 3.3) Candidate early-publish: trackers that are matched this frame
        // and already have a non-trivial KF velocity, but have not yet passed
        // the SUSTAINED hysteresis (tracker.isDynamic == false). These are
        // emitted with confidence = 0.5 so downstream may apply tighter
        // safety margins without waiting several frames. Skips trackers that
        // are already published above.
        std::set<int> already_pub(obstacles_msg.box_ids.begin(), obstacles_msg.box_ids.end());
        for (const auto& mapping : id_mapping_) {
            int tracker_id_key = mapping.first;
            int measurement_id = mapping.second;
            int tracker_index = -1, tracker_id = -1;
            if (!resolveMapping(tracker_id_key, tracker_index, tracker_id)) continue;
            if (already_pub.count(tracker_id)) continue;

            const auto& tracker = trackers_[tracker_index];
            if (tracker.history_vel.empty()) continue;
            double v_norm = tracker.history_vel.back().norm();
            if (v_norm < params_.CANDIDATE_VELOCITY_THRESHOLD) continue;
            if (v_norm > params_.DYNAMIC_VELOCITY_MAX) continue;  // ignore wild outliers

            auto state = tracker.getState();

            obstacles_msg.box_ids.push_back(tracker_id);
            obstacles_msg.centroid_x.push_back(state.position.x());
            obstacles_msg.centroid_y.push_back(state.position.y());
            obstacles_msg.centroid_z.push_back(state.position.z());
            obstacles_msg.width.push_back(curr_sizes_[measurement_id][0]);
            obstacles_msg.height.push_back(curr_sizes_[measurement_id][1]);
            obstacles_msg.thickness.push_back(curr_sizes_[measurement_id][2]);
            obstacles_msg.mask.push_back(curr_masks_[measurement_id]);
            obstacles_msg.box_x.push_back(curr_boxes2D_raw_.x[measurement_id]);
            obstacles_msg.box_y.push_back(curr_boxes2D_raw_.y[measurement_id]);
            obstacles_msg.box_width.push_back(curr_boxes2D_raw_.width[measurement_id]);
            obstacles_msg.box_height.push_back(curr_boxes2D_raw_.height[measurement_id]);
            obstacles_msg.observed.push_back(true);
            obstacles_msg.confidence.push_back(0.5f);  // candidate

            nav_msgs::Path his_path;
            his_path.header.frame_id = kFrameId;
            his_path.header.stamp = current_time;
            obstacles_msg.obstacles_history_pos.push_back(his_path);
        }

        // Publish obstacles message (downstream consumes via OUTPUT frame)
        obstacles_msg.header.stamp = current_time;
        obstacles_msg.header.frame_id = kOutputFrame;
        obstacles_pub_.publish(obstacles_msg);
    };

    // Orchestrate
    publishCubesAndPaths();
    publishTextMarkers();
    publishObstaclesMsg();

    // Remember which IDs we actively published this frame, so next frame can
    // emit DELETE markers for any that drop out.
    last_published_ids_ = current_published_ids;
}