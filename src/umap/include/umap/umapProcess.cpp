#include "umapType.h" // Umap header file
#include "umapProcess.h"

#define DETAIL_HISTOGRAM 1 // for more information

/// @brief Initialize ROS subscribers & publishers (Constructor func)
umapProcess::umapProcess() {

    depth_sub_.subscribe(nh_, "/camera/aligned_depth_to_color/image_raw", 1);
    rgb_sub_.subscribe(nh_, "/camera/color/image_raw", 1);

    // Synchronize time stamp of depth & RGB image
    sync_ = std::make_shared<Sync>(SyncPolicy(10), depth_sub_, rgb_sub_);
    sync_->registerCallback(boost::bind(&umapProcess::syncCallback, this, _1, _2));

    // get camera pose
    camera_pose_sub = nh_.subscribe("/camera/current_pose", 10, &umapProcess::cameraPoseCallback, this);

    bbox_2D_pub_ = nh_.advertise<umap::BoundingBoxes2D>("/umap/2D_bounding_boxes", 10);
    u_map_pub_ = nh_.advertise<sensor_msgs::Image>("/umap/u_map", 10);
    rgb_bbox_pub_ = nh_.advertise<sensor_msgs::Image>("/umap/visualization/bbox_rgb_visualization", 10);
    bbox_3D_pub_ = nh_.advertise<umap::BoundingBoxes3D>("/umap/3D_bounding_boxes", 10);
    marker_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("/umap/visualization/3D_bounding_boxes", 10);

    // Get params from params server
    nh_.param("umap_node/threshold_point", params_.threshold_point, 3.0f);
    nh_.param("umap_node/hist_size", params_.hist_size, 150);
    nh_.param("umap_node/depth_factor", params_.depth_factor, 1.0f);
    nh_.param("umap_node/min_dist", params_.min_dist, 400);
    nh_.param("umap_node/max_dist", params_.max_dist, 5000);
    nh_.param("umap_node/threshold_line", params_.threshold_line, 5.0f);
    nh_.param("umap_node/min_length_line", params_.min_length_line, 10);
    nh_.param("umap_node/inverse_depth", params_.inverse_depth, false);

    nh_.param("umap_node/camera_frame", params_.camera_frame, std::string("camera_color_optical_frame"));
    nh_.param("umap_node/camera/fx", params_.fx, 525.0f);
    nh_.param("umap_node/camera/fy", params_.fy, 525.0f);
    nh_.param("umap_node/camera/px", params_.px, 319.5f);
    nh_.param("umap_node/camera/py", params_.py, 239.5f);

    ROS_INFO("Umap Process Node Initialized.");
}


void umapProcess::cameraPoseCallback(const geometry_msgs::PoseStampedConstPtr& msg) {
    cam_to_world_ = Eigen::Matrix4d::Identity();
    Eigen::Matrix3d rotx;
    rotx << 1, 0, 0,
            0, 0, 1,
            0, -1, 0;
    cam_to_world_.block<3,3>(0,0) = Eigen::Quaterniond(msg->pose.orientation.w, msg->pose.orientation.x, msg->pose.orientation.y, msg->pose.orientation.z).toRotationMatrix()*rotx;
    
    
    
    cam_to_world_.block<3,1>(0,3) = Eigen::Vector3d(msg->pose.position.x, msg->pose.position.y, msg->pose.position.z);

    has_pose = true;
}
/// @brief Callback function for synchronized depth and RGB images
/// @param depth_msg 
/// @param rgb_msg 
void umapProcess::syncCallback(const sensor_msgs::ImageConstPtr& depth_msg, const sensor_msgs::ImageConstPtr& rgb_msg) {
    if (!has_pose) {
        ROS_WARN("No camera pose received yet, skipping frame");
        return;
    }

    try {
        cv_bridge::CvImagePtr depth_ptr = cv_bridge::toCvCopy(depth_msg, "32FC1");
        cv_bridge::CvImagePtr rgb_ptr = cv_bridge::toCvCopy(rgb_msg, "bgr8");
        
        // cv::Mat depth_val = depth_ptr->image / params_.depth_factor; // converted to real depth
        cv::Mat depth_val = depth_ptr->image;
        cv::Mat rgb_img = rgb_ptr->image;

        // Preprocess depth image: statistics and inversion
        cv::Mat valid_mask = depth_val > 0;
        double min_depth, max_depth;
        cv::minMaxLoc(depth_val, &min_depth, &max_depth, nullptr, nullptr, valid_mask);
        
        ROS_INFO("Depth image stats - Min: %.2f, Max: %.2f, Mean: %.2f", 
             min_depth, max_depth, cv::mean(depth_val, valid_mask)[0]);
        
        // Apply depth inversion if enabled
        if (params_.inverse_depth) {
            cv::Mat inverted_depth = cv::Mat::zeros(depth_val.size(), depth_val.type());
            cv::Mat flipped_values = (max_depth + min_depth) - depth_val;
            flipped_values.copyTo(inverted_depth, valid_mask);
            depth_val = inverted_depth;
            ROS_INFO("Depth image inverted");
        }

        if (depth_val.empty() || rgb_img.empty()) {
            ROS_ERROR("Empty depth or RGB image received");
            return;
        }

        current_depth_ = depth_val.clone();
        current_rgb_ = rgb_img.clone();

        // Main Image Process
        ROS_INFO_STREAM("Processing frame" << current_frame_++);
        cv::Mat hist_matrix = generateHistogramMatrix(depth_val);
        // ROS_INFO_STREAM("Histogram matrix size: " << hist_matrix.cols << hist_matrix.rows);

        applyGaussianBlur(hist_matrix);
        extract_bb(hist_matrix);
        // ROS_INFO("bb extracted: %zu", this->bounding_box_U.size());

        const ros::Time current_time_stamp = depth_msg->header.stamp; // as 2D, 3D boxes' timestamp in messageheader
        // ROS_INFO("Current time stamp: %f", current_time_stamp.toSec());
        extract_2DboxRGB(current_time_stamp, depth_val);
        // ROS_INFO("2D bounding boxes extracted: %zu", bounding_box_HW.size());
        
        extract_3Dbox(current_time_stamp, depth_val);
        visualizeUmap(hist_matrix, depth_val);

    } catch (cv_bridge::Exception& e) {
        ROS_ERROR("CV Bridge error: %s", e.what());
    } catch (std::exception& e) {
        ROS_ERROR("Error in sync callback: %s", e.what());
    }
}


/// @brief: Generate histogram matrix from depth image
/// @param: cv::Mat& depth_val
/// @return: cv::Mat& hist_matrix
cv::Mat umapProcess::generateHistogramMatrix(const cv::Mat& depth_val){
    cv::Mat hist_matrix = cv::Mat::zeros(params_.hist_size, depth_val.cols,  CV_32F);
    for (int col_idx = 0; col_idx < depth_val.cols; col_idx++) {
        std::vector<float> valid_col;
        for (int row = 0; row < depth_val.rows; row++) {
            float depth = depth_val.at<float>(row, col_idx);
            if (std::isfinite(depth) && depth > params_.min_dist && depth < params_.max_dist) {
                valid_col.push_back(depth);
            }
        }

        if (valid_col.empty()) continue;

        std::vector<int> hist(params_.hist_size, 0);
        for (float depth : valid_col) {
            int bin_idx = static_cast<int>((depth - params_.min_dist) / (params_.max_dist - params_.min_dist) * params_.hist_size);
            if (bin_idx >= 0 && bin_idx < params_.hist_size) {
                hist[bin_idx]++;
            }
        }

        for (int bin_idx = 0; bin_idx < params_.hist_size; bin_idx++) {
            hist_matrix.at<float>(bin_idx, col_idx) = static_cast<float>(hist[bin_idx]) / valid_col.size();
        }
    }
    return hist_matrix;
}


/// @brief: Apply Gaussian blur to histogram matrix
void umapProcess::applyGaussianBlur(cv::Mat& hist_matrix){
    cv::GaussianBlur(hist_matrix, hist_matrix, cv::Size(5, 9), 1.5, 0);
}


/// @brief A part of generate 2Dbox from histogram
UVbox umapProcess::merge_two_UVbox(UVbox father, UVbox son){
    // merge the bounding box
    int top =       (father.bb.tl().y < son.bb.tl().y)?father.bb.tl().y:son.bb.tl().y;
    int left =      (father.bb.tl().x < son.bb.tl().x)?father.bb.tl().x:son.bb.tl().x;
    int bottom =    (father.bb.br().y > son.bb.br().y)?father.bb.br().y:son.bb.br().y;
    int right =     (father.bb.br().x > son.bb.br().x)?father.bb.br().x:son.bb.br().x;
    father.bb = cv::Rect(cv::Point2f(left, top), cv::Point2f(right, bottom));
    return father;
}

/// @brief Extract bounding boxes from HISTOGRAM matrix, store in class variable bounding_box_U
/// @param hist_matrix
void umapProcess::extract_bb(cv::Mat& hist_matrix){
        // initialize a mask
        std::vector<std::vector<int>> mask(hist_matrix.rows, std::vector<int>(hist_matrix.cols, 0));

        // initialize parameters for extraction
        float u_min = params_.threshold_point;  // Adjust this threshold to work with unnormalized values
        float sum_line = 0;
        float max_line = 0;
        int length_line = 0;
        int seg_id = 0;

        std::vector<UVbox> UVboxes;

        for (int row = 0; row < hist_matrix.rows; row++){

            for (int col = 0; col < hist_matrix.cols; col++){
                if (hist_matrix.at<float>(row, col) >= u_min){
                    // ROS_INFO_STREAM("Valid point found!");
                    length_line++; // line is horizontal
                    sum_line += hist_matrix.at<float>(row,col);
                    max_line = std::max(max_line, hist_matrix.at<float>(row,col));
                }
                if (hist_matrix.at<float>(row, col) < u_min || col == hist_matrix.cols - 1){
                    // Meet the first non-interest point OR meet the end of row
                    // Thus should end sum 'length_line'
                    
                    // is end of the row
                    col = (col == hist_matrix.cols - 1)? col + 1:col;

                    // is good line candidate (length and sum)
                    // ROS_INFO_STREAM("length_line: " << length_line << "\n");
                    // ROS_INFO_STREAM("max_line: " << max_line << "\n");
                    if(length_line > params_.min_length_line && sum_line > params_.threshold_line * max_line)
                    {
                        seg_id++;
                        UVboxes.push_back(UVbox(seg_id, row, col-length_line, col-1));
                        // ROS_INFO_STREAM("UVboxes pushback: " << seg_id << "\n");
                        // overwrite the mask with segementation id
                        for(int c = col-length_line; c < col; c++)
                        {
                            mask[row][c] = seg_id;
                        }

                        // when current row is not first row, we need to merge neighbour segementation
                        if(row != 0){
                            // merge all parents
                            for(int c = col - length_line; c < col - 1; c++){
                                if(mask[row - 1][c] != 0) // 像素头顶上的位置有意义
                                {
                                    // initially, toppest parent id is just seg id.
                                    // mask stores seg_ids
                                    if(UVboxes[mask[row - 1][c] - 1].toppest_parent_id < UVboxes.back().toppest_parent_id) // seg_id从1开始计
                                    {
                                        UVboxes.back().toppest_parent_id = UVboxes[mask[row - 1][c] - 1].toppest_parent_id;
                                    }
                                    else
                                    {
                                        int temp = UVboxes[mask[row - 1][c] - 1].toppest_parent_id;
                                        for(size_t b = 0; b < UVboxes.size(); b++)
                                        {
                                            UVboxes[b].toppest_parent_id = (UVboxes[b].toppest_parent_id == temp)?UVboxes.back().toppest_parent_id:UVboxes[b].toppest_parent_id;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    sum_line = 0;
                    max_line = 0;
                    length_line = 0;
                }
            }
        }
        // group lines into boxes
        this->bounding_box_U.clear();
        // merge boxes with same toppest parent
        // ROS_INFO_STREAM("UVBOXES size: " << UVboxes.size() << "\n");
        for(size_t b = 0; b < UVboxes.size(); b++)
        {
            // ROS_INFO_STREAM(UVboxes[b].id);
            
            if(UVboxes[b].id == UVboxes[b].toppest_parent_id)
            {
                // ROS_INFO_STREAM("\nUVboxes[b].id == UVboxes[b].toppest_parent_id");
                for(size_t s = b + 1; s < UVboxes.size(); s++)
                {
                    if(UVboxes[s].toppest_parent_id == UVboxes[b].id)
                    {
                        UVboxes[b] = merge_two_UVbox(UVboxes[b], UVboxes[s]);
                    }
                }
                // check box's size
                if(UVboxes[b].bb.area() >= 0)
                {
                    this->bounding_box_U.push_back(UVboxes[b].bb);
                    // printf("bbox_U [b] %d",UVboxes[b].bb.tl().y);
                }
            }
        }
    }


/// @brief  Extract 2D bounding boxes from depth/RGB image and store in bounding_box_D 
/// @param depth_val
/// @param stamp
void umapProcess::extract_2DboxRGB(const ros::Time& stamp, const cv::Mat& depth_val) {
    bounding_box_HW.clear(); // bounding_box_HW 仅保存当前帧的Bbox

    std::lock_guard<std::mutex> rgb_lock(rgb_mutex_);
    std::lock_guard<std::mutex> depth_lock(depth_mutex_);
    
    if (current_rgb_.empty()) {
        ROS_WARN("No RGB image available for visualization");
        return;
    }

    cv::Mat visualization = current_rgb_.clone(); // For generating rgb_bbox_visualization
    // 对每个histogram中的UVbox进行处理
    for(const auto& bb_U : bounding_box_U) {
        // Find corresponding 3D vertical range
        // 找到histogram中UVbox对应的3D垂直范围
        float bin_width = (params_.max_dist - params_.min_dist) / (params_.hist_size*1.0);
        float depth_in_near = params_.min_dist + bb_U.y * bin_width;
        float depth_in_far = depth_in_near + bb_U.height * bin_width * 1.3;

        // Vertical range detection in RGB image
        int y_up = current_rgb_.rows, y_down = 0;
        int x_start = std::max(0, bb_U.x);
        int x_end = std::min(current_rgb_.cols, bb_U.x + bb_U.width);

        if (x_start < depth_val.cols && x_end > 0) {
            for(int i = x_start; i < x_end; i++) {
                for(int j = 0; j < current_rgb_.rows && j < depth_val.rows; j++) {
                    if (i < depth_val.cols) {
                        float depth = depth_val.at<float>(j, i);
                        if (std::isfinite(depth) && depth >= depth_in_near && depth <= depth_in_far) {
                            y_up = std::min(y_up, j);
                            y_down = std::max(y_down, j);
                        }
                    }
                }
            }
        }

        if (y_up < y_down) {
            // Draw rectangle on RGB image
            cv::Point pt1(x_start, y_up); // top-left
            cv::Point pt2(x_end, y_down); // bottom-right
            
            cv::Scalar color = cv::Scalar(0, 255, 0);  // Green for static

            cv::rectangle(visualization, pt1, pt2, color, 2); // visualization size: (480, 640)
            // 将bboxes pushback to bounding_box_HW或再创建一个成员变量存储框框 2D bounding boxes
            bounding_box_HW.push_back(cv::Rect(pt1, pt2));
        }
    } // end for loop

    // Publish visualization after iterating through all bounding boxes
    sensor_msgs::ImagePtr rgb_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", visualization).toImageMsg();
    rgb_msg->header.stamp = stamp;
    rgb_bbox_pub_.publish(rgb_msg);
    ROS_INFO("RGB image with bounding boxes published");

    // publish2DBoxes(bounding_box_HW, stamp, current_rgb_.cols, current_rgb_.rows);
    umap::BoundingBoxes2D HW_box_msg;
    HW_box_msg.header.stamp = stamp;
    HW_box_msg.header.frame_id = params_.camera_frame;
    HW_box_msg.image_width = current_rgb_.cols;
    HW_box_msg.image_height = current_rgb_.rows;
    for (size_t i = 0; i < bounding_box_HW.size(); ++i) {
        HW_box_msg.box_ids.push_back(i);  // 每个框的唯一ID
        HW_box_msg.x.push_back(bounding_box_HW[i].x);
        HW_box_msg.y.push_back(bounding_box_HW[i].y);
        HW_box_msg.height.push_back(bounding_box_HW[i].height);
        HW_box_msg.width.push_back(bounding_box_HW[i].width);
        // msg.confidence.push_back(1.0);  // default 1.0
    }

    bbox_2D_pub_.publish(HW_box_msg);
    ROS_INFO("2D bounding boxes 框框msg published: %zu", bounding_box_HW.size());
    
}

void umapProcess::extract_3Dbox(const ros::Time& stamp, const cv::Mat& depth_val){
    boxes3D.clear();
    std::vector<cv::Mat> box_masks;

    // current_frame_++;

    // For boost
    // cv::Mat depth_resize;
    // cv::resize(depth_val, depth_resize, cv::Size(), 1.0/row_downsample, 1.0);

    for (const auto& bb_U : bounding_box_U) {
        // Find corresponding 3D vertical range
        float bin_width = (params_.max_dist - params_.min_dist) / (params_.hist_size*1.0);
        float depth_in_near = params_.min_dist + bb_U.y * bin_width;
        float depth_in_far = depth_in_near + bb_U.height * bin_width * 1.3;

        // Vertical range detection in RGB image
        int y_up = current_rgb_.rows, y_down = 0;
        int x_start = std::max(0, bb_U.x);
        int x_end = std::min(current_rgb_.cols, bb_U.x + bb_U.width);

        if (x_start < depth_val.cols && x_end > 0) { // if valid
            for(int i = x_start; i < x_end; i++) {
                for(int j = 0; j < current_rgb_.rows && j < depth_val.rows; j++) {
                    if (i < depth_val.cols) {
                        float depth = depth_val.at<float>(j, i);
                        if (std::isfinite(depth) && depth >= depth_in_near && depth <= depth_in_far) {
                            y_up = std::min(y_up, j);
                            y_down = std::max(y_down, j);
                        }
                    }
                }
            }
        }

        if (y_up < y_down) {
            // 1. 生成3D box
            box3D curr_box;
            float mid_depth = (depth_in_near + depth_in_far) / 2.0;

            mid_depth /= params_.depth_factor; // convert mm to meter
            depth_in_far /= params_.depth_factor;
            depth_in_near /= params_.depth_factor;

            // 3D box centeroid in camera coordinates
            // (x, y, z)为camera坐标系(右手)下的位置，单位：m
            // x: 水平← 宽，y: 竖直↑ 高，z: 深
            float cam_x = (bb_U.x + bb_U.width / 2.0 - params_.px) * mid_depth / params_.fx;
            float cam_y = ((y_up + y_down) / 2.0 - params_.py) * mid_depth / params_.fy;
            float cam_z = mid_depth;

            // Transform camera coordinates to world coordinates
            Eigen::Vector4d cam_pos(cam_x, cam_y, cam_z, 1.0);
            Eigen::Vector4d world_pos = cam_to_world_ * cam_pos;

            curr_box.x = world_pos.x();
            curr_box.y = world_pos.y();
            curr_box.z = world_pos.z();

            // Calculate box dimensions in meters
            curr_box.x_width = std::max(0.1f, bb_U.width * mid_depth / params_.fx); //宽度，保证尺寸不小于0.1*0.1*0.1
            curr_box.y_width = std::max(0.1f, (y_down - y_up) * mid_depth / params_.fy); //高度
            curr_box.z_width = std::max(0.1f, depth_in_far - depth_in_near); //厚度
            boxes3D.push_back(curr_box);

            // 2.Create mask data
            cv::Mat mask = extractRGBPointsInBox(depth_val, current_rgb_, curr_box);
            box_masks.push_back(mask);
        }
    } // end for loop

    // Publish 3D bounding boxes message
    umap::BoundingBoxes3D bbox3D_msg;

    bbox3D_msg.header.stamp = stamp; // Use the same timestamp as the input images
    bbox3D_msg.header.frame_id = "world"; //params_.camera_frame
    bbox3D_msg.image_width = current_rgb_.cols;
    bbox3D_msg.image_height = current_rgb_.rows;


    for(size_t i = 0; i < boxes3D.size(); i++) {
        bbox3D_msg.box_ids.push_back(i);

        // Convert center coordinates and dimensions
        bbox3D_msg.x.push_back(boxes3D[i].x);
        bbox3D_msg.y.push_back(boxes3D[i].y);
        bbox3D_msg.z.push_back(boxes3D[i].z);

        bbox3D_msg.width.push_back(boxes3D[i].x_width);
        bbox3D_msg.height.push_back(boxes3D[i].y_width);
        bbox3D_msg.thickness.push_back(boxes3D[i].z_width);
        sensor_msgs::ImagePtr mask_msg = cv_bridge::CvImage(std_msgs::Header(), "mono8", box_masks[i]).toImageMsg();
        bbox3D_msg.mask.push_back(*mask_msg);

    }

    // 发布3D边界框消息
    bbox_3D_pub_.publish(bbox3D_msg);
    ROS_INFO("Published 3D bounding boxes message with %zu boxes", boxes3D.size());

    // Publish 3D bounding boxes for Visualization
    visualization_msgs::MarkerArray marker_array;

    for (size_t i = 0; i<boxes3D.size(); i++) {
        visualization_msgs::Marker marker;
        marker.header.frame_id = "world"; // Changed from camera frame to world frame
        marker.header.stamp = ros::Time::now();
        marker.ns = "3D_bounding_boxes";
        marker.id = i;
        marker.type = visualization_msgs::Marker::CUBE;
        marker.action = visualization_msgs::Marker::ADD;

        // Set the position (already in world coordinates)
        marker.pose.position.x = boxes3D[i].x;
        marker.pose.position.y = boxes3D[i].y;
        marker.pose.position.z = boxes3D[i].z;

        // Orientation (identity quaternion)
        marker.pose.orientation.w = 1.0;
        marker.pose.orientation.x = 0.0;
        marker.pose.orientation.y = 0.0;
        marker.pose.orientation.z = 0.0;

        marker.scale.x = boxes3D[i].z_width;
        marker.scale.y = boxes3D[i].x_width;
        marker.scale.z = boxes3D[i].y_width;

        // Set color and transparency
        marker.color.r = 0.0f;
        marker.color.g = 1.0f;
        marker.color.b = 0.0f;
        marker.color.a = 0.5f; // semi-transparent

        marker.lifetime = ros::Duration(0.2);  // Exist lifetime

        // Add the marker to the array
        marker_array.markers.push_back(marker);
    }
    marker_pub_.publish(marker_array);
    ROS_INFO("3D bounding boxes published: %zu", boxes3D.size());
}


/// @brief  已弃用！Generate a visualisation of the U-depth-MAP results
/// @param stamp 
/// @param depth_val 
void umapProcess::visualize_rgb_bboxes(const ros::Time& stamp, const cv::Mat& depth_val) {
    std::lock_guard<std::mutex> rgb_lock(rgb_mutex_);
    std::lock_guard<std::mutex> depth_lock(depth_mutex_);
    
    if (current_rgb_.empty()) {
        ROS_WARN("No RGB image available for visualization");
        return;
    }

    cv::Mat visualization = current_rgb_.clone();

    for(const auto& bb_U : bounding_box_U) {
        // Find corresponding 3D vertical range
        float bin_width = (params_.max_dist - params_.min_dist) / (params_.hist_size*1.0);
        float depth_in_near = params_.min_dist + bb_U.y * bin_width;
        float depth_in_far = depth_in_near + bb_U.height * bin_width * 1.3;

        // Vertical range detection in RGB image
        int y_up = current_rgb_.rows, y_down = 0;
        int x_start = std::max(0, bb_U.x);
        int x_end = std::min(current_rgb_.cols, bb_U.x + bb_U.width);

        if (x_start < depth_val.cols && x_end > 0) {
            for(int i = x_start; i < x_end; i++) {
                for(int j = 0; j < current_rgb_.rows && j < depth_val.rows; j++) {
                    if (i < depth_val.cols) {
                        float depth = depth_val.at<float>(j, i);
                        if (std::isfinite(depth) && depth >= depth_in_near && depth <= depth_in_far) {
                            y_up = std::min(y_up, j);
                            y_down = std::max(y_down, j);
                        }
                    }
                }
            }
        }

        if (y_up < y_down) {
            // Draw rectangle on RGB image
            cv::Point pt1(x_start, y_up);
            cv::Point pt2(x_end, y_down);
            
            // Get track info for this bbox
            int track_id = -1;
            float velocity = 0.0f;
            for (const auto& track : centroid_history_) {
                if (!track.second.empty() && 
                    track.second.back().is_dynamic) {
                    Eigen::Vector3f centroid = track.second.back().position;
                    // Check if this centroid corresponds to current bbox
                    // (You'll need to add logic to match centroids to bboxes)
                    track_id = track.first;
                    velocity = track.second.back().velocity;
                    break;
                }
            }

            // Color dynamic objects red, static ones green
            cv::Scalar color = (track_id >= 0) ? 
                                cv::Scalar(0, 0, 255) : // Red for dynamic
                                cv::Scalar(0, 255, 0);  // Green for static
            
            cv::rectangle(visualization, pt1, pt2, color, 2);

            // Add velocity label for dynamic objects
            std::stringstream ss;
            if (track_id >= 0) {
                ss << "ID:" << track_id << " v:" << std::fixed << std::setprecision(2) << velocity << "m/s";
            } else {
                ss << "static";
            }
            cv::putText(visualization, ss.str(), 
                        cv::Point(x_start, y_up - 5),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, 
                        color, 1);
        }
    }

    // Publish visualization
    sensor_msgs::ImagePtr rgb_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", visualization).toImageMsg();
    rgb_msg->header.stamp = stamp;
    rgb_bbox_pub_.publish(rgb_msg);
}



/// @brief Generate a visualisation of the U-depth-MAP results
/// @param hist_matrix 
/// @param depth_val 
/// @return cv::Mat
cv::Mat umapProcess::visualizeUmap(const cv::Mat& hist_matrix, const cv::Mat& depth_val){
    cv::Mat hist_matrix_t = hist_matrix.clone();
    
    cv::Mat hist_norm;
    double hist_max;
    cv::minMaxLoc(hist_matrix_t, nullptr, &hist_max);

    if (hist_max > 0) {
        hist_norm = hist_matrix_t.clone();
        hist_norm = (hist_norm / hist_max) * 255;
        hist_norm.convertTo(hist_norm, CV_8U);
    } else {
        hist_norm = cv::Mat::zeros(hist_matrix_t.size(), CV_8U);
    }
    

    cv::Mat colormap; // colormap
    cv::applyColorMap(hist_norm, colormap, cv::COLORMAP_JET);

#if DETAIL_HISTOGRAM
    int height = hist_matrix.rows, width = hist_matrix.cols;
    int margin = 50;
    cv::Mat vis_img(height + 2*margin, width + 2*margin, CV_8UC3, cv::Scalar(255, 255, 255));

    colormap.copyTo(vis_img(cv::Rect(margin, margin, width, height)));

    // Add title and labels
    cv::putText(vis_img, "Normalized Depth Histogram by Column", 
                cv::Point(margin, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, 
                cv::Scalar(0, 0, 0), 2);
    // X-axis label
    cv::putText(vis_img, "Column Index", 
                cv::Point(width/2, height + 2*margin - 15), 
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 1);
    // Y-axis label
    cv::putText(vis_img, "Depth", 
                cv::Point(5, height/2 + margin), 
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 1);

    // Add depth range markers
    std::stringstream min_ss, max_ss;
    min_ss << params_.min_dist << " mm";
    max_ss << params_.max_dist << " mm";
    cv::putText(vis_img, max_ss.str(), 
            cv::Point(5, height + margin - 5), 
            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    
    cv::putText(vis_img, min_ss.str(), 
                cv::Point(5, margin + 15), 
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    
    // Add colorbar
    int colorbar_width = 20;
    cv::Mat colorbar(height, colorbar_width, CV_8UC3, cv::Scalar(0, 0, 0));
    
    for (int i = 0; i < height; i++) {
        cv::Mat value(1, 1, CV_8UC1, cv::Scalar(255 * (height - i - 1) / height));
        cv::Mat colored_value;
        cv::applyColorMap(value, colored_value, cv::COLORMAP_JET);
        
        // Copy the color to the colorbar
        for (int j = 0; j < colorbar_width; j++) {
            colorbar.at<cv::Vec3b>(i, j) = colored_value.at<cv::Vec3b>(0, 0);
        }
    }

    colorbar.copyTo(vis_img(cv::Rect(margin + width + 10, margin, colorbar_width, height)));

    // Add colorbar labels
    cv::putText(vis_img, "0", 
                cv::Point(margin + width + 10 + colorbar_width + 5, height + margin - 5), 
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
                
    cv::putText(vis_img, "Max", 
                cv::Point(margin + width + 10 + colorbar_width + 5, margin + 15), 
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    
    // 绘制2D边界框到colormap上
    for (int i = 0; i < bounding_box_U.size(); i++)
    {
        const auto& bb = bounding_box_U[i];
        cv::Rect visible_bb = bb;
        if(visible_bb.height < 3) {
            visible_bb.height = 3;  // 确保边界框可见
        }
        cv::rectangle(colormap, visible_bb, cv::Scalar(0, 255, 0), 2);
    }

    colormap.copyTo(vis_img(cv::Rect(margin, margin, width, height)));


    // Convert to ROS message and publish
    sensor_msgs::ImagePtr ros_image = cv_bridge::CvImage(std_msgs::Header(), "bgr8", vis_img).toImageMsg();
    u_map_pub_.publish(ros_image);
    
    // // After extract_3Dbox and before visualize_3Dboxes:
    // visualize_rgb_bboxes(depth_msg->header.stamp, depth_val);
    return vis_img;
#else

#endif

}


/**
 * @brief Extract RGB points within depth range of 3D box
 * @param depth_img Input depth image
 * @param rgb_img Input RGB image
 * @param box 3D bounding box
 * @return cv::Mat RGB points within depth range
*/
cv::Mat umapProcess::extractRGBPointsInBox(const cv::Mat& depth_img, const cv::Mat& rgb_img, const box3D& box) {
    cv::Mat mask = cv::Mat::zeros(depth_img.size(), CV_8UC1);
    
    // 获取3D框的深度范围
    float min_depth = box.z - box.z_width/2;
    float max_depth = box.z + box.z_width/2;
    ROS_INFO("min depth = %f, max depth = %f", min_depth, max_depth);
    ROS_INFO("box centroid points: %f, %f, %f, \n xyz width: %f, %f, %f \n", box.x, box.y, box.z, box.x_width, box.y_width, box.z_width);

    // 创建深度掩码
    for(int v = 0; v < depth_img.rows; v++) {
        for(int u = 0; u < depth_img.cols; u++) {
            float depth = depth_img.at<float>(v, u) / params_.depth_factor;
            // ROS_INFO("depth val = %f", depth);

            // 检查深度是否有效且在范围内
            if(depth > 0 && depth >= min_depth && depth <= max_depth) {
                // 计算3D点坐标
                float x = (u - params_.px) * depth / params_.fx;
                float y = (v - params_.py) * depth / params_.fy;
                float z = depth;
                // ROS_INFO("coordinate 3D point: %f, %f, %f", x, y, z);
                
                // 检查点是否在3D框内
                if(std::abs(x - box.x) <= box.x_width/2 && std::abs(y - box.y) <= box.y_width/2) {
                    mask.at<uint8_t>(v, u) = 255;
                }
            }
        }
    }
    
    return mask;
}