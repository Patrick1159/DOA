#include "umapProcess.h"

/// @brief Initialize ROS subscribers & publishers (Constructor func)
umapProcess::umapProcess() {

    depth_sub_.subscribe(nh_, "/camera/aligned_depth_to_color/image_raw", 1);
    rgb_sub_.subscribe(nh_, "/camera/color/image_raw", 1);

    // Synchronize time stamp of depth & RGB image
    sync_ = std::make_shared<Sync>(SyncPolicy(10), depth_sub_, rgb_sub_);
    sync_->registerCallback(boost::bind(&umapProcess::syncCallback, this, _1, _2));

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
    nh_.param("umap_node/enable_horizontal_merge", params_.enable_horizontal_merge, false);
    nh_.param("umap_node/merge_horizontal_gap", params_.merge_horizontal_gap, 30);
    nh_.param("umap_node/merge_max_row_gap", params_.merge_max_row_gap, 8);
    nh_.param("umap_node/merge_min_vertical_overlap_ratio", params_.merge_min_vertical_overlap_ratio, 0.3f);
    nh_.param("umap_node/enable_adaptive_horizontal_merge", params_.enable_adaptive_horizontal_merge, true);
    nh_.param("umap_node/adaptive_merge_min_gap", params_.adaptive_merge_min_gap, 20);
    nh_.param("umap_node/adaptive_merge_max_gap", params_.adaptive_merge_max_gap, 100);
    nh_.param("umap_node/adaptive_merge_near_ratio", params_.adaptive_merge_near_ratio, 0.25f);
    nh_.param("umap_node/adaptive_merge_far_ratio", params_.adaptive_merge_far_ratio, 0.85f);

    nh_.param("umap_node/camera_frame", params_.camera_frame, std::string("camera_color_optical_frame"));
    nh_.param("umap_node/camera/fx", params_.fx, 525.0f);
    nh_.param("umap_node/camera/fy", params_.fy, 525.0f);
    nh_.param("umap_node/camera/px", params_.px, 319.5f);
    nh_.param("umap_node/camera/py", params_.py, 239.5f);

    ROS_INFO("Umap Process Node Initialized.");
}


/// @brief Callback function for synchronized depth and RGB images
/// @param depth_msg
/// @param rgb_msg
void umapProcess::syncCallback(const sensor_msgs::ImageConstPtr& depth_msg, const sensor_msgs::ImageConstPtr& rgb_msg) {
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

        // ROS_INFO("Depth image stats - Min: %.2f, Max: %.2f, Mean: %.2f",
        //      min_depth, max_depth, cv::mean(depth_val, valid_mask)[0]);

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
        ROS_INFO_STREAM("\033[30;43m[UMAP Module] Processing frame " << current_frame_++ << "\033[0m");
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
    cv::GaussianBlur(hist_matrix, hist_matrix, cv::Size(5, 9), 10, 10);
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
    // ROS_INFO("min depth = %f, max depth = %f", min_depth, max_depth);
    // ROS_INFO("box centroid points: %f, %f, %f, \n xyz width: %f, %f, %f \n", box.x, box.y, box.z, box.x_width, box.y_width, box.z_width);

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
