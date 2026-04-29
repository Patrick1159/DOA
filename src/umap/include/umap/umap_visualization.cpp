#include "umapProcess.h"

#define DETAIL_HISTOGRAM 1 // for more information


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
