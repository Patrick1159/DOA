#include "umapVisualisation.h"

#define DETAIL_HISTOGRAM 1 // for more information

/// @brief Constructor for umapVisualisation
umapVisualisation::umapVisualisation()
{
    // u_map_pub_ = nh_.advertise<sensor_msgs::Image>("umap/visualization", 1);
    
    // // 如果需要从ROS参数服务器加载参数
    // // nh_.getParam("some_param", params_.some_param);
    
    ROS_INFO("umapVisualisation initialized");
}

/// @brief Destructor for umapVisualisation
umapVisualisation::~umapVisualisation()
{
    // Destructor implementation (if needed)
    ROS_INFO("umapVisualisation destroyed");
}

// /// @brief Generate a visualisation of the U-depth-MAP results
// /// @param hist_matrix 
// /// @param depth_val 
// /// @return cv::Mat
// cv::Mat umapVisualisation::visualizeUmap(const cv::Mat& hist_matrix, const cv::Mat& depth_val){
//     cv::Mat hist_matrix_t = hist_matrix.clone();
    
//     cv::Mat hist_norm;
//     double hist_max;
//     cv::minMaxLoc(hist_matrix_t, nullptr, &hist_max);

//     if (hist_max > 0) {
//         hist_norm = hist_matrix_t.clone();
//         hist_norm = (hist_norm / hist_max) * 255;
//         hist_norm.convertTo(hist_norm, CV_8U);
//     } else {
//         hist_norm = cv::Mat::zeros(hist_matrix_t.size(), CV_8U);
//     }
    

//     cv::Mat colormap; // colormap
//     cv::applyColorMap(hist_norm, colormap, cv::COLORMAP_JET);

// #if DETAIL_HISTOGRAM
//     int height = hist_matrix.rows, width = hist_matrix.cols;
//     int margin = 50;
//     cv::Mat vis_img(height + 2*margin, width + 2*margin, CV_8UC3, cv::Scalar(255, 255, 255));

//     colormap.copyTo(vis_img(cv::Rect(margin, margin, width, height)));

//     // Add title and labels
//     cv::putText(vis_img, "Normalized Depth Histogram by Column", 
//                 cv::Point(margin, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, 
//                 cv::Scalar(0, 0, 0), 2);
//     // X-axis label
//     cv::putText(vis_img, "Column Index", 
//                 cv::Point(width/2, height + 2*margin - 15), 
//                 cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 1);
//     // Y-axis label
//     cv::putText(vis_img, "Depth", 
//                 cv::Point(5, height/2 + margin), 
//                 cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 1);

//     // Add depth range markers
//     std::stringstream min_ss, max_ss;
//     min_ss << params_.min_dist << " mm";
//     max_ss << params_.max_dist << " mm";
//     cv::putText(vis_img, max_ss.str(), 
//             cv::Point(5, height + margin - 5), 
//             cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    
//     cv::putText(vis_img, min_ss.str(), 
//                 cv::Point(5, margin + 15), 
//                 cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    
//     // Add colorbar
//     int colorbar_width = 20;
//     cv::Mat colorbar(height, colorbar_width, CV_8UC3, cv::Scalar(0, 0, 0));
    
//     for (int i = 0; i < height; i++) {
//         cv::Mat value(1, 1, CV_8UC1, cv::Scalar(255 * (height - i - 1) / height));
//         cv::Mat colored_value;
//         cv::applyColorMap(value, colored_value, cv::COLORMAP_JET);
        
//         // Copy the color to the colorbar
//         for (int j = 0; j < colorbar_width; j++) {
//             colorbar.at<cv::Vec3b>(i, j) = colored_value.at<cv::Vec3b>(0, 0);
//         }
//     }

//     colorbar.copyTo(vis_img(cv::Rect(margin + width + 10, margin, colorbar_width, height)));

//     // Add colorbar labels
//     cv::putText(vis_img, "0", 
//                 cv::Point(margin + width + 10 + colorbar_width + 5, height + margin - 5), 
//                 cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
                
//     cv::putText(vis_img, "Max", 
//                 cv::Point(margin + width + 10 + colorbar_width + 5, margin + 15), 
//                 cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    
//     // 绘制2D边界框到colormap上
//     for (int i = 0; i < umapResults.bounding_box_U.size(); i++)
//     {
//         const auto& bb = umapResults.bounding_box_U[i];
//         cv::Rect visible_bb = bb;
//         if(visible_bb.height < 3) {
//             visible_bb.height = 3;  // 确保边界框可见
//         }
//         cv::rectangle(colormap, visible_bb, cv::Scalar(0, 255, 0), 2);
//     }

//     colormap.copyTo(vis_img(cv::Rect(margin, margin, width, height)));

//     return vis_img;
    
//     // // Convert to ROS message and publish
//     // sensor_msgs::ImagePtr ros_image = cv_bridge::CvImage(std_msgs::Header(), "bgr8", vis_img).toImageMsg();
//     // u_map_pub_.publish(ros_image);
    
//     // // After extract_3Dbox and before visualize_3Dboxes:
//     // visualize_rgb_bboxes(depth_msg->header.stamp, depth_val);
// #else

// #endif

// }
