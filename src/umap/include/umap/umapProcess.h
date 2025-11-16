#ifndef UMAP_PROCESS_H
#define UMAP_PROCESS_H
/*
@file: umapProcess.h
@brief: Header file for umapProcess.cpp
@description: Depth image processing: 
    1. Generate histogram matrix from depth image
    2. Apply Gaussian blur to histogram matrix
    3. Extract 2D bounding boxes(bb) from histogram matrix
    4. Extract 3D boxes from depth image
*/
#include "umapType.h"
#include "params.h"
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/sync_policies/exact_time.h>

class umapProcess {
private:
    ros::NodeHandle nh_;
    message_filters::Subscriber<sensor_msgs::Image> depth_sub_;
    message_filters::Subscriber<sensor_msgs::Image> rgb_sub_;
    ros::Subscriber camera_pose_sub;

    ros::Publisher bbox_2D_pub_; // LS + BR + width + height for downflow job (框框数据，自定义信息)
    ros::Publisher rgb_bbox_pub_; // RGB + bboxes (only for visualization)

    ros::Publisher u_map_pub_;

    ros::Publisher bbox_3D_pub_;
    ros::Publisher marker_pub_;


    // Define synchronization policy (ApproximateTime or ExactTime)
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> SyncPolicy;
    // typedef message_filters::sync_policies::ExactTime<sensor_msgs::Image, sensor_msgs::Image> SyncPolicy;

    typedef message_filters::Synchronizer<SyncPolicy> Sync;
    std::shared_ptr<Sync> sync_;

    // Load parameters
    umapParams params_;

    Eigen::Matrix4d cam_to_world_;
    bool has_pose = false;
    // Image DATA
    cv::Mat current_rgb_;
    cv::Mat current_depth_;
    std::mutex rgb_mutex_;
    std::mutex depth_mutex_;
    
    std::map<int, std::vector<TrackedCentroid>> centroid_history_; // bbox_id -> history

public:
    umapProcess();


    std::vector<UVbox> UVboxes;
    std::vector<cv::Rect> bounding_box_U; // store 2D bboxes extracted from histogram (width, depth)
    std::vector<cv::Rect> bounding_box_HW; // store 2D bboxes extracted from depth image (height, width)
    std::vector<box3D> boxes3D; // store 3D boxes extracted from depth image (x, y, z, x_width, y_width, z_width) 

    int current_frame_ = 0;

    void syncCallback(const sensor_msgs::ImageConstPtr& depth_msg, const sensor_msgs::ImageConstPtr& rgb_msg); // Synchronized callback
    void cameraPoseCallback(const geometry_msgs::PoseStampedConstPtr& pose_msg);

    cv::Mat generateHistogramMatrix(const cv::Mat& depth_val);

    void applyGaussianBlur(cv::Mat& hist_matrix);
    UVbox merge_two_UVbox(UVbox father, UVbox son); // A part of generate 2Dbox from histogram(extract_bb)

    void extract_bb(cv::Mat& hist_matrix);
    
    void extract_2DboxRGB(const ros::Time& stamp, const cv::Mat& depth_val);
    void visualize_rgb_bboxes(const ros::Time& stamp, const cv::Mat& depth_val);

    void extract_3Dbox(const ros::Time& stamp, const cv::Mat& depth_val);
    cv::Mat visualizeUmap(const cv::Mat& hist_matrix, const cv::Mat& depth_val);

private:
    cv::Mat extractRGBPointsInBox(const cv::Mat& depth_img, const cv::Mat& rgb_img, const box3D& box);
};

#endif // UMAP_PROCESS_H