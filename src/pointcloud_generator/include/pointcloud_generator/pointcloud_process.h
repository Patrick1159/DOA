
#ifndef POINTCLOUDPROCESSOR_H
#define POINTCLOUDPROCESSOR_H

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/PointCloud2.h>
#include <std_msgs/Int32.h>
#include <string>
#include <iostream>
#include <vector>
#include <time.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/QuaternionStamped.h>
#include <nav_msgs/Odometry.h>
#include <std_msgs/Float64.h>

// Eigen 库
#include <Eigen/Eigen>
#include <Eigen/Core>
#include <Eigen/Geometry>

// TF 库
#include <tf/transform_listener.h>
#include <tf/transform_datatypes.h>
#include <tf_conversions/tf_eigen.h>
#include <tf/transform_broadcaster.h>
 
// PCL 库
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/common/common.h>
#include <pcl/point_cloud.h>
// #include <pcl/visualization/pcl_visualizer.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/random_sample.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/common/transforms.h>
// #include <pcl/concatenate.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/registration/icp.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/crop_box.h>

// 自定义消息
#include <umap/Obstacles.h>

class PointCloudProcessor
{
private:
    ros::NodeHandle nh_;
    ros::Publisher pc_pub;
    ros::Subscriber pose_sub, odom_sub;
    ros::Subscriber obstacles_sub;
    ros::Subscriber obstacles_mask_sub;
    image_transport::Subscriber color_sub, depth_sub;

    std::string pointcloud_frame_id;
    double fx, fy, cx, cy, camera_factor;
    double max_depth, min_depth, voxel_size, eud_range;
    int eud_min_points_num;

    bool has_odom, has_color, has_depth, has_pose;

    Eigen::Matrix4d uav_to_world, cam_to_world;

    bool dynamic_only; // 是否只保留动态障碍物
    int segmentation_mode; // 0为box mode; 1为mask mode; 2为都考虑
    struct edge_points{ // To store the edge points of the bounding box in pixel coordinates
        std::pair<int, int> top_left_pt;
        std::pair<int, int> top_right_pt;
        std::pair<int, int> bottom_left_pt;
        std::pair<int, int> bottom_right_pt;
    };
    std::vector<edge_points> curr_edge_pts;
    struct depth_range {
        double min_depth;
        double max_depth;
    };
    std::vector<depth_range> curr_depth_ranges;
    std::vector<cv::Mat> curr_masks;

    cv::Mat color_pic, depth_pic;

    void odom_callback(const nav_msgs::Odometry::ConstPtr& msg);
    void color_callback(const sensor_msgs::ImageConstPtr& msg);
    void depth_callback(const sensor_msgs::ImageConstPtr& msg);
    void pose_callback(const geometry_msgs::PoseStamped::ConstPtr& msg);
    void obstacles_callback(const umap::Obstacles &obstacles_msg);

public:
    PointCloudProcessor(){}
    ~PointCloudProcessor(){}

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr global_pc;

    int init(const ros::NodeHandle& nh){
        nh_ = nh;

        image_transport::ImageTransport it_(nh);
        
        pc_pub = nh_.advertise<sensor_msgs::PointCloud2>("/RED/pointcloud_process/pointcloud", 1);
        pose_sub = nh_.subscribe("/RED/pointcloud_process/camera_pose", 1, &PointCloudProcessor::pose_callback, this);
        odom_sub = nh_.subscribe("/RED/pointcloud_process/odom", 1, &PointCloudProcessor::odom_callback, this);
        color_sub = it_.subscribe("/RED/pointcloud_process/color_image", 1, &PointCloudProcessor::color_callback, this);
        depth_sub = it_.subscribe("/RED/pointcloud_process/depth_image", 1, &PointCloudProcessor::depth_callback, this);

        obstacles_sub = nh_.subscribe("/model_output/obstacles", 100, &PointCloudProcessor::obstacles_callback, this);
        
        nh_.getParam("camera/view_field",max_depth);
        nh_.getParam("camera/self_filter",min_depth);
        nh_.getParam("camera/grid_size",voxel_size);
        nh_.getParam("camera/eud_range",eud_range);
        nh_.getParam("camera/eud_min_points_num",eud_min_points_num);

        nh_.getParam("camera/fx",fx);
        nh_.getParam("camera/fy",fy);
        nh_.getParam("camera/cx",cx);
        nh_.getParam("camera/cy",cy);
        nh_.getParam("camera/camera_factor", camera_factor);
        nh_.getParam("camera/pointcloud_frame_id", pointcloud_frame_id);

        nh_.param("pointcloud_generate/dynamic_only", dynamic_only, false);
        nh_.param("pointcloud_generate/segmentation_mode", segmentation_mode, 0);

        // std::cout<<"fx is "<<fx<<std::endl;

        has_odom = false;
        has_color = false;
        has_depth = false;
        has_pose = false;
        
        return 0;
    }
};

#endif 