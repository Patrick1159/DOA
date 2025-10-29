/// @file dynamic_detector.h
/// @brief: Header file for dynamic_detector.cpp
/// @details: Define structures and types used in dynamic detection
/// @description: 实现功能：3d box -> tracker, 判断

#include <umap/KalmanFilter.h>
#include <umap/ObstacleTracker.h>
#include <umap/umapProcess.h>
#include <umap/umapType.h>

#include <Eigen/Dense>
#include <vector>
#include <thread>
#include <chrono>
#include <iostream>
#include <ros/ros.h>
#include <visualization_msgs/MarkerArray.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>
#include <std_msgs/ColorRGBA.h>

#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>


class dynamicDetector {
private:
    dynamicDetector(ros::NodeHandle& nh);
    ~dynamicDetector();

    void boxes3DtoTracker(const umap::BoundingBoxes3D& boxes3D_msg); // Callback for 3D bounding boxes
};