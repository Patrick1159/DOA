/*
@file: umapType.h
@brief: Header file for umapType.cpp
@description: Define structures and types used in umap processing
*/
#ifndef UMAP_TYPE_H
#define UMAP_TYPE_H

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <sstream>
#include <algorithm> // for std::max
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <mutex> // for std::mutex
#include <Eigen/Dense> // for Eigen::Vector3f
#include <map> // for std::map
#include <umap/BoundingBoxes2D.h>  // msg for 2D bounding boxes
#include <umap/BoundingBoxes3D.h>  // msg for 3D bounding boxes
#include <umap/MultiImageFeatures.h> // msg for clip features
#include <umap/Obstacles.h> // msg for dynamic obstacles


/// @brief 2D bounding box extracted from histogram
struct UVbox
{
    cv::Rect bb;
    int id;
    int toppest_parent_id;

    UVbox(int _id, int row, int start_col, int end_col) : 
        id(_id), 
        toppest_parent_id(_id),
        bb(cv::Point(start_col, row), cv::Point(end_col, row+1)) {}
};

/// @brief 3D bounding 
struct box3D
{
    float x, y, z; // Top-Left
    float x_width, y_width, z_width;
    box3D() : x(0), y(0), z(0), x_width(0), y_width(0), z_width(0) {}
};

// Dynamic detection structures
// deprecated
struct DynamicBbox {
    cv::Rect rgb_bbox;
    Eigen::Vector3f centroid;
    bool is_dynamic;
};

// Tracking structures
// deprecated
struct TrackedCentroid {
    Eigen::Vector3f position;
    int frame_id;
    float total_displacement;
    float velocity;           // Add velocity tracking
    int dynamic_count;        // Count consecutive dynamic detections
    bool is_dynamic;
    
    TrackedCentroid(const Eigen::Vector3f& pos, int id) : 
        position(pos), frame_id(id), total_displacement(0), 
        velocity(0), dynamic_count(0), is_dynamic(false) {}
};

#endif // UMAP_TYPE_H