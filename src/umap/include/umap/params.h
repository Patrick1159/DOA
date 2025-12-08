/// @brief Params used in each module

#ifndef UMAP_PARAMS_H
#define UMAP_PARAMS_H

#include <ros/ros.h>

struct umapParams {
    int max_dist = 5000;
    int min_dist = 500;
    float depth_factor = 1.0f;
    int hist_size = 150;
    float threshold_point = 0.05f;
    float threshold_line = 5.0f;
    int min_length_line = 30;
    int row_downsample = 1; // default 1
    bool inverse_depth = false;  // 控制是否翻转深度图

    std::string camera_frame = "camera_color_optical_frame"; // default frame id
    float fx = 525.0;
    float fy = 525.0;
    float px = 319.5;
    float py = 239.5;
};

struct trackerParams{
    double DYNAMIC_IOU_THRESHOLD; // IoU threshold for in matching process
    double DYNAMIC_DIST_THRESHOLD; // m threshold for dynamic obstacles
    double iou_weight;
    double dist_weight;

    double DYNAMIC_VELOCITY_THRESHOLD; // m/s threshold for dynamic obstacles
    double DYNAMIC_VELOCITY_MAX; // m/s threshold for dynamic obstacles
    double SUSTAINED_VELOCITY_THRESHOLD;
    int SUSTAINED_VELOCITY_COUNT; // number of frames to check sustained velocity
    double MAX_TRACKER_AGE; // seconds, max age of a tracker without updates
    double UNMATCHED_PUBLISH_TIMEOUT; // seconds, how long to keep publishing unmatched dynamic trackers
    double EXPANSION_RATE = 0.3; // per second dimension expansion rate for dynamic trackers
    
    std::string COARSE_VIS_FRAME = "camera_color_optical_frame"; // default frame id
    double AVERAGE_DEPLACEMENT_THRESHOLD; // m threshold for average displacement of a tracker
    int HISTORY_SIZE; // 每个tracker的历史vel和pos留存长度
    int STATUS_HISTORY_SIZE; // 每个tracker的历史STATUS留存长度
    int MIN_CONSECUTIVE_FRAMES_FOR_STATUS_CHANGE; // tracker改变状态所需要的最小稳定帧, <= STATUS_HISTORY_SIZE!
};

struct rematchParams{
    double SIMILARITY_THRESHOLD;
    double SIMILARITY_WEIGHT;

    double DISTANCE_THRESHOLD;
    double DISTANCE_WEIGHT;

    double IOU_THRESHOLD;
    double IOU_WEIGHT; // TODO: 在rematch中订阅yaml的参数更新

    double MAX_TRACKER_AGE; // seconds, max age of a tracker without updates
    double UNMATCHED_PUBLISH_TIMEOUT; // seconds, how long to keep publishing unmatched trackers
    double EXPANSION_RATE = 0.3; // per second dimension expansion rate for dynamic trackers

    std::string REMATCH_VIS_FRAME = "camera_color_optical_frame";
};

#endif // UMAP_PARAMS_H