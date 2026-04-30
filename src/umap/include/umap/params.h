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
    bool inverse_depth = false;  // 控制是否翻转深度图
    bool enable_horizontal_merge = false;
    int merge_horizontal_gap = 30;
    int merge_max_row_gap = 8;
    float merge_min_vertical_overlap_ratio = 0.3f;

    bool enable_adaptive_horizontal_merge = true;
    int adaptive_merge_min_gap = 20;
    int adaptive_merge_max_gap = 100;
    float adaptive_merge_near_ratio = 0.25f;
    float adaptive_merge_far_ratio = 0.85f;

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

    // Candidate publishing: trackers that have at least 1 KF-estimated velocity
    // and whose latest |v| > CANDIDATE_VELOCITY_THRESHOLD are published with
    // confidence=0.5 even before the SUSTAINED hysteresis fires. Set to a
    // large value (e.g. 1e6) to disable.
    double CANDIDATE_VELOCITY_THRESHOLD = 0.2;

    std::string COARSE_VIS_FRAME = "camera_color_optical_frame"; // visualization frame for RViz markers
    std::string COARSE_OUTPUT_FRAME = ""; // frame_id stamped on obstacles_msg downstream; empty -> fallback to COARSE_VIS_FRAME
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
    std::string REMATCH_OUTPUT_FRAME = ""; // frame_id stamped on tracker_details / obstacles downstream; empty -> fallback to REMATCH_VIS_FRAME
};

#endif // UMAP_PARAMS_H