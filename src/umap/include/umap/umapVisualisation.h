#ifndef UMAP_VISUALISATION_H
#define UMAP_VISUALISATION_H

#include "umapType.h"
#include "params.h"
#include "umapProcess.h"
#include <ros/ros.h>
#include <sensor_msgs/Image.h>

class umapVisualisation
{
private:
    umapParams params_;

    // ros::NodeHandle nh_;
    // ros::Publisher u_map_pub_;

    umapProcess umapResults; // Umap processing resuslts
    
public:
    umapVisualisation();
    ~umapVisualisation();

    // cv::Mat visualizeUmap(const cv::Mat& hist_matrix, const cv::Mat& depth_val);
};

#endif // UMAP_VISUALISATION_H