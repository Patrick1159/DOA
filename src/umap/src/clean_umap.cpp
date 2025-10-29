#include "ros/ros.h"
#include "umap/umapProcess.h"

int main(int argc, char** argv)
{
    ros::init(argc, argv, "umap_node");
    ros::NodeHandle nh;
    ROS_INFO("Starting umap node...");

    umapProcess processor;

    ros::spin();
    
    return 0;
}
