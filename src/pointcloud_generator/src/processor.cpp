#include "pointcloud_generator/pointcloud_process.h"

int main(int argc, char **argv){
    ros::init(argc, argv, "pointcloud_process_node");

    ros::NodeHandle nh("~");
    PointCloudProcessor pcp;
    pcp.init(nh);

    ros::spin();

    return 0;
}