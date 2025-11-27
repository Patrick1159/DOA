#include "pointcloud_generator/pointcloud_process.h"

void PointCloudProcessor::odom_callback(const nav_msgs::Odometry::ConstPtr& msg){
    uav_to_world = Eigen::Matrix4d::Identity();
    uav_to_world.block<3,3>(0,0) = Eigen::Quaterniond(msg->pose.pose.orientation.w, msg->pose.pose.orientation.x, msg->pose.pose.orientation.y, msg->pose.pose.orientation.z).toRotationMatrix();
    uav_to_world.block<3,1>(0,3) = Eigen::Vector3d(msg->pose.pose.position.x, msg->pose.pose.position.y, msg->pose.pose.position.z);
    has_odom = true;


}

void PointCloudProcessor::color_callback(const sensor_msgs::ImageConstPtr& msg){
    try
    {
        cv_bridge::CvImagePtr color_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        has_color = true;

        color_pic = color_ptr->image;
        
        cv::waitKey(50); 
    }
    catch (cv_bridge::Exception& e )
    {
        ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
    }
    
}

/// @brief Extract corresponding 3D bounding boxes from obstacles message, For segmentation
/// @param obstacles_msg 
void PointCloudProcessor::obstacles_callback(const umap::Obstacles &obstacles_msg){
    ROS_INFO("Received obstacles message with %zu boxes", obstacles_msg.box_ids.size());
    // TODO: 提取出obstacle对应的范围，然后筛选点云数据，只保留当前3D box中的点云
    std::vector<edge_points> edge_pts;
    for (size_t i = 0; i < obstacles_msg.box_ids.size(); i++) {
        // 1. HW limit
        edge_points edge_pt;
        int ltpt_x = obstacles_msg.box_x[i];
        int ltpt_y = obstacles_msg.box_y[i];
        int width = obstacles_msg.box_width[i];
        int height = obstacles_msg.box_height[i];
        edge_pt.top_left_pt = std::pair<int, int>(ltpt_x, ltpt_y);
        edge_pt.top_right_pt = std::pair<int, int>(ltpt_x + width, ltpt_y);
        edge_pt.bottom_left_pt = std::pair<int, int>(ltpt_x, ltpt_y + height);
        edge_pt.bottom_right_pt = std::pair<int, int>(ltpt_x + width, ltpt_y + height);
        edge_pts.push_back(edge_pt);

        // 2. Depth limit
        depth_range d_range;
        d_range.min_depth = obstacles_msg.centroid_x[i] - obstacles_msg.thickness[i] / 2.0; // z
        d_range.max_depth = obstacles_msg.centroid_x[i] + obstacles_msg.thickness[i] / 2.0; // z
        curr_depth_ranges.push_back(d_range);
        ROS_DEBUG("Obstacle %zu: Center (x, y, z) = (%f, %f, %f), Width = %f, Height = %f, Thickness = %f",
                 i, obstacles_msg.centroid_x[i], obstacles_msg.centroid_y[i], obstacles_msg.centroid_z[i],
                 obstacles_msg.width[i], obstacles_msg.height[i], obstacles_msg.thickness[i]);
        ROS_DEBUG("Obstacle %zu: Depth range [%f, %f]", i, d_range.min_depth, d_range.max_depth);
        }
    curr_edge_pts = edge_pts;
    try {
        cv_bridge::CvImagePtr mask_ptr = cv_bridge::toCvCopy(obstacles_msg.binary_mask, sensor_msgs::image_encodings::MONO8);
        cv::Mat single_mask = mask_ptr->image;
        curr_masks.clear();
        curr_masks.push_back(single_mask);
        ROS_INFO("Binary mask contains %d non-zero pixels", cv::countNonZero(single_mask));
    } catch (cv_bridge::Exception& e) {
        ROS_ERROR("Could not convert binary_mask from '%s' to 'mono8'.", obstacles_msg.binary_mask.encoding.c_str());
    }
}


void PointCloudProcessor::depth_callback(const sensor_msgs::ImageConstPtr& msg){
    try
    {
        cv_bridge::CvImagePtr depth_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_32FC1);
        has_depth = true;
        depth_pic = depth_ptr->image;

        double cluster_tolerance = eud_range; 
        int min_cluster_size = eud_min_points_num;     
        int max_cluster_size = INT_MAX; 
        int min_points_per_cluster = eud_min_points_num; 

        // if(!has_color || !has_pose || !has_odom) return; // To verify if all the data is ready
        if(!has_color || !has_pose) { // For test
            if (!has_color)
                ROS_WARN("Color image not received yet.");
            if (!has_pose)
                ROS_WARN("Pose not received yet. run python script: static_pose.py!");
            return;
        }

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr raw_pc(new pcl::PointCloud<pcl::PointXYZRGB>);
        for (int m = 0; m < depth_pic.rows; m++)
        {
            for (int n = 0; n < depth_pic.cols; n++)
            {
                float d = depth_pic.ptr<float>(m)[n];
                
                if (d == 0)
                    continue;
                pcl::PointXYZRGB p;

                p.z = double(d) / camera_factor;
                p.x = (n - cx) * p.z / fx;
                p.y = (m - cy) * p.z / fy;

                p.b = color_pic.ptr<uchar>(m)[n*3];
                p.g = color_pic.ptr<uchar>(m)[n*3+1];
                p.r = color_pic.ptr<uchar>(m)[n*3+2];

                // 进行动态障碍物筛选
                if (dynamic_only) { // Segmentation dynamic objects
                    if (segmentation_mode == 0) {
                        bool in_bbox = false;
                        bool in_depth = false;
                        // 1. world系的深度是否在任意一个box中
                        // ROS_INFO("Checking point: (%f, %f, %f)", p.x, p.y, p.z);
                        for (const auto& depth_limit : curr_depth_ranges) {
                            // ROS_INFO("Checking depth: %f against limits: [%f, %f]", p.z, depth_limit.min_depth, depth_limit.max_depth);
                            if (p.z >= depth_limit.min_depth && p.z <= depth_limit.max_depth) {
                                in_depth = true;
                                break;
                            }
                        }
                        if (!in_depth)
                            continue;
                        // 2. 在任意一个box中则true
                        for (const auto& edge_pt : curr_edge_pts) {
                            if (n >= edge_pt.top_left_pt.first && n <= edge_pt.top_right_pt.first &&
                                m >= edge_pt.top_left_pt.second && m <= edge_pt.bottom_left_pt.second) {
                                in_bbox = true;
                                break;
                            }
                        }
                        if (!in_bbox)
                            continue;
                    }else if (segmentation_mode == 1) {
                        bool in_mask = false;
                        // Check if the point is within any mask
                        for (const auto& mask : curr_masks) {
                            if (mask.at<uchar>(m, n) != 0) {
                                in_mask = true;
                                break;
                            }
                        }
                        if (!in_mask)
                            continue;
                    }else if (segmentation_mode == 2) {
                        bool in_bbox = false;
                        for (const auto& edge_pt : curr_edge_pts) {
                            if (n >= edge_pt.top_left_pt.first && n <= edge_pt.top_right_pt.first &&
                                m >= edge_pt.top_left_pt.second && m <= edge_pt.bottom_left_pt.second) {
                                in_bbox = true;
                                break;
                            }
                        }

                        bool in_mask = false;
                        for (const auto& mask : curr_masks) {
                            if (mask.at<uchar>(m, n) != 0) {
                                in_mask = true;
                                break;
                            }
                        }

                        if (!(in_bbox && in_mask))
                            continue;
                    }
                    
                }

                // 把p加入到点云中
                raw_pc->points.push_back( p );
            }
        }
        raw_pc->height = 1;
        raw_pc->width = raw_pc->points.size();

        ROS_INFO("Total points in raw point cloud: %ld", raw_pc->size());
        // 直通滤波
        pcl::PassThrough<pcl::PointXYZRGB> pass_z;
        pass_z.setInputCloud(raw_pc);
        pass_z.setFilterFieldName("z");
        pass_z.setFilterLimits(0.0,max_depth);
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr pass_z_pc(new pcl::PointCloud<pcl::PointXYZRGB>);
        pass_z.filter(*pass_z_pc);
        pass_z_pc->height = 1;
        pass_z_pc->width = pass_z_pc->points.size();
        ROS_INFO("Points after pass-through filter: %d", pass_z_pc->size());

        // 体素滤波
        pcl::VoxelGrid<pcl::PointXYZRGB> vox;
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr vox_pc(new pcl::PointCloud<pcl::PointXYZRGB>);
        vox.setInputCloud(pass_z_pc);
        vox.setLeafSize(voxel_size, voxel_size, voxel_size);
        vox.filter(*vox_pc);
        vox_pc->height = 1;
        vox_pc->width = vox_pc->points.size();
        vox_pc->is_dense = false;
        ROS_INFO("Points after voxel grid filter: %ld", vox_pc->size());

        // 噪点去除
        pcl::EuclideanClusterExtraction<pcl::PointXYZRGB> euclidean_cluster;
        euclidean_cluster.setInputCloud(vox_pc);
        euclidean_cluster.setClusterTolerance(cluster_tolerance); 
        euclidean_cluster.setMinClusterSize(min_cluster_size);   
        euclidean_cluster.setMaxClusterSize(max_cluster_size); 
        std::vector<pcl::PointIndices> cluster_indices;
        euclidean_cluster.extract(cluster_indices);
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr filtered_pc(new pcl::PointCloud<pcl::PointXYZRGB>);
        for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin(); it != cluster_indices.end(); ++it)
        {
            if (it->indices.size() >= min_points_per_cluster)
            {
                for (std::vector<int>::const_iterator pit = it->indices.begin(); pit != it->indices.end(); ++pit)
                {
                    filtered_pc->points.push_back(vox_pc->points[*pit]);
                }
            }
        }
        filtered_pc->width = filtered_pc->points.size();
        filtered_pc->height = 1;
        filtered_pc->is_dense = false;
        ROS_INFO("Points after euclidean cluster filter: %ld", filtered_pc->size());

        // 自身滤除
        pcl::CropBox<pcl::PointXYZRGB> crop_box_filter;
        Eigen::Vector4f min_pt(-min_depth, -min_depth, -min_depth, 1.0); // 最小点
        Eigen::Vector4f max_pt(min_depth, min_depth, min_depth, 1.0);    // 最大点
        crop_box_filter.setInputCloud(filtered_pc);
        crop_box_filter.setMin(min_pt);
        crop_box_filter.setMax(max_pt);
        crop_box_filter.setNegative(true); 
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr filtered_self_pc(new pcl::PointCloud<pcl::PointXYZRGB>);
        crop_box_filter.filter(*filtered_self_pc);
        ROS_INFO("Points after self cluster filter: %ld", filtered_self_pc->size());

        global_pc.reset(new pcl::PointCloud<pcl::PointXYZRGB>);
        
        pcl::transformPointCloud(*filtered_self_pc, *global_pc, cam_to_world); 
        // ROS_INFO("Final points after all filters: %d", base_pc->size());

        sensor_msgs::PointCloud2 pc_msg;
        pcl::toROSMsg(*global_pc,pc_msg);
        pc_msg.header.frame_id = pointcloud_frame_id;
        pc_msg.header.stamp = ros::Time::now();
        pc_pub.publish(pc_msg);
        ROS_INFO("Published point cloud with frame_id: %s", pc_msg.header.frame_id.c_str());
        ROS_INFO("Published point cloud with %ld points", global_pc->size());
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("Could not convert from '%s' to 'mono16'.", msg->encoding.c_str());
    }

}

void PointCloudProcessor::pose_callback(const geometry_msgs::PoseStamped::ConstPtr& msg){
    cam_to_world= Eigen::Matrix4d::Identity();
    Eigen::Matrix3d rotx;
    rotx << 1, 0, 0,
            0, 0, 1,
            0, -1, 0;
    cam_to_world.block<3,3>(0,0) = 
    Eigen::Quaterniond(
        msg->pose.orientation.w, 
        msg->pose.orientation.x, 
        msg->pose.orientation.y, 
        msg->pose.orientation.z).toRotationMatrix()*rotx; // With correction
        
    // cam_to_world.block<3,3>(0,0) =
    //     Eigen::Quaterniond(msg->pose.orientation.w,
    //                         msg->pose.orientation.x,
    //                         msg->pose.orientation.y,
    //                         msg->pose.orientation.z).toRotationMatrix(); // without correction
    
    
    cam_to_world.block<3,1>(0,3) = Eigen::Vector3d(msg->pose.position.x, msg->pose.position.y, msg->pose.position.z);

    has_pose = true;
    
}
