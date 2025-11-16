#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <visualization_msgs/MarkerArray.h>
#include <nav_msgs/Path.h>

class BboxFrameConverter
{
private:
    ros::NodeHandle nh_;
    ros::Subscriber camera_pose_sub_;
    ros::Subscriber obstacle_paths_sub_;
    ros::Subscriber tracked_obstacles_sub_;
    ros::Subscriber text_markers_sub_;
    
    ros::Publisher obstacle_paths_pub_;
    ros::Publisher tracked_obstacles_pub_;
    ros::Publisher text_markers_pub_;
    
    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;
    tf2_ros::TransformBroadcaster tf_broadcaster_;
    
    geometry_msgs::PoseStamped camera_pose_;
    bool camera_pose_received_;

public:
    BboxFrameConverter() : tf_listener_(tf_buffer_), camera_pose_received_(false)
    {
        // Subscribers
        camera_pose_sub_ = nh_.subscribe("/camera/current_pose", 1, &BboxFrameConverter::cameraPoseCallback, this);
        obstacle_paths_sub_ = nh_.subscribe("/rematch/obstacle_paths", 1, &BboxFrameConverter::obstaclePathsCallback, this);
        tracked_obstacles_sub_ = nh_.subscribe("/rematch/visualization/tracked_obstacles", 1, &BboxFrameConverter::trackedObstaclesCallback, this);
        text_markers_sub_ = nh_.subscribe("/rematch/visualizationtext_markers", 1, &BboxFrameConverter::textMarkersCallback, this);
        
        // Publishers
        obstacle_paths_pub_ = nh_.advertise<nav_msgs::Path>("/rematch/obstacle_paths_world", 1);
        tracked_obstacles_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("/rematch/visualization/tracked_obstacles_world", 1);
        text_markers_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("/rematch/visualizationtext_markers_world", 1);
    }

    void cameraPoseCallback(const geometry_msgs::PoseStamped::ConstPtr& msg)
    {
        camera_pose_ = *msg;
        camera_pose_received_ = true;
        
        // Broadcast transform from world to d400_color_optical_frame
        geometry_msgs::TransformStamped transform;
        transform.header.stamp = ros::Time::now();
        transform.header.frame_id = "world";
        transform.child_frame_id = "d400_color_optical_frame";
        transform.transform.translation.x = msg->pose.position.x;
        transform.transform.translation.y = msg->pose.position.y;
        transform.transform.translation.z = msg->pose.position.z;
        transform.transform.rotation = msg->pose.orientation;
        
        tf_broadcaster_.sendTransform(transform);
    }

    void obstaclePathsCallback(const nav_msgs::Path::ConstPtr& msg)
    {
        if (!camera_pose_received_) return;
        
        nav_msgs::Path converted_path = *msg;
        converted_path.header.frame_id = "world";
        
        for (auto& pose_stamped : converted_path.poses)
        {
            try
            {
                geometry_msgs::PoseStamped transformed_pose;
                tf_buffer_.transform(pose_stamped, transformed_pose, "world", ros::Duration(0.1));
                pose_stamped = transformed_pose;
            }
            catch (tf2::TransformException& ex)
            {
                ROS_WARN("Failed to transform pose: %s", ex.what());
            }
        }
        
        obstacle_paths_pub_.publish(converted_path);
    }

    void trackedObstaclesCallback(const visualization_msgs::MarkerArray::ConstPtr& msg)
    {
        if (!camera_pose_received_) return;
        
        visualization_msgs::MarkerArray converted_markers = *msg;
        
        for (auto& marker : converted_markers.markers)
        {
            marker.header.frame_id = "world";
            try
            {
                geometry_msgs::PoseStamped pose_in, pose_out;
                pose_in.header = marker.header;
                pose_in.pose = marker.pose;
                
                tf_buffer_.transform(pose_in, pose_out, "world", ros::Duration(0.1));
                marker.pose = pose_out.pose;
            }
            catch (tf2::TransformException& ex)
            {
                ROS_WARN("Failed to transform marker: %s", ex.what());
            }
        }
        
        tracked_obstacles_pub_.publish(converted_markers);
    }

    void textMarkersCallback(const visualization_msgs::MarkerArray::ConstPtr& msg)
    {
        if (!camera_pose_received_) return;
        
        visualization_msgs::MarkerArray converted_markers = *msg;
        
        for (auto& marker : converted_markers.markers)
        {
            marker.header.frame_id = "world";
            try
            {
                geometry_msgs::PoseStamped pose_in, pose_out;
                pose_in.header = marker.header;
                pose_in.pose = marker.pose;
                
                tf_buffer_.transform(pose_in, pose_out, "world", ros::Duration(0.1));
                marker.pose = pose_out.pose;
            }
            catch (tf2::TransformException& ex)
            {
                ROS_WARN("Failed to transform text marker: %s", ex.what());
            }
        }
        
        text_markers_pub_.publish(converted_markers);
    }
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "bbox_frame_converter");
    BboxFrameConverter converter;
    ros::spin();
    return 0;
}