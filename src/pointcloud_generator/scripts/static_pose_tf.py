#!/home/patrick/miniconda3/envs/ros-noetic/bin/python3
import rospy
from geometry_msgs.msg import PoseStamped
import tf.transformations
import tf2_ros
from geometry_msgs.msg import TransformStamped
import logging

def publish_camera_pose():
    rospy.init_node('camera_pose_publisher', anonymous=True)
    
    rospy.loginfo("Starting camera pose and TF publisher...")
    
    # Get frame IDs from ROS parameters
    parent_frame_id = rospy.get_param('~parent_frame_id', 'base_link') # world
    child_frame_id = rospy.get_param('~child_frame_id', 'camera_depth_optical_frame') # camera_world_optical_frame
    pub_rate = rospy.get_param('~publish_rate', 60)  # Hz
    pose_only = rospy.get_param('~pose_only', False)  # Only publish pose if True

    rospy.loginfo(f"Using parent frame ID: {parent_frame_id}")
    rospy.loginfo(f"Using child frame ID: {child_frame_id}")
    
    # 1. 相机姿态发布
    pose_pub = rospy.Publisher('/camera/current_pose', PoseStamped, queue_size=1)
    rate = rospy.Rate(pub_rate)
    camera_pose = PoseStamped()
    
    camera_pose.header.frame_id = parent_frame_id  # 使用参数化的参考坐标系
    
    camera_pose.pose.position.x = 0.0
    camera_pose.pose.position.y = 0.0
    camera_pose.pose.position.z = 0.0

    q = tf.transformations.quaternion_from_euler(0, 0, -1.5708)  # Roll, Pitch, Yaw
    camera_pose.pose.orientation.x = q[0]
    camera_pose.pose.orientation.y = q[1]
    camera_pose.pose.orientation.z = q[2]
    camera_pose.pose.orientation.w = q[3]
    
    rospy.loginfo(f"Position: {camera_pose.pose.position.x}, {camera_pose.pose.position.y}, {camera_pose.pose.position.z}")
    rospy.loginfo(f"Orientation: {camera_pose.pose.orientation.x}, {camera_pose.pose.orientation.y}, {camera_pose.pose.orientation.z}, {camera_pose.pose.orientation.w}")
    
    # 2. 发布TF
    tf_broadcaster = tf2_ros.TransformBroadcaster()
    transform = TransformStamped()
    
    transform.header.frame_id = parent_frame_id
    transform.child_frame_id = child_frame_id
    
    transform.transform.translation.x = camera_pose.pose.position.x
    transform.transform.translation.y = camera_pose.pose.position.y
    transform.transform.translation.z = camera_pose.pose.position.z
    transform.transform.rotation = camera_pose.pose.orientation
    
    # 循环发布姿态消息和TF
    while not rospy.is_shutdown():
        current_time = rospy.Time.now()
        camera_pose.header.stamp = current_time
        transform.header.stamp = current_time
        
        pose_pub.publish(camera_pose)
        if not pose_only:

            tf_broadcaster.sendTransform(transform)
        
            rospy.loginfo_throttle(10, "Publishing camera pose and TF...")
        
        rate.sleep()

if __name__ == '__main__':
    try:
        publish_camera_pose()
    except rospy.ROSInterruptException:
        rospy.loginfo("Camera pose publisher shutting down...")
        pass