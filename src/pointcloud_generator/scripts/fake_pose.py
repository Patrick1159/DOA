#!/home/cius/miniconda3/envs/DOA/bin/python3

import rospy
import tf2_ros
from geometry_msgs.msg import TransformStamped
import math
from geometry_msgs.msg import PoseStamped


def publish_circular_pose():
    rospy.init_node('fake_pose_publisher')
    
    br = tf2_ros.TransformBroadcaster()
    
    rate = rospy.Rate(10)  # 10Hz
    
    # Circle parameters
    radius = 2.0  # 圆的半径
    height = 1.0  # 相机高度
    angular_speed = 0.5  # 角速度 (rad/s)
    
    start_time = rospy.Time.now().to_sec()
    
    while not rospy.is_shutdown():
        current_time = rospy.Time.now()
        elapsed_time = current_time.to_sec() - start_time
        
        # Calculate position on circle
        angle = angular_speed * elapsed_time
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        z = height
        
        # Create transform message
        t = TransformStamped()
        t.header.stamp = current_time
        t.header.frame_id = "map"
        t.child_frame_id = "camera_link"
        
        t.transform.translation.x = x
        t.transform.translation.y = y
        t.transform.translation.z = z
        
        # Make camera always look towards center
        yaw = angle + math.pi  # Camera looks inward
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = math.sin(yaw / 2.0)
        t.transform.rotation.w = math.cos(yaw / 2.0)
        
        br.sendTransform(t)
        rate.sleep()
        # publish a PoseStamped topic for the camera
        if 'pose_pub' not in locals():
            pose_pub = rospy.Publisher('camera_pose', PoseStamped, queue_size=10)
            rospy.sleep(0.1)  # give ROS a moment to register the publisher

        ps = PoseStamped()
        ps.header.stamp = current_time
        ps.header.frame_id = "map"
        ps.pose.position.x = x
        ps.pose.position.y = y
        ps.pose.position.z = z
        ps.pose.orientation = t.transform.rotation

        pose_pub.publish(ps)
if __name__ == '__main__':
    try:
        publish_circular_pose()
    except rospy.ROSInterruptException:
        pass