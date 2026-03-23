#!/home/user/miniconda3/envs/oa/bin/python3
import rospy
from geometry_msgs.msg import PoseStamped
import tf.transformations
import tf2_ros
from geometry_msgs.msg import TransformStamped
import math

class D435PosePublisher:
    def __init__(self):
        rospy.init_node('d435_pose_publisher', anonymous=False)
        self.pose_pub = rospy.Publisher('/camera/current_pose', PoseStamped, queue_size=1)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.yaw_correction = tf.transformations.quaternion_from_euler(math.pi/2, 0, 0)
        self.parent_frame_id = rospy.get_param('~parent_frame_id', 'odom')
        self.child_frame_id = rospy.get_param('~child_frame_id', 'd400_color_optical_frame')
        self.rate = rospy.Rate(rospy.get_param('~publish_rate', 60))

        rospy.loginfo(f"Listening to tf {self.parent_frame_id} -> {self.child_frame_id}")

    def publish_camera_pose(self):
        tf_world_to_camera = self.tf_buffer.lookup_transform(self.parent_frame_id, self.child_frame_id, rospy.Time(0))
        # rospy.loginfo(f"TF: {tf_world_to_camera}")

        pose = PoseStamped()
        pose.header.stamp = tf_world_to_camera.header.stamp
        pose.header.frame_id = self.parent_frame_id  # pose of the camera in world
        pose.pose.position.x = tf_world_to_camera.transform.translation.x
        pose.pose.position.y = tf_world_to_camera.transform.translation.y
        pose.pose.position.z = tf_world_to_camera.transform.translation.z
        
        original_quat = tf_world_to_camera.transform.rotation
        original_q = [original_quat.x, original_quat.y, original_quat.z, original_quat.w]
        corrected_q = tf.transformations.quaternion_multiply(original_q, self.yaw_correction) # rotation fix
        # corrected_q = original_q # no rotation fix
        pose.pose.orientation.x = corrected_q[0]
        pose.pose.orientation.y = corrected_q[1]
        pose.pose.orientation.z = corrected_q[2]
        pose.pose.orientation.w = corrected_q[3]

        self.pose_pub.publish(pose)

    def spin(self):
        while not rospy.is_shutdown():
            try:
                self.publish_camera_pose()
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                rospy.logwarn("Transform not available yet")
            self.rate.sleep()


if __name__ == '__main__':
    try:
        d435_pose_publisher = D435PosePublisher()
        d435_pose_publisher.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Camera pose publisher shutting down...")
        pass
