import rospy
import tf2_ros
from geometry_msgs.msg import PoseStamped, TransformStamped

#!/usr/bin/env python3
"""
Subscribe to /camera/current_pose (geometry_msgs/PoseStamped, in world frame)
and publish a TF from "world" -> "t265_link" based on that pose.
"""


def pose_callback(msg, args):
    broadcaster, parent_frame, child_frame, last_stamp = args
    t = TransformStamped()
    
    # use message timestamp if valid, otherwise use now
    if msg.header.stamp.secs == 0 and msg.header.stamp.nsecs == 0:
        t.header.stamp = rospy.Time.now()
    else:
        t.header.stamp = msg.header.stamp
    
    # Skip if timestamp is the same as the last one to avoid TF_REPEATED_DATA warning
    if last_stamp['stamp'] is not None and t.header.stamp == last_stamp['stamp']:
        rospy.logdebug("Skipping duplicate timestamp: %s", t.header.stamp)
        return
    
    # Update last timestamp
    last_stamp['stamp'] = t.header.stamp
    
    t.header.frame_id = parent_frame
    t.child_frame_id = child_frame

    # Fill translation from pose
    t.transform.translation.x = msg.pose.position.x
    t.transform.translation.y = msg.pose.position.y
    t.transform.translation.z = msg.pose.position.z

    # Fill rotation from pose (assumed normalized quaternion)
    t.transform.rotation = msg.pose.orientation

    try:
        broadcaster.sendTransform(t)
    except Exception as e:
        rospy.logwarn_throttle(5.0, "Failed to send transform: %s" % str(e))

def main():
    rospy.init_node('pose_to_tf', anonymous=False)

    topic = rospy.get_param('~pose_topic', '/camera/current_pose')
    parent_frame = rospy.get_param('~parent_frame', 'world')
    child_frame = rospy.get_param('~child_frame', 't265_link')
    queue_size = rospy.get_param('~queue_size', 1)

    broadcaster = tf2_ros.TransformBroadcaster()
    
    # Track last timestamp to avoid duplicates
    last_stamp = {'stamp': None}

    rospy.Subscriber(topic, PoseStamped, pose_callback, 
                    (broadcaster, parent_frame, child_frame, last_stamp), 
                    queue_size=queue_size)

    rospy.loginfo("pose_to_tf: subscribing to %s, publishing TF %s -> %s", topic, parent_frame, child_frame)
    rospy.spin()

if __name__ == '__main__':
    main()