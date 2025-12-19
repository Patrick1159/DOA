#!/usr/bin/env python3

import rospy
import math
from umap.msg import TrackerDetails
from geometry_msgs.msg import Point, Vector3

def publish_dummy_obstacles():
    rospy.init_node('dummy_tracker_publisher', anonymous=True)
    pub = rospy.Publisher('/rematch/tracker_details', TrackerDetails, queue_size=10)
    rate = rospy.Rate(10) # 10hz

    start_time = rospy.Time.now().to_sec()

    rospy.loginfo("Starting dummy tracker publisher...")

    while not rospy.is_shutdown():
        current_time = rospy.Time.now().to_sec()
        elapsed = current_time - start_time

        msg = TrackerDetails()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "odom" # Assuming odom frame for testing

        # Obstacle 1: Moving in a circle
        # Radius 2.0, centered at (0,0)
        angle = elapsed * 0.5 # 0.5 rad/s
        r = 2.0
        x1 = r * math.cos(angle)
        y1 = r * math.sin(angle)
        
        # Velocity is tangent to the circle
        vx1 = -r * 0.5 * math.sin(angle)
        vy1 = r * 0.5 * math.cos(angle)

        msg.tracker_ids.append(1)
        msg.positions.append(Point(x=x1, y=y1, z=0.0))
        msg.velocities.append(Vector3(x=vx1, y=vy1, z=0.0))
        msg.box_dimensions.append(Vector3(x=0.5, y=0.5, z=1.0))
        msg.last_update_times.append(current_time)
        msg.ages.append(elapsed)

        # Obstacle 2: Moving back and forth along X axis
        # Period 10s
        x2 = 3.0 + 2.0 * math.sin(elapsed * 2 * math.pi / 10.0)
        y2 = 3.0
        vx2 = 2.0 * (2 * math.pi / 10.0) * math.cos(elapsed * 2 * math.pi / 10.0)
        vy2 = 0.0

        msg.tracker_ids.append(2)
        msg.positions.append(Point(x=x2, y=y2, z=0.0))
        msg.velocities.append(Vector3(x=vx2, y=vy2, z=0.0))
        msg.box_dimensions.append(Vector3(x=0.6, y=0.6, z=1.0))
        msg.last_update_times.append(current_time)
        msg.ages.append(elapsed)

        # Obstacle 3: Static obstacle at (-2, -2)
        msg.tracker_ids.append(3)
        msg.positions.append(Point(x=-2.0, y=-2.0, z=0.0))
        msg.velocities.append(Vector3(x=0.0, y=0.0, z=0.0))
        msg.box_dimensions.append(Vector3(x=0.8, y=0.8, z=1.0))
        msg.last_update_times.append(current_time)
        msg.ages.append(elapsed)

        pub.publish(msg)
        rate.sleep()

if __name__ == '__main__':
    try:
        publish_dummy_obstacles()
    except rospy.ROSInterruptException:
        pass
