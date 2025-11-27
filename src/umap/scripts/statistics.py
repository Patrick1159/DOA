#!/usr/bin/env python3

"""
Statistics script to log tracker details to CSV file.
Subscribes to /rematch/tracker_details topic and saves data to CSV.
"""

import rospy
import csv
import os
from datetime import datetime
from umap.msg import TrackerDetails

class TrackerStatistics:
    def __init__(self):
        rospy.init_node('tracker_statistics', anonymous=True)
        
        # Create CSV file with timestamp
        self.csv_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'benchmarks')
        if not os.path.exists(self.csv_dir):
            os.makedirs(self.csv_dir)
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_filename = os.path.join(self.csv_dir, f'tracker_statistics_{timestamp}.csv')
        
        # Initialize CSV file with headers
        with open(self.csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['time', 'tracker_id', 'pos_x', 'pos_y', 'pos_z', 'vel_x', 'vel_y', 'vel_z'])
        
        # Subscribe to tracker details topic
        self.subscriber = rospy.Subscriber('/rematch/tracker_details', TrackerDetails, self.tracker_callback)
        
        rospy.loginfo(f"TrackerStatistics node started. Logging to: {self.csv_filename}")
        
    def tracker_callback(self, msg):
        """Callback function to process tracker details messages"""
        try:
            with open(self.csv_filename, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                
                # Convert ROS time to timestamp
                timestamp = msg.header.stamp.to_sec()
                
                # Write data for each tracker
                for i in range(len(msg.tracker_ids)):
                    row = [
                        timestamp,
                        msg.tracker_ids[i],
                        msg.positions[i].x,
                        msg.positions[i].y,
                        msg.positions[i].z,
                        msg.velocities[i].x,
                        msg.velocities[i].y,
                        msg.velocities[i].z
                    ]
                    writer.writerow(row)
                    
            rospy.loginfo(f"Logged {len(msg.tracker_ids)} tracker entries at time {timestamp}")
            
        except Exception as e:
            rospy.logerr(f"Error writing to CSV file: {e}")
            
    def run(self):
        """Main run loop"""
        rospy.loginfo("Tracker statistics logger is running...")
        rospy.spin()

if __name__ == '__main__':
    try:
        stats = TrackerStatistics()
        stats.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("Tracker statistics node interrupted.")
    except Exception as e:
        rospy.logerr(f"Error in tracker statistics node: {e}")