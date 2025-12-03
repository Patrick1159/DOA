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
            
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_filename = os.path.join(self.csv_dir, f'tracker_statistics_{self.timestamp}.csv')
        
        # Initialize CSV file with headers
        with open(self.csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['time', 'tracker_id', 'pos_x', 'pos_y', 'pos_z', 'vel_x', 'vel_y', 'vel_z'])
        
        # Generate processing script
        self.generate_processing_script()
        
        # Subscribe to tracker details topic
        self.subscriber = rospy.Subscriber('/rematch/tracker_details', TrackerDetails, self.tracker_callback)
        
        rospy.loginfo(f"TrackerStatistics node started. Logging to: {self.csv_filename}")
        
    def generate_processing_script(self):
        """Generate a shell script for processing the collected data"""
        script_content = f"""#!/bin/bash

# Auto-generated processing script for timestamp: {self.timestamp}
# Usage: ./process_{self.timestamp}.sh

echo "Starting data processing for timestamp: {self.timestamp}"

# Step 1: Filter the data by tracker index
echo "Step 1: Filtering data by tracker index..."
python3 src/umap/scripts/benchmark_utils/index_filter.py src/benchmarks/tracker_statistics_{self.timestamp}.csv

# Step 2: Clean and generate ground truth files
echo "Step 2: Generating ground truth files..."
python3 src/umap/scripts/benchmark_utils/clean_gt.py src/benchmarks/transverse_GT_camera_frame.csv -o src/benchmarks/tracker_GT --tracker-dir src/benchmarks/tracker_statistics_{self.timestamp}_filtered

# Step 3: Calculate error indicators for all trackers (0-29)
echo "Step 3: Calculating error indicators for all trackers..."
for i in {{0..29}}; do
    echo "Processing tracker_$i"
    python3 src/umap/scripts/benchmark_utils/calculate_indicator.py src/benchmarks/tracker_GT/tracker_${{i}}_tracker_statistics_{self.timestamp}_GT.csv src/benchmarks/tracker_statistics_{self.timestamp}_filtered/tracker_${{i}}_tracker_statistics_{self.timestamp}.csv -o src/benchmarks/error_indicators_test --xy-only
    
    if [ $? -eq 0 ]; then
        echo "tracker_$i processed successfully"
    else
        echo "ERROR: tracker_$i processing failed"
    fi
    echo "---"
done

echo "Data processing completed for timestamp: {self.timestamp}"
"""
        
        script_filename = os.path.join(self.csv_dir, f'process_{self.timestamp}.sh')
        
        with open(script_filename, 'w') as script_file:
            script_file.write(script_content)
        
        # Make the script executable
        os.chmod(script_filename, 0o755)
        
        rospy.loginfo(f"Processing script generated: {script_filename}")
        
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