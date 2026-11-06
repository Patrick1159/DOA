#!/usr/bin/env python3

"""
Subscribe topic /human_pose1 to get ground truth PoseStamped messages, 
optionally transform to camera frame using camera_pose, and export to csv format.

Parameters:
  ~use_camera_transform (bool, default: True): 
    If True, transforms poses to camera frame using /camera/current_pose
    If False, uses original world frame without transformation
    
Usage:
  # With camera frame transformation (default):
  rosrun umap GT_NOKOV.py
  
  # Without camera frame transformation (world frame):
  rosrun umap GT_NOKOV.py _use_camera_transform:=false
"""
import rospy
from geometry_msgs.msg import PoseStamped
import numpy as np
from scipy.spatial.transform import Rotation as R
import os
import csv
import threading

class GTAnalyse:
    def __init__(self):
        rospy.init_node('gt_analyse', anonymous=True)
        
        # Get parameter to control camera frame transformation
        self.use_camera_transform = rospy.get_param('~use_camera_transform', True)
        
        # Initialize camera pose
        self.camera_pose = None
        self.camera_pose_lock = threading.Lock()
        
        # Subscribers
        self.subscriber = rospy.Subscriber('/human_pose_1', PoseStamped, self.human_pose_callback)
        
        # Only subscribe to camera pose if transformation is enabled
        if self.use_camera_transform:
            self.camera_pos_sub = rospy.Subscriber('/camera/current_pose', PoseStamped, self.camera_pose_callback)
            rospy.loginfo("GT Analyse node initialized with camera frame transformation ENABLED")
        else:
            rospy.loginfo("GT Analyse node initialized with camera frame transformation DISABLED (using world frame)")

    def camera_pose_callback(self, pose_msg):
        """Update camera pose for coordinate transformation"""
        with self.camera_pose_lock:
            self.camera_pose = pose_msg
        rospy.loginfo_throttle(10, "Camera pose updated")
    
    def transform_pose_to_camera_frame(self, world_pose):
        """Transform pose from world frame to camera frame"""
        with self.camera_pose_lock:
            if self.camera_pose is None:
                rospy.logwarn("No camera pose available, cannot transform")
                return None
            
            camera_pose = self.camera_pose
        
        try:
            # Extract world pose
            world_pos = np.array([world_pose.pose.position.x, 
                                world_pose.pose.position.y, 
                                world_pose.pose.position.z])
            world_quat = [world_pose.pose.orientation.x,
                         world_pose.pose.orientation.y,
                         world_pose.pose.orientation.z,
                         world_pose.pose.orientation.w]
            
            # Extract camera pose
            camera_pos = np.array([camera_pose.pose.position.x,
                                 camera_pose.pose.position.y,
                                 camera_pose.pose.position.z])
            camera_quat = [camera_pose.pose.orientation.x,
                          camera_pose.pose.orientation.y,
                          camera_pose.pose.orientation.z,
                          camera_pose.pose.orientation.w]
            
            # Convert quaternions to rotation matrices
            world_rot = R.from_quat(world_quat)
            camera_rot = R.from_quat(camera_quat)
            correction_mat = np.array([[0, -1, 0],
                                  [1, 0, 0],
                                  [0, 0, 1]])
            camera_rot = camera_rot * R.from_matrix(correction_mat)

            
            # Transform position: P_camera = R_camera^T * (P_world - T_camera)
            pos_diff = world_pos - camera_pos
            camera_frame_pos = camera_rot.inv().apply(pos_diff)
            
            # Transform orientation: R_relative = R_camera^T * R_world
            relative_rot = camera_rot.inv() * world_rot
            relative_quat = relative_rot.as_quat()
            
            # Create transformed pose
            transformed_pose = PoseStamped()
            transformed_pose.header = world_pose.header
            transformed_pose.pose.position.x = camera_frame_pos[0]
            transformed_pose.pose.position.y = camera_frame_pos[1]
            transformed_pose.pose.position.z = camera_frame_pos[2]
            transformed_pose.pose.orientation.x = relative_quat[0]
            transformed_pose.pose.orientation.y = relative_quat[1]
            transformed_pose.pose.orientation.z = relative_quat[2]
            transformed_pose.pose.orientation.w = relative_quat[3]
            
            return transformed_pose
            
        except Exception as e:
            rospy.logerr(f"Error in pose transformation: {e}")
            return None
    
    def human_pose_callback(self, pose_msg):
        # Transform pose to camera frame if enabled
        if self.use_camera_transform:
            transformed_pose = self.transform_pose_to_camera_frame(pose_msg)
            if transformed_pose is None:
                rospy.logwarn("Skipping pose - transformation failed")
                return
        else:
            # Use original world frame pose without transformation
            transformed_pose = pose_msg
        
        if not hasattr(self, '_csv_setup_done'):
            # configure CSV path and lock on first callback
            if self.use_camera_transform:
                self.csv_path = os.path.join(os.environ.get('SCOT_BENCHMARK_DIR', os.path.expanduser('~/DOA/src/benchmarks')), 'occlusion_GT_camera_frame.csv')
            else:
                self.csv_path = os.path.join(os.environ.get('SCOT_BENCHMARK_DIR', os.path.expanduser('~/DOA/src/benchmarks')), 'occlusion_GT_world_frame.csv')
            self._csv_lock = threading.Lock()
            self._csv_setup_done = True

        # extract timestamp (robust to different rospy Time interfaces)
        try:
            stamp = pose_msg.header.stamp
            try:
                timestamp = stamp.secs + stamp.nsecs * 1e-9
            except Exception:
                timestamp = float(stamp.to_sec())
        except Exception:
            timestamp = float(rospy.Time.now().to_sec())

        # Use transformed pose data
        pos = transformed_pose.pose.position
        ori = transformed_pose.pose.orientation

        row = [timestamp, pos.x, pos.y, pos.z, ori.x, ori.y, ori.z, ori.w]
        frame_name = "camera frame" if self.use_camera_transform else "world frame"
        rospy.loginfo(f"Received GT pose at time {timestamp} ({frame_name}): position=({pos.x:.3f}, {pos.y:.3f}, {pos.z:.3f}), orientation=({ori.x:.3f}, {ori.y:.3f}, {ori.z:.3f}, {ori.w:.3f})")

        # ensure directory exists
        csv_dir = os.path.dirname(self.csv_path)
        if csv_dir and not os.path.exists(csv_dir):
            try:
                os.makedirs(csv_dir)
            except OSError:
                pass

        # write (append) to CSV, create header if file is new
        write_header = not os.path.exists(self.csv_path)
        with self._csv_lock:
            with open(self.csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(['time', 'pos_x', 'pos_y', 'pos_z', 'ori_x', 'ori_y', 'ori_z', 'ori_w'])
                writer.writerow(row)

        frame_name = "camera frame" if self.use_camera_transform else "world frame"
        rospy.loginfo_throttle(5, "GT pose (%s) logged to %s (t=%.6f)" % (frame_name, self.csv_path, timestamp))

    def main(self):
        if self.use_camera_transform:
            rospy.loginfo("GT Analyse node started - transforming poses to camera frame")
            rospy.loginfo("Waiting for both human pose and camera pose topics...")
        else:
            rospy.loginfo("GT Analyse node started - using world frame (no transformation)")
            rospy.loginfo("Waiting for human pose topic...")
        rospy.spin()

if __name__ == "__main__":
    gt_analyse = GTAnalyse()
    gt_analyse.main()