#!/home/user/miniconda3/envs/oa/bin/python3

"""
Publish a normalized camera frame based on T265 pose data.
"""

import rospy
import numpy as np
import tf2_ros
import tf.transformations as tf_trans
from geometry_msgs.msg import TransformStamped
from scipy.spatial.transform import Rotation as R
import time

class NormalizedCameraFramePublisher:
    def __init__(self):
        rospy.init_node('normalized_camera_frame_publisher', anonymous=True)
        
        self.living_time = time.time()
        self.source_frame = rospy.get_param('~source_frame', 't265_pose_frame')
        self.target_frame = rospy.get_param('~target_frame', 'map')
        self.output_frame = rospy.get_param('~output_frame', 'real_baselink')
        self.k_init_frames = rospy.get_param('~k_init_frames', 10)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        self.rate = rospy.Rate(rospy.get_param('~publish_rate', 30.0))
        
        self.R_init = None
        self.R_init_inv = None
        self.R_avg = None
        self.rotation_history = []

        rospy.loginfo(f"Normalized Camera Frame Publisher initialized:")
        rospy.loginfo(f" - Source frame: {self.source_frame}")
        rospy.loginfo(f" - Target frame: {self.target_frame}")
        rospy.loginfo(f" - Output frame: {self.output_frame}")
    
    def get_init_R(self, transform_stamped):
        """通过前k帧的TransformStamped消息计算初始旋转矩阵R"""
        rotation = transform_stamped.transform.rotation
        quat = [rotation.x, rotation.y, rotation.z, rotation.w]
        rot_matrix = R.from_quat(quat).as_matrix()

        return rot_matrix

    def update_initial_rotation(self, rot_matrix):
        if self.k_init_frames <= 0 or self.R_init is not None:
            rospy.logdebug("Initial rotation already set or k_init_frames <= 0")
            return

        self.rotation_history.append(rot_matrix)

        if len(self.rotation_history) < self.k_init_frames:
            rospy.loginfo(
                f"Accumulating initial rotation frames: {len(self.rotation_history)}/"
                f"{self.k_init_frames}"
            )
            return

        self.R_avg = self.compute_karcher_mean(self.rotation_history[:self.k_init_frames])
        self.R_init = self.R_avg
        self.R_init_inv = self.R_init.T

        quat = R.from_matrix(self.R_avg).as_quat()
        rospy.loginfo(
            f"Computed initial rotation average over {self.k_init_frames} frames: "
            f"[{quat[0]:.4f}, {quat[1]:.4f}, {quat[2]:.4f}, {quat[3]:.4f}]"
        )

    def compute_karcher_mean(self, rotations):
        """使用矩阵log/exp求SO(3)上的Karcher均值"""
        if not rotations:
            return np.eye(3)

        rotation_objects = [R.from_matrix(rot) for rot in rotations]
        mean_rotation = rotation_objects[0]
        max_iterations = 20
        tolerance = 1e-6

        for _ in range(max_iterations):
            log_sum = np.zeros(3)
            for rot in rotation_objects:
                relative = mean_rotation.inv() * rot
                log_sum += relative.as_rotvec()

            log_sum /= len(rotations)

            if np.linalg.norm(log_sum) < tolerance:
                break

            mean_rotation = R.from_rotvec(log_sum) * mean_rotation

        return mean_rotation.as_matrix()
    
    def get_transform_matrix(self, transform_stamped):
        """将TransformStamped消息转换为4x4变换矩阵"""
        # 提取平移
        translation = transform_stamped.transform.translation
        t = np.array([translation.x, translation.y, translation.z])
        
        # 提取旋转并转换为旋转矩阵
        rotation = transform_stamped.transform.rotation
        quat = [rotation.x, rotation.y, rotation.z, rotation.w]
        rot_matrix = R.from_quat(quat).as_matrix()
        
        # 构建4x4齐次变换矩阵
        T = np.eye(4)
        T[:3, :3] = rot_matrix
        T[:3, 3] = t
        
        return T
    
    def create_normalized_transform(self, T_source_target):
        """创建标准化坐标系变换"""
        # 提取源坐标系原点在目标坐标系中的位置
        camera_origin = T_source_target[:3, 3]
        
        # 创建标准化坐标系到目标坐标系的变换
        # 平移部分为相机原点位置
        T_target_normalized = np.eye(4)
        T_target_normalized[:3, 3] = camera_origin

        if self.R_init_inv is not None:
            current_rot = T_source_target[:3, :3]
            # 左乘右乘?
            T_target_normalized[:3, :3] = self.R_init_inv @ current_rot
            # T_target_normalized[:3, :3] = current_rot @ self.R_avg
        
        return T_target_normalized
    
    def matrix_to_transform_stamped(self, T, parent_frame, child_frame, time):
        """将4x4变换矩阵转换为TransformStamped消息"""
        transform = TransformStamped()
        
        transform.header.stamp = time
        transform.header.frame_id = parent_frame
        transform.child_frame_id = child_frame
        
        # 设置平移
        transform.transform.translation.x = T[0, 3]
        transform.transform.translation.y = T[1, 3]
        transform.transform.translation.z = T[2, 3]
        
        # 设置旋转（从旋转矩阵转换为四元数）
        rot_matrix = T[:3, :3]
        quat = R.from_matrix(rot_matrix).as_quat()
        transform.transform.rotation.x = quat[0]
        transform.transform.rotation.y = quat[1]
        transform.transform.rotation.z = quat[2]
        transform.transform.rotation.w = quat[3]
        
        return transform
    
    def run(self):
        while not rospy.is_shutdown():
            try:
                # 获取最新的变换
                transform_stamped = self.tf_buffer.lookup_transform(
                    self.target_frame,
                    self.source_frame,
                    rospy.Time(0),  # 获取最新可用变换
                    rospy.Duration(1.0)  # 超时时间
                )
                
                T_source_target = self.get_transform_matrix(transform_stamped)
                self.update_initial_rotation(T_source_target[:3, :3])
                
                T_target_normalized = self.create_normalized_transform(T_source_target)
                
                normalized_transform = self.matrix_to_transform_stamped(
                    T_target_normalized,
                    self.target_frame,
                    self.output_frame,
                    transform_stamped.header.stamp
                )
                
                self.tf_broadcaster.sendTransform(normalized_transform)
                
                rospy.loginfo_throttle(5.0, "Successfully publishing normalized camera frame")
                
            except (tf2_ros.LookupException, 
                    tf2_ros.ConnectivityException, 
                    tf2_ros.ExtrapolationException) as e:
                rospy.logwarn_throttle(5.0, f"TF lookup failed: {str(e)}")
            
            self.rate.sleep()

if __name__ == '__main__':
    try:
        node = NormalizedCameraFramePublisher()
        node.run()
    except rospy.ROSInterruptException:
        pass