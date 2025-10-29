#!/home/patrick/miniconda3/envs/ros-noetic/bin/python3

"""
Feature extraction node for dynamic obstacle tracking.
Subscribes to dynamic obstacles and RGB images, extracts CLIP features and SAM segmentation results.
"""
import time
import rospy
import cv2
import PIL.Image
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Float32MultiArray
import message_filters
from message_filters import Subscriber, TimeSynchronizer, ApproximateTimeSynchronizer
from umap.msg import BoundingBoxes2D, BoundingBoxes3D, SegmentedImage, MultiImageFeatures, Obstacles
from scipy.optimize import linear_sum_assignment

import sys
# Verify work path
# print("Python executable:", sys.executable)
# print("Python path:", sys.path)

import torch
import clip
import sys

import umap

sys.path.append("/home/patrick/FastSAM/")

from fastsam import FastSAM, FastSAMPrompt
from src import segment, funcs


class Extractor:
    def __init__(self):

        self.bridge = CvBridge()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        rospy.loginfo(f"Using device: {self.device}")
        self.model = FastSAM('/home/patrick/FastSAM/weights/FastSAM.pt')
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device="cuda")
        self.frontGround_only = False # 提取ROI的时候只提取前景
        self.BOOSTclip = True

        self.run_SAM = 0 # cancle to boost!
        self.SAM_conf = 0.5
        self.SAM_iou = 0.8
        self.SAM_imgsz = 128 # multiply of 32

        self.image_sub = rospy.Subscriber('/d400/color/image_raw', Image, self.image_callback)

        # self.bbox_sub = message_filters.Subscriber('/tracker/dynamic_2D_boxes', BoundingBoxes2D)
        # self.box3D_sub = message_filters.Subscriber('/tracker/dynamic_3D_boxes', BoundingBoxes3D)
        # self.sync = ApproximateTimeSynchronizer([self.bbox_sub, self.box3D_sub], queue_size=30, slop=1)
        # self.sync.registerCallback(self.callback)
        self.dynamic_obstacles_sub = rospy.Subscriber('tracker/dynamic_obstacles', Obstacles, self.callbackDync3D)
        # self.dynamic_3dbox_sub = rospy.Subscriber('/tracker/dynamic_3D_boxes', BoundingBoxes3D, self.callbackDync3D)

        # Publisher for segmented results
        self.seg_pub = rospy.Publisher("/umap/segmented_boxes", BoundingBoxes2D, queue_size=10)
        self.vis_pub = rospy.Publisher("/model_output/segmented_image", Image, queue_size=10)

        self.feature_pub = rospy.Publisher('/model_output/clip_features', MultiImageFeatures, queue_size=10)
        self.box3D_repub = rospy.Publisher('/model_output/3D_bounding_boxes', BoundingBoxes3D, queue_size=10)
        self.obstacles_pub = rospy.Publisher('/model_output/obstacles', Obstacles, queue_size=10)

        # 添加用于发布拼接ROI图像的发布器
        self.roi_vis_pub = rospy.Publisher("/model_output/visualization/roi_image", Image, queue_size=10)

        self.rgb_map = [self.id_to_color(i) for i in range(100)]
        self.curr_img = None
        self.curr_depth = None

        rospy.loginfo("Bbox segmenter node is ready...")


    def image_callback(self, image_msg):
        rospy.loginfo("Received image message.")
        self.curr_img = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
        


    def id_to_color(self, index):
        """Generate unique color for visualization"""
        import hashlib
        hash_hex = hashlib.md5(str(index).encode()).hexdigest()[:6]
        return tuple(int(hash_hex[i:i+2], 16) for i in (0, 2, 4))
    


    def segment_Image(self, image, input_size = 320):
        """Segment the image based on the bounding box"""
        original_h, original_w = image.shape[:2]
        # Resize image for processing
        scale = min(input_size / original_w, input_size / original_h)
        new_w = int(original_w * scale)
        new_h = int(original_h * scale)
        resized_image = cv2.resize(image, (new_w, new_h))
        image_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)

        # Run FastSAM
        results = self.model(image_rgb, device='cuda', verbose=False, imgsz=input_size, conf=0.8, iou=0.95)


        # Process results
        if results[0].masks is not None:
            masks_origsize = []
            for mask in results[0].masks.data:
                mask_np = mask.cpu().numpy().astype(np.uint8)
                mask_original = cv2.resize(mask_np, (original_w, original_h))
                masks_origsize.append(mask_original)

            if masks_origsize:
                masks_origsize = np.stack(masks_origsize, axis=0)
                segments = segment.extract_segment_from_rgb(
                    IMAGE_PATH='', IMAGE=image, masks=masks_origsize)
                return segments

        return []


    def get_segment_from_bbox(self, image, x, y, w, h):
            """Extract image region within bbox"""
            # Create empty mask
            height, width = image.shape[:2]
            mask = np.zeros((height, width, 3), dtype=np.uint8)

            # Extract ROI
            roi = image[y:y+h, x:x+w]
            if roi.size == 0:
                return mask

            mask[y:y+h, x:x+w] = roi

            return mask # 480, 640


    def get_clip_features(self, image, boost = True):
        """Extract CLIP features from the image
        @input_type: cv_image
        """
        # Load CLIP model
        device = self.device
        clip_model, preprocess = self.clip_model, self.clip_preprocess
        
        # Resize image for faster processing
        # resize_shape=(224, 224)
        if boost:
            resize_shape = (int(image.shape[0] / 2), int(image.shape[1] / 2))
            image = cv2.resize(image, resize_shape)

        # Preprocess image
        image = PIL.Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        image = preprocess(image).unsqueeze(0).to(device)

        # Extract features
        with torch.no_grad():
            feature = clip_model.encode_image(image)
            feature /= feature.norm(dim=-1, keepdim=True)

        return feature.cpu().numpy()

    def getROI(self, cv_img, mask, noBackground = True):
        non_zero_points = np.argwhere(mask > 0)

        if len(non_zero_points) > 0:  # 确保mask中有非零点

            min_y, min_x = non_zero_points.min(axis=0)
            max_y, max_x = non_zero_points.max(axis=0)

            # 提取最小外包矩形区域
            roi = cv_img[min_y:max_y+1, min_x:max_x+1]

            # 提取相应的mask区域
            mask_roi = mask[min_y:max_y+1, min_x:max_x+1]
            # 在ROI区域内应用mask
            if noBackground:
                # 仅提取mask的地方，框中无mask的地方不提取
                result_roi = np.zeros_like(roi)
                result_roi[mask_roi > 0] = roi[mask_roi > 0]
            else:
                result_roi =roi
            # 扩展回原始尺寸
            # result = np.zeros_like(rgb_image)
            # result[min_y:max_y+1, min_x:max_x+1] = result_roi
            print(f"ROI提取成功: 大小为 {result_roi.shape}")
            return result_roi

        else:
            print("mask中没有非零点")
            result_roi = np.zeros((0, 0, 3), dtype=cv_img.dtype)
            return result_roi


    def concatenate_rois(self, rois, box_ids):
        """将多个ROI图像拼接成一张大图，并标注ID"""
        if not rois:
            return None

        # 计算合适的布局（尽量接近正方形）
        n = len(rois)
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))
        
        # 找出所有ROI中的最大尺寸，以便统一缩放
        max_h = max([roi.shape[0] for roi in rois]) if rois else 0
        max_w = max([roi.shape[1] for roi in rois]) if rois else 0
        
        # 如果ROI非常大，进行等比例缩放
        max_size = 200  # 每个ROI的最大尺寸
        scale = min(max_size / max_h, max_size / max_w) if max(max_h, max_w) > max_size else 1
        
        target_h = int(max_h * scale)
        target_w = int(max_w * scale)
        # 创建拼接画布
        grid_h = rows * target_h
        grid_w = cols * target_w
        grid_img = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
        # 将每个ROI放置到画布上
        for i, (roi, box_id) in enumerate(zip(rois, box_ids)):
            # 计算在网格中的位置
            row = i // cols
            col = i % cols

            # 缩放ROI到目标大小
            resized_roi = cv2.resize(roi, (target_w, target_h))

            # 计算放置位置
            y_start = row * target_h
            x_start = col * target_w

            # 放置ROI
            grid_img[y_start:y_start+target_h, x_start:x_start+target_w] = resized_roi

            # 添加ID标签
            cv2.putText(grid_img, f"ID:{box_id}", (x_start+5, y_start+20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return grid_img
    
    def callbackDync3D(self, dync_obstacles_msg):
        rospy.loginfo("\n\n Received dync_obstacles.")
        # cv_image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")

        cv_image = self.curr_img
        height, width = cv_image.shape[:2]

        # Prepare output message
        box_ids = dync_obstacles_msg.box_ids

        if len(box_ids) == 0:
            rospy.logwarn("No bounding boxes received.")
            return
        else:
            rospy.loginfo(f"3D dync boxes ids: {box_ids}")

        # decode BoundingBoxes3D
        boxes_x = dync_obstacles_msg.box_x
        boxes_y = dync_obstacles_msg.box_y
        boxes_width = dync_obstacles_msg.box_width
        boxes_height = dync_obstacles_msg.box_height
        box_masks = dync_obstacles_msg.mask
        
        vis_image = cv_image.copy() # Create visualization image
        features = [] # featrues to publish
        clip_timecost = 0
        sam_timecost = 0
        
        # 收集所有ROI用于拼接可视化
        all_rois = []
        
        for box_idx in range(len(box_ids)):
            # (x, y) is top-left corner of the bounding box
            mask = self.bridge.imgmsg_to_cv2(box_masks[box_idx], "passthrough")
            rospy.loginfo(f"size of mask: {mask.shape}")
            roi = self.getROI(cv_img=cv_image, mask=mask, noBackground=self.frontGround_only)

            if roi.size == 0:
                rospy.loginfo(f"ROI is empty at index {box_idx}")
                continue
            if roi.shape[0] < 10 or roi.shape[1] < 10:
                rospy.loginfo(f"ROI is too small at index {box_idx}")


                continue

            rospy.loginfo(f"size of roi: {roi.shape}")

            # 保存ROI用于后续拼接
            all_rois.append(roi.copy())

            # 1. Extract roi CLIP features
            clip_s = time.time()
            roi_features = self.get_clip_features(roi, boost=self.BOOSTclip)
            features.append(roi_features)
            clip_e = time.time()
            clip_timecost += clip_e - clip_s
            rospy.loginfo(f"Clip time: {clip_e - clip_s}s")

            # 2. Run FastSAM on ROI
            if self.run_SAM:
                timeSAMs = time.time()
                everything_results = self.model(roi, device=self.device, retina_masks=True, conf=self.SAM_conf, iou=self.SAM_iou, imgsz=self.SAM_imgsz)
                prompt_process = FastSAMPrompt(roi, everything_results, device=self.device)
                # Get segmentation masks
                ann = prompt_process.everything_prompt()
                timeSAMe = time.time()
                sam_timecost += timeSAMe - timeSAMs
                rospy.loginfo(f"FastSAM time: {timeSAMe - timeSAMs}s")
                if len(ann) == 0:
                    continue

                # Process each mask
                for mask_id, mask in enumerate(ann):
                    # Convert mask to bounding box
                    mask = mask.cpu().numpy()
                    y_indices, x_indices = np.where(mask)
                    if len(x_indices) == 0 or len(y_indices) == 0:
                        continue

                    # Calculate box coordinates relative to ROI
                    x_min, x_max = np.min(x_indices), np.max(x_indices)
                    y_min, y_max = np.min(y_indices), np.max(y_indices)

                    # Convert to image coordinates
                    # abs_x = x + x_min
                    # abs_y = y + y_min

                    abs_x = boxes_x[box_idx]
                    abs_y = boxes_y[box_idx]
                    abs_w = boxes_width[box_idx]
                    abs_h = boxes_height[box_idx]
                    # abs_w = x_max - x_min
                    # abs_h = y_max - y_min

                    # Draw on visualization image
                    color = self.rgb_map[mask_id % len(self.rgb_map)]
                    cv2.rectangle(vis_image, (abs_x, abs_y), 
                                (abs_x + abs_w, abs_y + abs_h), 
                                color, 2)
                    cv2.putText(vis_image, f"ID:{box_ids[box_idx]}-{mask_id}", 
                                (abs_x, abs_y-5), cv2.FONT_HERSHEY_SIMPLEX, 
                                0.5, color, 2)

                    # ===== 可视化ROI区域 =====
                    # 获取mask的非零区域的外接矩形
                    non_zero_points = np.argwhere(mask > 0)
                    if len(non_zero_points) > 0:
                        min_y, min_x = non_zero_points.min(axis=0)
                        max_y, max_x = non_zero_points.max(axis=0)
                        min_x += abs_x
                        min_y += abs_y
                        max_x += abs_x
                        max_y += abs_y
                        # 画矩形框
                        color = self.rgb_map[box_idx % len(self.rgb_map)]
                        cv2.rectangle(vis_image, (min_x, min_y), (max_x, max_y), color, 2)
                        # 标识ID
                        cv2.putText(vis_image, f"ID:{box_ids[box_idx]}", 
                                    (min_x, min_y-5), cv2.FONT_HERSHEY_SIMPLEX, 
                                    0.7, color, 2)
                    # ===== 可视化ROI区域结束 =====
            # 拼接并发布ROI可视化图像
            if all_rois:
                concat_roi_img = self.concatenate_rois(all_rois, box_ids[:len(all_rois)])
                if concat_roi_img is not None:
                    roi_vis_msg = self.bridge.cv2_to_imgmsg(concat_roi_img, "bgr8")
                    roi_vis_msg.header = dync_obstacles_msg.header
                    self.roi_vis_pub.publish(roi_vis_msg)
                    rospy.loginfo(f"Published ROI visualization with {len(all_rois)} regions")

            # 1. Publish visualization
            vis_msg = self.bridge.cv2_to_imgmsg(vis_image, "bgr8")
            # vis_msg.header = image_msg.header
            self.vis_pub.publish(vis_msg)
            rospy.loginfo("Published RGB segmented image.")

            # 2. Publish Obstacles msg
            obstacles_msg = Obstacles()
            obstacles_msg.header = dync_obstacles_msg.header
            obstacles_msg.box_ids = box_ids

            obstacles_msg.centroid_x = dync_obstacles_msg.centroid_x
            obstacles_msg.centroid_y = dync_obstacles_msg.centroid_y
            obstacles_msg.centroid_z = dync_obstacles_msg.centroid_z
            obstacles_msg.thickness = dync_obstacles_msg.thickness
            obstacles_msg.height = dync_obstacles_msg.height
            obstacles_msg.width = dync_obstacles_msg.width

            obstacles_msg.box_x = dync_obstacles_msg.box_x
            obstacles_msg.box_y = dync_obstacles_msg.box_y
            obstacles_msg.box_width = dync_obstacles_msg.box_width
            obstacles_msg.box_height = dync_obstacles_msg.box_height

            obstacles_msg.features = []
            for feature in features:
                feature_msg = Float32MultiArray()
                feature_msg.data = feature.flatten().tolist()
                obstacles_msg.features.append(feature_msg)

            self.obstacles_pub.publish(obstacles_msg)
            rospy.loginfo("Published segmented boxes.")

if __name__ == '__main__':
    try:
        rospy.init_node("Extractor", anonymous=True)
        segmenter = Extractor()
        rospy.loginfo("Extractor node is running...")
        rospy.spin()
    except rospy.ROSInterruptException:
        pass