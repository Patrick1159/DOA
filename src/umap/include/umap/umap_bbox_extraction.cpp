#include "umapProcess.h"

/// @brief A part of generate 2Dbox from histogram
UVbox umapProcess::merge_two_UVbox(UVbox father, UVbox son){
    // merge the bounding box
    int top =       (father.bb.tl().y < son.bb.tl().y)?father.bb.tl().y:son.bb.tl().y;
    int left =      (father.bb.tl().x < son.bb.tl().x)?father.bb.tl().x:son.bb.tl().x;
    int bottom =    (father.bb.br().y > son.bb.br().y)?father.bb.br().y:son.bb.br().y;
    int right =     (father.bb.br().x > son.bb.br().x)?father.bb.br().x:son.bb.br().x;
    father.bb = cv::Rect(cv::Point2f(left, top), cv::Point2f(right, bottom));
    return father;
}

cv::Rect umapProcess::mergeRects(const cv::Rect& first, const cv::Rect& second) const {
    const int left = std::min(first.x, second.x);
    const int top = std::min(first.y, second.y);
    const int right = std::max(first.x + first.width, second.x + second.width);
    const int bottom = std::max(first.y + first.height, second.y + second.height);
    return cv::Rect(left, top, right - left, bottom - top);
}

int umapProcess::computeAdaptiveMergeGap(const cv::Rect& first, const cv::Rect& second) const {
    if (!params_.enable_adaptive_horizontal_merge || params_.hist_size <= 0) {
        return params_.merge_horizontal_gap;
    }

    const float first_center_row = first.y + first.height * 0.5f;
    const float second_center_row = second.y + second.height * 0.5f;
    const float average_center_row = 0.5f * (first_center_row + second_center_row);
    const float normalized_depth_ratio = std::max(0.0f, std::min(1.0f, average_center_row / static_cast<float>(params_.hist_size)));

    const float near_ratio = std::max(0.0f, std::min(1.0f, params_.adaptive_merge_near_ratio));
    const float far_ratio = std::max(near_ratio, std::min(1.0f, params_.adaptive_merge_far_ratio));
    const int min_gap = std::min(params_.adaptive_merge_min_gap, params_.adaptive_merge_max_gap);
    const int max_gap = std::max(params_.adaptive_merge_min_gap, params_.adaptive_merge_max_gap);

    if (normalized_depth_ratio <= near_ratio) {
        return max_gap;
    }

    if (normalized_depth_ratio >= far_ratio) {
        return min_gap;
    }

    const float transition = (normalized_depth_ratio - near_ratio) / std::max(1e-6f, far_ratio - near_ratio);
    const float interpolated_gap = max_gap - transition * static_cast<float>(max_gap - min_gap);
    return static_cast<int>(std::round(interpolated_gap));
}

bool umapProcess::shouldMergeUVBoxes(const cv::Rect& first, const cv::Rect& second) const {
    const int first_right = first.x + first.width;
    const int second_right = second.x + second.width;

    int horizontal_gap = 0;
    if (first_right < second.x) {
        horizontal_gap = second.x - first_right;
    } else if (second_right < first.x) {
        horizontal_gap = first.x - second_right;
    }

    const int effective_horizontal_gap = computeAdaptiveMergeGap(first, second);
    if (horizontal_gap > effective_horizontal_gap) {
        return false;
    }

    const int overlap_top = std::max(first.y, second.y);
    const int overlap_bottom = std::min(first.y + first.height, second.y + second.height);
    const int overlap_height = std::max(0, overlap_bottom - overlap_top);
    const int min_height = std::max(1, std::min(first.height, second.height));
    const float vertical_overlap_ratio = static_cast<float>(overlap_height) / static_cast<float>(min_height);

    const float first_center_y = first.y + first.height * 0.5f;
    const float second_center_y = second.y + second.height * 0.5f;
    const float center_row_gap = std::abs(first_center_y - second_center_y);

    return vertical_overlap_ratio >= params_.merge_min_vertical_overlap_ratio ||
           center_row_gap <= params_.merge_max_row_gap;
}

void umapProcess::mergeNearbyUVBoxes() {
    if (bounding_box_U.size() < 2) {
        return;
    }

    bool merged_in_pass = true;
    while (merged_in_pass) {
        merged_in_pass = false;
        std::vector<cv::Rect> merged_boxes;
        std::vector<bool> consumed(bounding_box_U.size(), false);

        for (size_t i = 0; i < bounding_box_U.size(); ++i) {
            if (consumed[i]) {
                continue;
            }

            cv::Rect current = bounding_box_U[i];
            consumed[i] = true;

            bool expanded = true;
            while (expanded) {
                expanded = false;
                for (size_t j = i + 1; j < bounding_box_U.size(); ++j) {
                    if (consumed[j]) {
                        continue;
                    }

                    if (shouldMergeUVBoxes(current, bounding_box_U[j])) {
                        current = mergeRects(current, bounding_box_U[j]);
                        consumed[j] = true;
                        merged_in_pass = true;
                        expanded = true;
                    }
                }
            }

            merged_boxes.push_back(current);
        }

        bounding_box_U = merged_boxes;
    }
}

/// @brief Extract bounding boxes from HISTOGRAM matrix, store in class variable bounding_box_U
/// @param hist_matrix
void umapProcess::extract_bb(cv::Mat& hist_matrix){
        // initialize a mask
        std::vector<std::vector<int>> mask(hist_matrix.rows, std::vector<int>(hist_matrix.cols, 0));

        // initialize parameters for extraction
        float u_min = params_.threshold_point;  // Adjust this threshold to work with unnormalized values
        float sum_line = 0;
        float max_line = 0;
        int length_line = 0;
        int seg_id = 0;

        std::vector<UVbox> UVboxes;

        for (int row = 0; row < hist_matrix.rows; row++){

            for (int col = 0; col < hist_matrix.cols; col++){
                if (hist_matrix.at<float>(row, col) >= u_min){
                    // ROS_INFO_STREAM("Valid point found!");
                    length_line++; // line is horizontal
                    sum_line += hist_matrix.at<float>(row,col);
                    max_line = std::max(max_line, hist_matrix.at<float>(row,col));
                }
                if (hist_matrix.at<float>(row, col) < u_min || col == hist_matrix.cols - 1){
                    // Meet the first non-interest point OR meet the end of row
                    // Thus should end sum 'length_line'

                    // is end of the row
                    col = (col == hist_matrix.cols - 1)? col + 1:col;

                    // is good line candidate (length and sum)
                    // ROS_INFO_STREAM("length_line: " << length_line << "\n");
                    // ROS_INFO_STREAM("max_line: " << max_line << "\n");
                    if(length_line > params_.min_length_line && sum_line > params_.threshold_line * max_line)
                    {
                        seg_id++;
                        UVboxes.push_back(UVbox(seg_id, row, col-length_line, col-1));
                        // ROS_INFO_STREAM("UVboxes pushback: " << seg_id << "\n");
                        // overwrite the mask with segementation id
                        for(int c = col-length_line; c < col; c++)
                        {
                            mask[row][c] = seg_id;
                        }

                        // when current row is not first row, we need to merge neighbour segementation
                        if(row != 0){
                            // merge all parents
                            for(int c = col - length_line; c < col - 1; c++){
                                if(mask[row - 1][c] != 0) // 像素头顶上的位置有意义
                                {
                                    // initially, toppest parent id is just seg id.
                                    // mask stores seg_ids
                                    if(UVboxes[mask[row - 1][c] - 1].toppest_parent_id < UVboxes.back().toppest_parent_id) // seg_id从1开始计
                                    {
                                        UVboxes.back().toppest_parent_id = UVboxes[mask[row - 1][c] - 1].toppest_parent_id;
                                    }
                                    else
                                    {
                                        int temp = UVboxes[mask[row - 1][c] - 1].toppest_parent_id;
                                        for(size_t b = 0; b < UVboxes.size(); b++)
                                        {
                                            UVboxes[b].toppest_parent_id = (UVboxes[b].toppest_parent_id == temp)?UVboxes.back().toppest_parent_id:UVboxes[b].toppest_parent_id;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    sum_line = 0;
                    max_line = 0;
                    length_line = 0;
                }
            }
        }
        // group lines into boxes
        this->bounding_box_U.clear();
        // merge boxes with same toppest parent
        // ROS_INFO_STREAM("UVBOXES size: " << UVboxes.size() << "\n");
        for(size_t b = 0; b < UVboxes.size(); b++)
        {
            // ROS_INFO_STREAM(UVboxes[b].id);

            if(UVboxes[b].id == UVboxes[b].toppest_parent_id)
            {
                // ROS_INFO_STREAM("\nUVboxes[b].id == UVboxes[b].toppest_parent_id");
                for(size_t s = b + 1; s < UVboxes.size(); s++)
                {
                    if(UVboxes[s].toppest_parent_id == UVboxes[b].id)
                    {
                        UVboxes[b] = merge_two_UVbox(UVboxes[b], UVboxes[s]);
                    }
                }
                // check box's size
                if(UVboxes[b].bb.area() >= 0)
                {
                    this->bounding_box_U.push_back(UVboxes[b].bb);
                    // printf("bbox_U [b] %d",UVboxes[b].bb.tl().y);
                }
            }
        }

        if (params_.enable_horizontal_merge) {
            mergeNearbyUVBoxes();
        }
    }


/// @brief  Extract 2D bounding boxes from depth/RGB image and store in bounding_box_D
/// @param depth_val
/// @param stamp
void umapProcess::extract_2DboxRGB(const ros::Time& stamp, const cv::Mat& depth_val) {
    bounding_box_HW.clear(); // bounding_box_HW 仅保存当前帧的Bbox

    std::lock_guard<std::mutex> rgb_lock(rgb_mutex_);
    std::lock_guard<std::mutex> depth_lock(depth_mutex_);

    if (current_rgb_.empty()) {
        ROS_WARN("No RGB image available for visualization");
        return;
    }

    cv::Mat visualization = current_rgb_.clone(); // For generating rgb_bbox_visualization
    // 对每个histogram中的UVbox进行处理
    for(const auto& bb_U : bounding_box_U) {
        // Find corresponding 3D vertical range
        // 找到histogram中UVbox对应的3D垂直范围
        float bin_width = (params_.max_dist - params_.min_dist) / (params_.hist_size*1.0);
        float depth_in_near = params_.min_dist + bb_U.y * bin_width;
        float depth_in_far = depth_in_near + bb_U.height * bin_width * 1.3;

        // Vertical range detection in RGB image
        int y_up = current_rgb_.rows, y_down = 0;
        int x_start = std::max(0, bb_U.x);
        int x_end = std::min(current_rgb_.cols, bb_U.x + bb_U.width);

        if (x_start < depth_val.cols && x_end > 0) {
            for(int i = x_start; i < x_end; i++) {
                for(int j = 0; j < current_rgb_.rows && j < depth_val.rows; j++) {
                    if (i < depth_val.cols) {
                        float depth = depth_val.at<float>(j, i);
                        if (std::isfinite(depth) && depth >= depth_in_near && depth <= depth_in_far) {
                            y_up = std::min(y_up, j);
                            y_down = std::max(y_down, j);
                        }
                    }
                }
            }
        }

        if (y_up < y_down) {
            // Draw rectangle on RGB image
            cv::Point pt1(x_start, y_up); // top-left
            cv::Point pt2(x_end, y_down); // bottom-right

            cv::Scalar color = cv::Scalar(0, 255, 0);  // Green for static

            cv::rectangle(visualization, pt1, pt2, color, 2); // visualization size: (480, 640)
            // 将bboxes pushback to bounding_box_HW或再创建一个成员变量存储框框 2D bounding boxes
            bounding_box_HW.push_back(cv::Rect(pt1, pt2));
        }
    } // end for loop

    // Publish visualization after iterating through all bounding boxes
    sensor_msgs::ImagePtr rgb_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", visualization).toImageMsg();
    rgb_msg->header.stamp = stamp;
    rgb_bbox_pub_.publish(rgb_msg);
    // ROS_INFO("[umap module] RGB image with bounding boxes published");

    // publish2DBoxes(bounding_box_HW, stamp, current_rgb_.cols, current_rgb_.rows);
    umap::BoundingBoxes2D HW_box_msg;
    HW_box_msg.header.stamp = stamp;
    HW_box_msg.header.frame_id = params_.camera_frame;
    HW_box_msg.image_width = current_rgb_.cols;
    HW_box_msg.image_height = current_rgb_.rows;
    for (size_t i = 0; i < bounding_box_HW.size(); ++i) {
        HW_box_msg.box_ids.push_back(i);  // 每个框的唯一ID
        HW_box_msg.x.push_back(bounding_box_HW[i].x);
        HW_box_msg.y.push_back(bounding_box_HW[i].y);
        HW_box_msg.height.push_back(bounding_box_HW[i].height);
        HW_box_msg.width.push_back(bounding_box_HW[i].width);
        // msg.confidence.push_back(1.0);  // default 1.0
    }

    bbox_2D_pub_.publish(HW_box_msg);
}

void umapProcess::extract_3Dbox(const ros::Time& stamp, const cv::Mat& depth_val){
    boxes3D.clear();
    std::vector<cv::Mat> box_masks;

    // current_frame_++;

    // For boost
    // cv::Mat depth_resize;
    // cv::resize(depth_val, depth_resize, cv::Size(), 1.0/row_downsample, 1.0);

    for (const auto& bb_U : bounding_box_U) {
        // Find corresponding 3D vertical range
        float bin_width = (params_.max_dist - params_.min_dist) / (params_.hist_size*1.0);
        float depth_in_near = params_.min_dist + bb_U.y * bin_width;
        float depth_in_far = depth_in_near + bb_U.height * bin_width * 1.3;

        // Vertical range detection in RGB image
        int y_up = current_rgb_.rows, y_down = 0;
        int x_start = std::max(0, bb_U.x);
        int x_end = std::min(current_rgb_.cols, bb_U.x + bb_U.width);

        if (x_start < depth_val.cols && x_end > 0) { // if valid
            for(int i = x_start; i < x_end; i++) {
                for(int j = 0; j < current_rgb_.rows && j < depth_val.rows; j++) {
                    if (i < depth_val.cols) {
                        float depth = depth_val.at<float>(j, i);
                        if (std::isfinite(depth) && depth >= depth_in_near && depth <= depth_in_far) {
                            y_up = std::min(y_up, j);
                            y_down = std::max(y_down, j);
                        }
                    }
                }
            }
        }

        if (y_up < y_down) {
            // 1. 生成3D box
            box3D curr_box;

            depth_in_far /= params_.depth_factor; // convert mm to meter
            depth_in_near /= params_.depth_factor;

            // The pixel UV box (bb_U.x..x+width on u-axis, y_up..y_down on
            // v-axis) is the projection of the obstacle's FRONT face — the
            // back face is occluded and not visible in the depth image.
            // Reproject UV using `depth_in_near` (front-face depth) to get
            // the 3D location of the front-face center, then translate
            // half a thickness ALONG THE VIEW RAY (not along the optical
            // z-axis) to obtain the geometric centroid.
            //
            // Why along the view ray: U-map bins are slices of the optical
            // z-axis, so depth_in_far - depth_in_near is the obstacle's
            // extent measured along the optical z-axis. To physically
            // represent the obstacle's centroid we must move along the
            // line of sight from the camera origin through the front-face
            // center — that is the direction in which the back face is
            // hidden behind the front face. For on-axis pixels the view
            // ray equals +z and the two are equivalent; for off-axis
            // pixels they diverge and using +z places the centroid in the
            // wrong direction.
            //
            // We also rescale the depth-axis thickness into a view-ray
            // thickness via the secant of the off-axis angle:
            //   thickness_view = thickness_z * |front_pt| / depth_in_near
            // so the box thickness in 3D matches what would be measured
            // along the same direction along which we just translated.
            const float u_center = bb_U.x + bb_U.width / 2.0f;
            const float v_center = (y_up + y_down) / 2.0f;

            const float x_front = (u_center - params_.px) * depth_in_near / params_.fx;
            const float y_front = (v_center - params_.py) * depth_in_near / params_.fy;
            const float front_norm = std::sqrt(x_front * x_front +
                                               y_front * y_front +
                                               depth_in_near * depth_in_near);

            const float thickness_z = depth_in_far - depth_in_near;
            const float thickness_view = thickness_z * front_norm / depth_in_near;
            const float scale = (front_norm + thickness_view / 2.0f) / front_norm;

            curr_box.x = x_front * scale;
            curr_box.y = y_front * scale;
            curr_box.z = depth_in_near * scale;

            // Box dimensions: width/height are reprojected at the front
            // face depth (consistent with the UV reprojection above);
            // thickness is the view-ray-aligned extent.
            curr_box.x_width = std::max(0.1f, bb_U.width * depth_in_near / params_.fx); // 宽度
            curr_box.y_width = std::max(0.1f, (y_down - y_up) * depth_in_near / params_.fy); // 高度
            curr_box.z_width = std::max(0.1f, thickness_view); // 厚度（沿视线）
            boxes3D.push_back(curr_box);

            // 2.Create mask data
            cv::Mat mask = extractRGBPointsInBox(depth_val, current_rgb_, curr_box);
            box_masks.push_back(mask);
        }
    } // end for loop

    // Publish 3D bounding boxes message
    umap::BoundingBoxes3D bbox3D_msg;

    bbox3D_msg.header.stamp = stamp; // Use the same timestamp as the input images
    bbox3D_msg.header.frame_id = params_.camera_frame;
    bbox3D_msg.image_width = current_rgb_.cols;
    bbox3D_msg.image_height = current_rgb_.rows;


    for(size_t i = 0; i < boxes3D.size(); i++) {
        bbox3D_msg.box_ids.push_back(i);

        // Convert center coordinates and dimensions
        bbox3D_msg.x.push_back(boxes3D[i].x);
        bbox3D_msg.y.push_back(boxes3D[i].y);
        bbox3D_msg.z.push_back(boxes3D[i].z);

        bbox3D_msg.width.push_back(boxes3D[i].x_width);
        bbox3D_msg.height.push_back(boxes3D[i].y_width);
        bbox3D_msg.thickness.push_back(boxes3D[i].z_width);
        sensor_msgs::ImagePtr mask_msg = cv_bridge::CvImage(std_msgs::Header(), "mono8", box_masks[i]).toImageMsg();
        bbox3D_msg.mask.push_back(*mask_msg);

    }

    bbox_3D_pub_.publish(bbox3D_msg);
    // ROS_INFO("[umap module]Published 3D bounding boxes message with %zu boxes", boxes3D.size());

    // Publish 3D bounding boxes for Visualization
    visualization_msgs::MarkerArray marker_array;

    for (size_t i = 0; i<boxes3D.size(); i++) {
        visualization_msgs::Marker marker;
        marker.header.frame_id = params_.camera_frame;
        marker.header.stamp = ros::Time::now();
        marker.ns = "3D_bounding_boxes";
        marker.id = i;
        marker.type = visualization_msgs::Marker::CUBE;
        marker.action = visualization_msgs::Marker::ADD;

        // Set the position and size of the box
        marker.pose.position.x = boxes3D[i].z;
        marker.pose.position.y = -boxes3D[i].x;
        marker.pose.position.z = -boxes3D[i].y;

        // Orientation (identity quaternion)
        marker.pose.orientation.w = 1.0;
        marker.pose.orientation.x = 0.0;
        marker.pose.orientation.y = 0.0;
        marker.pose.orientation.z = 0.0;

        marker.scale.x = boxes3D[i].z_width;
        marker.scale.y = boxes3D[i].x_width;
        marker.scale.z = boxes3D[i].y_width;

        // Set color and transparency
        marker.color.r = 0.0f;
        marker.color.g = 1.0f;
        marker.color.b = 0.0f;
        marker.color.a = 0.5f; // semi-transparent

        marker.lifetime = ros::Duration(0.2);  // Exist lifetime

        // Add the marker to the array
        marker_array.markers.push_back(marker);
    }
    marker_pub_.publish(marker_array);
}
