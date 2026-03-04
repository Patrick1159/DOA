std_msgs/Header header

# 节点标识
string node_name

# 本次处理的障碍物数量
int32 obstacle_count

# 总处理耗时 (s)
float32 total_time

# 模型推理耗时 (s)：SAM 节点填 sam_time，CLIP 节点填 clip_time，aggregator 保持 0
float32 model_time

# 预处理耗时：ROI 提取、mask 转换等 (s)
float32 preprocess_time

# 后处理耗时：mask 变换、overlay 合成、结果发布等 (s)
float32 postprocess_time

# 每个障碍物的平均模型推理耗时 (s)；障碍物数为 0 时填 0
float32 avg_model_time_per_obstacle

# aggregator 专用：等待同步的耗时 (s)（sam/clip 节点填 0）
float32 sync_wait_time

# 消息端到端延迟：当前时间 - message.header.stamp (s)
float32 e2e_latency

# ---- Rematch node profiling (ms) ----
int32 num_detections
int32 num_trackers
int32 num_matched
int32 num_unmatched_det

float32 t_parse_ms
float32 t_cost_matrix_ms
float32 t_association_ms
float32 t_update_ms
float32 t_publish_vis_ms
float32 t_publish_det_ms
float32 t_total_ms
