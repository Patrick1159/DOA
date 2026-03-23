# DOA - Dynamic Obstacle Avoidance

深度相机+视觉特征融合的动态障碍物检测与跟踪系统，用于移动机器人导航。

## 系统概述

DOA 提供完整的"感知→检测→跟踪"管线，将深度图与RGB图像融合，提取并跟踪动态障碍物，输出结构化的跟踪器状态供规划器使用。

**核心模块：**
- `umap`：从深度/RGB提取障碍物2D/3D边界框，使用特征匹配与卡尔曼滤波跟踪
- `pointcloud_generator`：深度图转点云，用于障碍物位置精确定位
- `path_planning`：costmap层与DWA插件（可选，对接move_base时使用）

**关键话题：**
- 输入：`/d435/image_raw` (RGB), `/d435/depth/image_raw` (深度)
- 输出：`/rematch/tracker_details` (TrackerDetailArray) - 障碍物ID、位置、速度、尺寸

---

## 依赖安装

```bash
# ROS基础包
sudo apt install ros-noetic-cv-bridge ros-noetic-image-transport \
  ros-noetic-tf2-ros ros-noetic-pcl-ros ros-noetic-vision-msgs

# Python依赖
pip3 install torch torchvision  # FastSAM/CLIP特征提取
pip3 install opencv-python numpy pillow
```

---

## 快速开始（仿真环境）

### 1. 启动Gazebo仿真与机器人

```bash
# Terminal 1: 启动TurtleBot3 + 相机
roslaunch my_tb3_description my_tb3_gazebo.launch
```

### 2. 启动DOA感知系统

```bash
# Terminal 2: 完整DOA管线（检测+跟踪+特征提取）
roslaunch umap simulation.launch
```

**话题映射（已在launch配置）：**
- RGB输入：`/d435/image_raw`
- 深度输入：`/d435/depth/image_raw`
- 障碍物输出：`/rematch/tracker_details`

### 3. 可视化

RViz会自动启动，显示：
- 检测到的2D/3D边界框
- 跟踪器轨迹与ID
- 点云（如启用）

或手动打开：
```bash
rviz -d src/DOA/src/umap/rviz/simulation.rviz
```

---

## 系统架构

```
深度图 ──┐
         ├──> [umapProcess]        提取2D/3D BBox
RGB图 ───┘         ↓
                [coarse_match]     粗匹配+动态判别
                    ↓
              [FastSAM/CLIP]       特征提取（可选）
                    ↓
              [detailed_match]     细匹配+跟踪器（可选）
                    ↓
          /rematch/tracker_details  输出到规划器
```

**运行模式：**
- **仿真模式**：`simulation.launch`（使用粗匹配，无细匹配，轻量高效）
- **真机模式**：`main.launch`（完整管线，需实际硬件）

---

## 配置文件

### 核心参数 (`config/simulation.yaml`)

```yaml
# UMAP 检测参数
umap_node:
  threshold_point: 3.0      # 直方图阈值
  hist_size: 150            # 深度直方图分辨率
  depth_factor: 1.0         # 深度缩放因子
  min_dist: 400             # 最近检测距离(mm)
  max_dist: 5000            # 最远检测距离(mm)
  camera_frame: "camera_color_optical_frame"

# Tracker 参数
tracker_node:
  DYNAMIC_VELOCITY_THRESHOLD: 1.0    # 动态判定速度阈值(m/s)
  MAX_TRACKER_AGE: 1.0               # 跟踪器超时时间(s)
  IOU_THRESHOLD: 0.3                 # IoU匹配阈值
  HISTORY_SIZE: 10                   # 历史轨迹长度

# 点云生成参数
pointcloud_generate:
  camera:
    view_field: 4.0         # 最远视野(m)
    self_filter: 0.0025     # 自身过滤半径(m)
    grid_size: 0.05         # 体素滤波格子大小
```

### UMAP 近距离聚类增强参数（`config/experiment.yaml` / `config/real_config.yaml`）

当目标距离 D435 很近时，同一个人可能在 U-map 中被切成左右两个峰（典型是双腿）。
现在 `umapProcess` 在 `extract_bb()` 结束后增加了一层“横向近邻合并”，会在深度层相近时，把左右间距较小的 `bounding_box_U` 合并成一个 obstacle。

```yaml
umap_node:
  threshold_point: 0.06
  threshold_line: 2.5
  min_length_line: 40
  hist_size: 250

  enable_horizontal_merge: true
  merge_horizontal_gap: 100
  merge_max_row_gap: 8
  merge_min_vertical_overlap_ratio: 0.3
  enable_adaptive_horizontal_merge: true
  adaptive_merge_min_gap: 20
  adaptive_merge_max_gap: 100
  adaptive_merge_near_ratio: 0.25
  adaptive_merge_far_ratio: 0.85
```

**新增参数说明：**

- `enable_horizontal_merge`
  - 是否启用横向近邻合并。
  - 建议真机场景开启。

- `merge_horizontal_gap`
  - 两个 U-map 框在 x 方向允许的最大间距（像素列）。
  - 值越大，越容易把“双腿”合成一个障碍物。
  - 过大时也更容易把两个并排的人误合并。

- `merge_max_row_gap`
  - 两个框在 U-map 深度 bin 方向允许的最大中心差。
  - 用于容忍少量上下错层。

- `merge_min_vertical_overlap_ratio`
  - 两个框在深度 bin 方向的最小重叠比例。
  - 值越大，合并越保守。

- `enable_adaptive_horizontal_merge`
  - 是否启用“近距离 gap 更大、远距离 gap 更小”的自适应横向合并。
  - 开启后，`merge_horizontal_gap` 会退化为兜底/兼容参数，实际阈值由下面 4 个参数控制。

- `adaptive_merge_min_gap` / `adaptive_merge_max_gap`
  - 分别对应远距离和近距离时的横向 gap 阈值。
  - 近距离目标更容易被拆成多个峰，因此通常 `max_gap` 明显大于 `min_gap`。

- `adaptive_merge_near_ratio` / `adaptive_merge_far_ratio`
  - 用 U-map 的深度 bin 归一化位置来定义“近”和“远”的过渡区间。
  - 小于 `near_ratio` 按“近距离”处理，大于 `far_ratio` 按“远距离”处理，中间线性插值。

**推荐调参顺序：**

1. 先调 `merge_horizontal_gap`
   - 近距离双腿仍分裂：增大到 `40 ~ 100`
   - 误合并明显：减小到 `20 ~ 40`
2. 若已启用自适应 merge，优先调 `adaptive_merge_max_gap`
  - 近距离双腿仍分裂：增大到 `80 ~ 140`
  - 近距离误合并明显：减小到 `50 ~ 80`
3. 再调 `adaptive_merge_min_gap`
  - 远距离误合并明显：减小到 `10 ~ 25`
  - 远距离仍容易碎裂：增大到 `25 ~ 40`
4. 再调 `merge_min_vertical_overlap_ratio`
   - 误合并多：提高到 `0.4 ~ 0.5`
   - 想更容易合并：降低到 `0.2 ~ 0.3`
5. 最后微调 `merge_max_row_gap`
   - 一般 `6 ~ 12` 足够

**和原始聚类参数的配合建议：**

- 想让双腿更容易先连成块：适当降低 `threshold_point` / `threshold_line`
- 想减少小噪声误合并：适当提高 `min_length_line`
- 若深度分箱太细导致近距离目标碎裂，可适当减小 `hist_size`

---

## 话题接口

### 订阅的话题

| 话题 | 类型 | 说明 |
|-----|------|------|
| `/camera/aligned_depth_to_color/image_raw` | `sensor_msgs/Image` | 对齐的深度图 |
| `/camera/color/image_raw` | `sensor_msgs/Image` | RGB彩色图 |
| `/camera/current_pose` | `geometry_msgs/PoseStamped` | 相机位姿（由pointcloud_generator发布） |

### 发布的话题

| 话题 | 类型 | 说明 |
|-----|------|------|
| `/rematch/tracker_details` | `umap/TrackerDetailArray` | 跟踪器详细信息（ID、位姿、速度、尺寸） |
| `/tracker/dynamic_obstacles` | `umap/Obstacles` | 动态障碍物列表（粗匹配输出） |
| `/umap/3D_bounding_boxes` | `umap/BoundingBoxes3D` | 检测的3D边界框 |
| `/umap/2D_bounding_boxes` | `umap/BoundingBoxes2D` | 检测的2D边界框 |
| `/model_output/obstacles` | `umap/Obstacles` | 特征匹配后的障碍物 |

### 可视化话题

| 话题 | 类型 | 说明 |
|-----|------|------|
| `/rematch/visualization/tracked_obstacles` | `visualization_msgs/MarkerArray` | 跟踪器3D标记 |
| `/rematch/obstacle_paths` | `nav_msgs/Path` | 跟踪器历史轨迹 |
| `/umap/visualization/bbox_rgb_visualization` | `sensor_msgs/Image` | 带框的RGB可视化 |

---

## 消息格式

### TrackerDetailArray
```
std_msgs/Header header
TrackerDetail[] trackers
  int32 tracker_id           # 唯一跟踪ID
  geometry_msgs/Point position
  geometry_msgs/Vector3 velocity
  geometry_msgs/Vector3 box_dimensions  # x=width, y=height, z=depth
  float64 last_update_time
  float64 age                # 自首次检测以来的时间(s)
```

---

## 与MPC导航集成

DOA感知系统可以直接替代简单的障碍物生成器，为MPC提供真实的视觉感知输入。

### 完整系统启动（推荐）

使用一键 launch 启动完整测试环境：

```bash
# 启动 Gazebo + DOA感知 + MPC导航 + 动态障碍物
roslaunch umap simulation_dync.launch
```

**包含的所有组件：**
- ✅ Gazebo仿真环境 + TurtleBot3
- ✅ 动态障碍物场景（走廊或随机）
- ✅ DOA完整感知管线（检测+跟踪）
- ✅ 适配器节点（自动转换格式）
- ✅ MPC规划器
- ✅ RViz可视化

**可选参数：**
```bash
# 使用随机移动场景
roslaunch umap simulation_dync.launch dynamic_scene:=0

# 使用走廊场景（默认）
roslaunch umap simulation_dync.launch dynamic_scene:=1

# 启动rematch节点
roslaunch umap detailed_match.launch
```

### 手动分步启动（调试用）

如果需要单独调试各模块：

```bash
# Terminal 1: Gazebo + 机器人
roslaunch my_tb3_description my_tb3_gazebo.launch

# Terminal 2: 动态障碍物
roslaunch my_tb3_description dync_obs.launch scene:=1

# Terminal 3: DOA感知
roslaunch umap simulation.launch

# Terminal 4: 适配器（DOA输出 → MPC输入）
rosrun mpc_nav doa_to_obstacles.py

# Terminal 5: MPC规划
roslaunch mpc_nav mpc_node.launch
```

### 适配器节点说明

`doa_to_obstacles.py` 负责将 DOA 的 `TrackerDetailArray` 转换为 MPC 的 `ObstacleArray`：

**订阅话题：**
- `/rematch/tracker_details` (umap/TrackerDetailArray)

**发布话题：**
- `/doa_obstacles` (mpc_nav/ObstacleArray) ← DOA观测结果，MPC读取此话题

**话题命名约定：**
- `/obstacles`：Ground truth（仿真脚本发布的真实障碍物状态，用于评估）
- `/doa_obstacles`：Observations（DOA视觉系统的检测结果，用于规划）

**转换逻辑：**
- ID映射：`tracker_id` → `obstacle.id`
- 位置/速度：直接复制
- 尺寸→半径：`radius = 0.5 * sqrt(width² + height²) + padding`
- 过滤：年龄 > `max_age` 的目标被丢弃
- 限速：超过 `velocity_limit` 的速度被裁剪

**可配置参数：**
```bash
rosrun mpc_nav doa_to_obstacles.py \
  _target_frame:=odom \
  _max_age:=0.5 \
  _velocity_limit:=5.0 \
  _radius_padding:=0.1
```

---

## Launch文件详解

### simulation_dync.launch（完整系统测试 - 推荐）

**用途**：一键启动 DOA感知 + MPC导航 + 动态环境的完整测试

**启动内容：**
- Gazebo仿真 + TurtleBot3
- 动态障碍物场景（可选 scene 0/1）
- DOA完整感知管线（umap + coarse_match + 特征提取 + 点云）
- `doa_to_obstacles.py` 适配器（自动转换格式）
- MPC规划器
- RViz可视化

**使用：**
```bash
roslaunch umap simulation_dync.launch
# 或指定场景
roslaunch umap simulation_dync.launch dynamic_scene:=0
```

**适用场景**：完整系统集成测试、演示、基准测试

---

### simulation.launch（DOA感知单独测试）

**用途**：仅测试 DOA 感知系统，不启动导航和仿真

启动内容：
- `umap_extract`：深度/RGB检测节点
- `coarse_match`：粗匹配跟踪器（轻量，适合仿真）
- `para_samnode/clipnode/aggregate`：FastSAM+CLIP特征提取
- `pointcloud_generator`：点云处理与相机位姿发布
- RViz可视化

**使用：**
```bash
# 需要先启动 Gazebo（单独终端）
roslaunch my_tb3_description my_tb3_gazebo.launch
# 然后启动 DOA
roslaunch umap simulation.launch
```

**适用场景**：调试感知算法、参数调优、性能测试

**不启用**（仿真环境下注释掉）：
- `detailed_match`：细匹配（需要高质量特征，真机使用）

---

### main.launch（真机完整管线）

完整管线，包含细匹配与rematch节点，用于实际硬件部署。

---

## 坐标系说明

**关键TF变换：**
```
odom (规划全局坐标)
  └─ base_link (机器人本体)
      └─ camera_link
          └─ d435_color_optical_frame (DOA输出坐标)
```

**重要：**
- DOA默认输出在相机光学坐标系 `d435_color_optical_frame`
- 适配器会检查并要求在 `odom` 坐标系（需要TF正确配置）
- `pointcloud_generator` 中的 `camera_pose.py` 负责发布相机→odom的变换

---

## 调试与问题排查

### 1. 检测不到障碍物
```bash
# 查看深度图是否有数据
rostopic hz /d435/depth/image_raw

# 查看检测结果
rostopic echo /umap/3D_bounding_boxes

# 调整检测阈值
rosparam set /umap_node/threshold_point 2.0
```

### 2. 跟踪器ID频繁跳变
```yaml
# 降低匹配阈值（config/simulation.yaml）
tracker_node:
  IOU_THRESHOLD: 0.25
  DYNAMIC_DIST_THRESHOLD: 0.4
```

### 3. 性能问题
- 关闭细匹配（`detailed_match`已注释）
- 降低特征提取分辨率（FastSAM配置）
- 减少点云密度：`grid_size: 0.1`

### 4. TF错误
```bash
# 查看当前TF树
rosrun tf view_frames
evince frames.pdf

# 检查相机位姿发布
rostopic echo /camera/current_pose
```

---

## Rematch Profiler（性能监控）

系统内置 `rematch_profiler.py`，用于订阅 `/rematch_statistics` 并实时打印
`detailed_match` 各阶段耗时（解析、代价矩阵、关联、更新、可视化、发布等），
支持滚动窗口统计、超时告警与可选 CSV 导出。

### 订阅话题

| 话题 | 类型 | 说明 |
|------|------|------|
| `/rematch_statistics` | `umap/RematchStatistics` | rematch 节点各阶段耗时统计 |

### 快速使用

```bash
# 直接运行（默认窗口=200帧）
rosrun umap rematch_profiler.py

# 输出 CSV
rosrun umap rematch_profiler.py _csv_path:=/tmp/rematch_stats.csv

# 启用实时绘图（需 matplotlib）
rosrun umap rematch_profiler.py _plot:=true
```

### 可选参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `~window` | 200 | 滚动统计窗口大小 |
| `~warn_total_ms` | 100.0 | 总耗时告警阈值（ms） |
| `~csv_path` | "" | CSV 输出路径（空则不输出） |
| `~plot` | false | 是否实时绘图（需 matplotlib） |

### 输出字段（RematchStatistics）

Profiler 主要使用以下字段（单位 ms）：
`t_parse_ms`、`t_cost_matrix_ms`、`t_association_ms`、`t_update_ms`、
`t_publish_vis_ms`、`t_publish_det_ms`、`t_total_ms`，以及
`num_detections`、`num_trackers`、`num_matched`、`num_unmatched_det`。

---

## 性能基准（仿真环境）

| 模块 | 频率 | CPU占用 |
|-----|------|---------|
| umap_extract | 15-20 Hz | ~25% |
| coarse_match | 15-20 Hz | ~15% |
| FastSAM/CLIP | 5-10 Hz | ~40% (GPU) |
| pointcloud_generator | 10-15 Hz | ~10% |

**总计**：约90% CPU，30 FPS（含Gazebo仿真）

---

## 文件结构

```
src/DOA/
├── src/
│   ├── umap/                    # 核心检测+跟踪
│   │   ├── src/
│   │   │   ├── clean_umap.cpp            # 主检测节点
│   │   │   ├── clean_coarse_match.cpp    # 粗匹配跟踪
│   │   │   └── clean_detailed_match.cpp  # 细匹配跟踪
│   │   ├── scripts/
│   │   │   ├── para_samnode.py           # FastSAM特征
│   │   │   ├── para_clipnode.py          # CLIP特征
│   │   │   └── para_aggregate.py         # 特征聚合
│   │   ├── launch/
│   │   │   ├── simulation.launch         # 仿真配置
│   │   │   └── main.launch               # 真机配置
│   │   ├── config/
│   │   │   ├── simulation.yaml           # 仿真参数
│   │   │   └── real_config.yaml          # 真机参数
│   │   └── msg/
│   │       ├── TrackerDetail.msg
│   │       └── TrackerDetailArray.msg
│   ├── pointcloud_generator/    # 点云处理
│   │   └── scripts/
│   │       └── camera_pose.py            # 相机位姿TF
│   └── path_planning/           # (可选) move_base集成
│       ├── src/
│       │   ├── dynamic_obstacle_layer.cpp  # costmap插件
│       │   └── dwa_controller.cpp          # DWA插件
│       └── scripts/
│           └── test_dynamic_obstacles.py   # 测试脚本
└── README.md
```

---

## 引用

如果在研究中使用DOA系统，请引用：

```bibtex
@misc{doa2026,
  title={DOA: Dynamic Obstacle Avoidance for Mobile Robots},
  author={Your Name},
  year={2026},
  howpublished={\url{https://github.com/Patrick1159/simulation_framework}}
}
```

---

## 许可证

TODO: 添加许可证信息
