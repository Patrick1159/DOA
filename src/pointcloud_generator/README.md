# 点云处理工具 (Point Cloud Processing Tool)

## 简介

该工具用于处理深度相机输出的深度图与彩色图像，转化为可视化的点云表征。

## 功能

- 输入为：相机的位置(geometry_msgs::PoseStamped)，相机深度图消息(sensor_msgs::Image)，相机彩色图消息(sensor_msgs::Image)
- 输出为：处理后世界系下点云

## 安装

### 依赖

该工具基于以下库：
- [ROS](https://www.ros.org/)
- [PCL (Point Cloud Library)](https://pointclouds.org/)
- [Eigen](http://eigen.tuxfamily.org/dox/)

在安装之前，请确保您的系统已安装这些依赖项。

### 编译
1. 创建工作空间（若有工作空间，则直接在工作空间`src`内即可）
    ```bash
   mkdir -p red_ws/src
   cd red_ws/src
   catkin_init_workspace
   ```
2. 克隆项目：
   ```bash
   git clone https://gitee.com/hx2020hx/pointcloud_generator.git
   ```
3. 编译
    ```bash
   cd ..
   catkin_make
   ```

## 使用方法

1. 启动，当需要链接到项目时，请修改`launch`文件中的消息
    ```bash
   source devel/setup.bash
   roslaunch pointcloud_generator pointcloud_processor_node.launch
   ```
2. 请按照自己的消息类型修改`launch`文件中的消息，点云消息输出为`/RED/pointcloud_process/pointcloud`。
    ```xml
    <launch>
        <node name="camera_node" pkg="pointcloud_generator" type="processor" output="screen">
            <rosparam file="$(find pointcloud_generator)/param/default_param.yaml" command="load"/>

            <!-- <remap from="/RED/pointcloud_process/pointcloud" to="/your_pc_name"/> -->
            <remap from="/RED/pointcloud_process/camera_pose" to="/camera/current_pose"/>
            <remap from="/RED/pointcloud_process/odom" to="/uav0/mavros/local_position/odom"/>
            <remap from="/RED/pointcloud_process/color_image" to="/camera/color/image_raw"/>
            <remap from="/RED/pointcloud_process/depth_image" to="/camera/depth/image_raw"/>
        </node>
    </launch>
   ```

## 参数调整
见`param/default_param`

```
    camera:
        view_field: 4.0       
        self_filter: 0.25    
        grid_size: 0.02     
        eud_range: 0.05         
        eud_min_points_num: 40  


        fx: 525.0              
        fy: 525.0             
        cx: 319.5              
        cy: 239.5              
        camera_factor: 1000    
```

### 1. 相机内参
其中`fx`,`fy`,`cx`,`cy`表示4个相机内参，分别为焦距与偏置。`camera_factor`为深度转换因子，多为1000。

### 2. 滤波参数
- `view_field`:直通滤波深度，点云保留的最远距离。
- `self_filter`:自身滤除球半径。
- `grid_size`: 体素滤波所选体素大小。
- `eud_range`，`eud_min_points_num`: 欧式聚类除噪点，在0.05的球内需要有40个点才不是噪点。

## 贡献

如果您有改进建议或发现了问题，欢迎提交Issue或Pull Request。
