cmake_minimum_required(VERSION 3.0.2)
project(pointcloud_generator)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  cv_bridge
  image_transport
  tf
  tf_conversions
  nav_msgs
  sensor_msgs
  geometry_msgs
  message_filters
  umap  # 添加umap依赖
)
find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(
  include ${catkin_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS} ${OPENCV_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIRS}
)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES pointcloud_generator
  CATKIN_DEPENDS rospy std_msgs sensor_msgs nav_msgs cv_bridge image_transport tf tf_conversions message_filters umap
  DEPENDS PCL OpenCV Eigen3
)

add_library(pointcloud_generator src/pointcloud_process.cpp)
target_link_libraries(pointcloud_generator ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBS})

add_executable(processor src/processor.cpp src/pointcloud_process.cpp)
target_link_libraries(processor ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBS})
