cmake_minimum_required(VERSION 3.0.2)
project(umap)

# Add C++11 support
add_compile_options(-std=c++11)

## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
    roscpp
    std_msgs
    nav_msgs
    geometry_msgs
    message_generation
    cv_bridge
    image_transport
    sensor_msgs
    tf2
    tf2_ros
    tf2_sensor_msgs
    pcl_conversions
    pcl_ros

)

find_package(PCL REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)

# Add message files
add_message_files(
  FILES
  BoundingBoxes2D.msg
  BoundingBoxes3D.msg
  Segments_list.msg
  SegmentedImage.msg
  MultiImageFeatures.msg
  Obstacles.msg
  TrackerDetail.msg
  TrackerDetailArray.msg
  RematchStatistics.msg
)

# Generate messages
generate_messages(
  DEPENDENCIES
  std_msgs
  sensor_msgs
  geometry_msgs # for history_pos msg
  nav_msgs
)

# Declare catkin package
catkin_package(
   INCLUDE_DIRS include
   LIBRARIES umap
   CATKIN_DEPENDS roscpp message_runtime cv_bridge image_transport sensor_msgs std_msgs nav_msgs pcl_conversions pcl_ros tf2 tf2_ros tf2_sensor_msgs
   DEPENDS EIGEN3 OpenCV
)

###########
## Build ##
###########

## Specify additional locations of header files
include_directories(
    include
    ${catkin_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
)

## Declare a C++ library
add_library(${PROJECT_NAME}
    include/umap/KalmanFilter.cpp
    include/umap/ObstacleTracker.cpp
    include/umap/umapProcess.cpp
    include/umap/umap_bbox_extraction.cpp
    include/umap/umap_visualization.cpp
    include/umap/umapVisualisation.cpp
)

target_link_libraries(${PROJECT_NAME}
    ${catkin_LIBRARIES}
    ${OpenCV_LIBRARIES}
)

## Make sure the library is built before the executables
add_dependencies(${PROJECT_NAME} ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

# add_executable(tracker_demo 
#     src/tracker_demo.cpp
# )

# add_executable(getParams 
#     src/params.cpp)
# target_link_libraries(getParams 
#     ${PROJECT_NAME}
#     ${catkin_LIBRARIES})

# coarse match
add_executable(coarse_match
    src/clean_coarse_match.cpp
    src/coarse_match_visualization.cpp
)

add_executable(umap_extract
    src/clean_umap.cpp
)

# add_executable(static_transform_node
#     src/static_tf.cpp
# )

# detailed match
add_executable(detailed_match
    src/clean_detailed_match.cpp
    src/detailed_match_visualization.cpp
)

# target_link_libraries(static_transform_node
#   ${catkin_LIBRARIES}
# )

# final frame convertion
add_executable(bbox_frame_converter src/convert_bbox_frame.cpp)
target_link_libraries(bbox_frame_converter ${catkin_LIBRARIES})
# Link against our library
target_link_libraries(umap_extract
    ${PROJECT_NAME}
    ${catkin_LIBRARIES}
    ${OpenCV_LIBRARIES}
)

target_link_libraries(coarse_match
    ${PROJECT_NAME}
    ${catkin_LIBRARIES}
    ${OpenCV_LIBRARIES}
)

target_link_libraries(detailed_match
    ${PROJECT_NAME}
    ${catkin_LIBRARIES}
    ${OpenCV_LIBRARIES}
)

# Install targets
install(TARGETS ${PROJECT_NAME} umap_extract coarse_match # static_transform_node rematch
    ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
    LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
    RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# Install headers
install(DIRECTORY include/${PROJECT_NAME}/
    DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
    FILES_MATCHING PATTERN "*.h"
)

