cmake_minimum_required(VERSION 3.0.2)
project(path_planning)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  costmap_2d
  dynamic_reconfigure
  geometry_msgs
  std_msgs
  message_generation
  pluginlib
  tf2_ros
  tf2_geometry_msgs
  nav_core
  base_local_planner
  angles
  umap
)

generate_dynamic_reconfigure_options(
  cfg/DWAController.cfg
)

add_message_files(
  FILES
  Obstacle.msg
  ObstacleArray.msg
)

generate_messages(
  DEPENDENCIES
  geometry_msgs
  std_msgs
)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES path_planning_layer dwa_controller
  CATKIN_DEPENDS roscpp costmap_2d dynamic_reconfigure geometry_msgs std_msgs message_runtime pluginlib tf2_ros tf2_geometry_msgs nav_core base_local_planner angles umap
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

add_library(path_planning_layer
  src/dynamic_obstacle_layer.cpp
)
add_dependencies(path_planning_layer ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(path_planning_layer ${catkin_LIBRARIES})

add_library(dwa_controller
  src/dwa.cpp
  src/dwa_controller.cpp
)
add_dependencies(dwa_controller ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(dwa_controller ${catkin_LIBRARIES})

install(TARGETS path_planning_layer dwa_controller
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION})
install(FILES costmap_plugins.xml dwa_controller_plugin.xml
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)

install(FILES costmap_plugins.xml
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
