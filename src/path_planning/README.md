## Brief
使用movebase框架，把我的动态障碍物的速度信息和位置放在obstacle layer中，生成最终的costmap，再接入DWA和A*进行规划

## TODO
[X] 用movebase框架，读取走廊静态地图，让小车能不撞墙(integrate DWA & A*)
[ ] 将现在的generate costmap内容改为只生成obstacle layer。sub: pointcloud + dynamic obstacle(Obstacle类型), pub: costmap2d。实现移动人，local costmap 里红块跟着动