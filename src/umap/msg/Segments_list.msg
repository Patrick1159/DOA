Header header

# identify the number of frame
uint32 frame_id

uint16 image_count

# segment size
uint16 width
uint16 height

# segments中的分割结果image列表，元素为ndarray(H, W, 3)类型
sensor_msgs/Image[] images 

# segment id
string[] image_ids
