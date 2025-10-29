# Message for segmented image results
Header header

# Original image dimensions
uint32 image_width
uint32 image_height

# Segmentation results
int32[] segment_ids      # IDs for each segment
uint32[] x              # x coordinates of segments
uint32[] y              # y coordinates of segments
uint32[] width          # width of segments
uint32[] height         # height of segments
float32[] confidence    # confidence scores