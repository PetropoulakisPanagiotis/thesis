import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
 

# Load the PNG depth image
img = cv2.imread('/home/petropoulakis/Desktop/thesis/code/datasets/scannet/data_converted/valid/depth/scene0655_01/00781.png', cv2.IMREAD_UNCHANGED).astype(np.float32)
img = img /  1000
 
depth_normalized = cv2.normalize(img, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
depth_colormap = cv2.applyColorMap(np.uint8(255 * depth_normalized), cv2.COLORMAP_INFERNO)
cv2.imshow('Depth Colormap', depth_colormap)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("./depth.jpg", depth_colormap)
