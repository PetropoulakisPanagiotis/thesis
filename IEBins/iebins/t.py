from PIL import Image
import numpy as np
import os 
directory = "/usr/stud/petp/storage/user/petp/datasets/nyu_depth_v2/sync"
extension = ".png"

for root, dirs, files in os.walk(directory):
    for file in files:
        if file.endswith(extension):
            file_path = os.path.join(root, file)
            depth = Image.open(file_path)
            depth = np.asarray(depth)
            if np.max(depth)/1000 <= 5:
                print(np.max(depth)/1000)
        
