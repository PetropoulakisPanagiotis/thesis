import os

# Define the paths to the RGB and Depth folders
rgb_folder = "./scannet_0314_00/rgb"
depth_folder = "./scannet_0314_00/depth"

# List all files in the RGB and Depth folders
rgb_files = sorted(os.listdir(rgb_folder))
depth_files = sorted(os.listdir(depth_folder))

# Remove the file extensions to get the file names
rgb_names = [os.path.splitext(f)[0] for f in rgb_files]
depth_names = [os.path.splitext(f)[0] for f in depth_files]

# Ensure the lists are of the same length
if len(rgb_names) != len(depth_names):
    raise ValueError("The number of RGB and Depth files do not match!")

# Create the output text files
with open("associations.txt", "w") as output_file, open("rgb.txt", "w") as rgb_file, open("depth.txt", "w") as depth_file:
    for rgb_name, depth_name in zip(rgb_names, depth_names):
        # Write to output.txt
        output_file.write(f"{rgb_name} rgb/{rgb_name}.jpg {depth_name} depth/{depth_name}.png\n")
        
        # Write to rgb.txt
        rgb_file.write(f"{rgb_name} rgb/{rgb_name}.jpg\n")
        
        # Write to depth.txt
        depth_file.write(f"{depth_name} depth/{depth_name}.png\n")

print("Output files created successfully!")
