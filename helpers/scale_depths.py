import cv2
import os

def process_depth_image(input_path, output_path):
    # Read the depth image
    depth_image = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    
    if depth_image is None:
        raise FileNotFoundError(f"Image not found at {input_path}")

    processed_image = depth_image * 5  

    # Save the processed image
    cv2.imwrite(output_path, processed_image)

def process_directory(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith(".png"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            process_depth_image(input_path, output_path)
            print(f"Processed and saved: {output_path}")

# Example usage
input_directory = '/usr/stud/petp/storage/user/petp/datasets/predictions/scene0314_00/global/depth'
output_directory = '/usr/stud/petp/storage/user/petp/datasets/predictions/scene0314_00/global/depth_scaled'
process_directory(input_directory, output_directory)

input_directory = '/usr/stud/petp/storage/user/petp/datasets/predictions/scene0314_00/per_class/depth'
output_directory = '/usr/stud/petp/storage/user/petp/datasets/predictions/scene0314_00/per_class/depth_scaled'
process_directory(input_directory, output_directory)

input_directory = '/usr/stud/petp/storage/user/petp/datasets/predictions/scene0314_00/per_instance/depth'
output_directory = '/usr/stud/petp/storage/user/petp/datasets/predictions/scene0314_00/per_instance/depth_scaled'
process_directory(input_directory, output_directory)
