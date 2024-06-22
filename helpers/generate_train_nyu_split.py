import os

def generate_txt_file(root_folder, output_file):
    with open(output_file, 'w') as txt_file:
        # Get subfolder names in the root folder
        subfolders = [f.name for f in os.scandir(root_folder) if f.is_dir()]

        for subfolder in subfolders:
            subfolder_path = os.path.join(root_folder, subfolder)

            # List all files in the subfolder
            files = [f.name for f in os.scandir(subfolder_path) if f.is_file()]

            # Filter files with names 'rgb' and 'sync_depth' and extensions 'jpg' and 'png'
            rgb_files = [file for file in files if file.lower().endswith('.jpg') and 'rgb' in file.lower()]
            
            # Find corresponding sync_depth files and remove 'rgb' from the filename
            sync_depth_files = [f'sync_depth_{os.path.splitext(rgb_file)[0].replace("rgb_", "")}.png' for rgb_file in rgb_files]

            # Write entries to the text file
            for rgb_file, sync_depth_file in zip(rgb_files, sync_depth_files):
                line = f"{subfolder}/{rgb_file} {subfolder}/{sync_depth_file}\n"
                txt_file.write(line)
                

# Example usage:
root_folder = '/home/petropoulakis/Desktop/thesis/code/dataset/nyu_depth_v2/official_splits/train'
output_file = './output.txt'  # Replace with the desired output file name

generate_txt_file(root_folder, output_file)
