import os

def list_files_without_extension(directory):
    """List all files in a directory without their extensions."""
    filenames = os.listdir(directory)
    filenames_without_extension = [os.path.splitext(filename)[0] for filename in filenames]
    return filenames_without_extension

def diff_filenames(dir1, dir2):
    """Compare filenames without extensions between two directories."""
    files1 = set(list_files_without_extension(dir1))
    files2 = set(list_files_without_extension(dir2))
    
    only_in_dir1 = files1 - files2
    only_in_dir2 = files2 - files1
    
    return only_in_dir1, only_in_dir2

# Define the directories to compare
directory1 = '/home/petropoulakis/Desktop/thesis/code/datasets/scannet/data_converted/valid/network_predictions/scene0608_00/depth/'
directory2 = '/home/petropoulakis/Desktop/thesis/code/datasets/scannet/data_converted/valid/network_predictions/scene0608_00/canonical_unc/'

# Get the differences
only_in_dir1, only_in_dir2 = diff_filenames(directory1, directory2)

# Print the results
print(f"Files only in {directory1}:")
for filename in only_in_dir1:
    print(filename)

print(f"\nFiles only in {directory2}:")
for filename in only_in_dir2:
    print(filename)
