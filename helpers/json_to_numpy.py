import os
import json
import numpy as np

# Directory containing JSON files
folder_path = '/home/petropoulakis/Desktop/thesis/code/datasets/scannet/data_converted/valid/network_predictions/scene0608_00/canonical_unc/'

# Iterate over files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.json'):
        # Load JSON data
        with open(os.path.join(folder_path, filename), 'r') as json_file:
            json_data = json.load(json_file)
            json_data = json_data['canonical_unc']

        # Convert JSON data to NumPy array
        numpy_array = np.array(json_data, dtype=np.float32)

        # Save NumPy array with a different extension
        numpy_filename = os.path.splitext(filename)[0] + '.npy'
        np.save(os.path.join(folder_path, numpy_filename), numpy_array)

        # Load the saved NumPy array
        loaded_numpy_array = np.load(os.path.join(folder_path, numpy_filename))

        # Print if the loaded array matches the original array
        print(f"File: {filename}, Equal: {np.array_equal(numpy_array, loaded_numpy_array)}")
