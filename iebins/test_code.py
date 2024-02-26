import torch

# Assuming your tensor is of size [13, 13, 120, 160]
tensor = torch.randn(13, 13, 120, 160)

# Assuming you have a list of 13 indexes
idx = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
print(idx.shape)
# Use advanced indexing to select slices along the second dimension
output = torch.gather(tensor, 1, idx)

print(output.size())  # Output size: [13, 120, 160]
