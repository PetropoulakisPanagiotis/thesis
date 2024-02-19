import torch

# Original tensor size
original_size = torch.Size([5, 128, 120, 160])

# Number of copies per element along the first dimension (n)
num_copies_per_element = torch.tensor([5, 6, 7, 8, 9])

# Desired size
desired_size = torch.sum(num_copies_per_element)

# Create the original tensor
original_tensor = torch.randn(original_size)

# Expand the original tensor along the first dimension
expanded_tensor = original_tensor.unsqueeze(1)

# Concatenate along the first dimension to form the larger tensor
larger_tensor = torch.cat([original_tensor[i, :, :, :].unsqueeze(0).repeat(times,1,1,1) for i, times in enumerate(num_copies_per_element)], dim=0)


import torch

# Example tensor
tensor = torch.tensor([[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7], [0, 8], [0, 9], [0, 10], [0, 11], [0, 12], [0, 13], [0, 14], [0, 15], [1, 0
], [1, 1], [1, 2], [1, 3], [1, 4], [1, 5], [1, 6], [1, 7], [1, 8], [1, 9], [1, 10], [1, 11], [1, 12], [1, 13], [1, 14], [1, 15], [1, 16], [1, 17], [1, 18], [1, 19], [1, 20], [1, 21], [1, 22], [1, 23], 
[1, 24], [1, 25], [1, 26], [1, 27], [1, 28], [1, 29]])

# Count occurrences of each batch index
batch_counts = torch.bincount(tensor[:, 0])

print(batch_counts)
