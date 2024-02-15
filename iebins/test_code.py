import torch

def transform_last_dim(tensor):
    # Define the permutation of values
    permutation = torch.tensor([1, 0, 3, 2])
    
    # Use gather to perform the transformation
    transformed_tensor = torch.gather(tensor, 1, permutation.unsqueeze(0).expand(tensor.size(0), -1))
    
    return transformed_tensor

# Example usage
b = 2

# Create a random tensor
tensor = torch.arange(b * 4).reshape(b, 4)

# Transform the last dimension
transformed_tensor = transform_last_dim(tensor)
print("Original Tensor:")
print(tensor)
print("\nTransformed Tensor:")
print(transformed_tensor)
