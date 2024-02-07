import torch

def slice_feature_map_batch(feature_map_batch, slicing_indices_batch):
    """
    Slice the feature map batch using the given slicing indices batch.

    Parameters:
        - feature_map_batch (torch.Tensor): The feature map batch tensor with shape [5, h, w].
        - slicing_indices_batch (torch.Tensor): Tensor containing slicing indices with shape [5, 4].

    Returns:
        - torch.Tensor: Sliced feature map batch tensor with shape [5, c, h, w].
    """
    # Expand the dimensions of the slicing indices to match the shape of the feature map batch
    slicing_indices_batch_expanded = slicing_indices_batch.unsqueeze(2).unsqueeze(3)

    # Perform gather operation along dim=1 and dim=2
    sliced_feature_map_batch = torch.gather(feature_map_batch, 2, slicing_indices_batch_expanded[:, :, :, :2])  # along dim=2
    sliced_feature_map_batch = torch.gather(sliced_feature_map_batch, 3, slicing_indices_batch_expanded[:, :, :, 2:])  # along dim=3

    return sliced_feature_map_batch

# Example usage:
feature_map_batch = torch.randn(5, 3, 64, 64)  # Example feature map batch with 5 channels, height=64, width=64
slicing_indices_batch = torch.tensor([
    [10, 20, 30, 40],
    [5, 10, 25, 35],
    [15, 25, 35, 45],
    [20, 30, 40, 50],
    [25, 35, 45, 55]
])  # Example slicing indices batch for 5 feature maps

sliced_feature_map_batch = slice_feature_map_batch(feature_map_batch, slicing_indices_batch)

print("Original Feature Map Batch Shape:", feature_map_batch.shape)
print("Sliced Feature Map Batch Shape:", sliced_feature_map_batch.shape)
