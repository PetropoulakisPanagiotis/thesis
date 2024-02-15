import torch

def mask_feature_map(feature_map, box_coordinates):
    batch_size = box_coordinates.size(0)
    height, width = feature_map.size(-2), feature_map.size(-1)

    # Expand box coordinates to match the shape of the feature map
    ymin, xmin, ymax, xmax = box_coordinates.split(1, dim=1)
    ymin = torch.clamp(ymin.to(torch.int64), 0, height)
    xmin = torch.clamp(xmin.to(torch.int64), 0, width)
    ymax = torch.clamp(ymax.to(torch.int64), 0, height)
    xmax = torch.clamp(xmax.to(torch.int64), 0, width)

    row_indices = torch.arange(height, device=feature_map.device).unsqueeze(0)

    col_indices = torch.arange(width, device=feature_map.device).unsqueeze(0)

    row_mask = (row_indices >= ymin) & (row_indices < ymax)
    col_mask = (col_indices >= xmin) & (col_indices < xmax)

    masks_row = torch.zeros_like(feature_map)
    masks_col = torch.zeros_like(feature_map)
  
    row_mask = row_mask.unsqueeze(1).unsqueeze(-1) 
    masks_row = masks_row + row_mask
    
    col_mask = col_mask.unsqueeze(1).unsqueeze(2) 
    masks_col = masks_col + col_mask

    masks = masks_row * masks_col

    masked_feature_map = feature_map * masks

    return masked_feature_map


# Example usage:
feature_map = torch.randn(2, 3, 3, 4)  # Example feature map with shape (batch_size, channels, height, width)
box_coordinates = torch.tensor([[0, 0, 0, 0], [2, 2, 3, 4]], dtype=torch.float32)  # Example box coordinates for two batches

masked_feature_map = mask_feature_map(feature_map, box_coordinates)
print(masked_feature_map)
