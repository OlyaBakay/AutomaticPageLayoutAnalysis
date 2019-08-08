import torch

# PyTroch version

SMOOTH = 1e-6


def iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor, reduce=True):
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W
    outputs = outputs.sigmoid() > 0.5
    labels = labels.squeeze(1) > 0.5

    intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1, 2))  # Will be zzero if both are 0

    # print("!", intersection.shape, intersection)
    # print("#", union.shape, union)
    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0

    thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds

    if reduce:
        thresholded = thresholded.mean()

    return thresholded
