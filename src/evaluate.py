import torch
from src.utils import calculate_dice_score

def evaluate(model, val_loader, device):
  
  # turn off dropout and batchnorm
  model.eval()

  total_dice =0.0
  
  with torch.no_grad():
    for images , masks in val_loader:
      images = images.to(device)
      masks = masks.to(device)

      outputs = model(images)

      dice = calculate_dice_score(outputs, masks)
      total_dice += dice

  avg_dice = total_dice/len(val_loader)
  return avg_dice