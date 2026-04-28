import torch

def calculate_dice_score ( pred_logits, true_masks, threshold = 0.5):
  # using sigmoid to make value range from 0~1 
  probs = torch.sigmoid(pred_logits)

  # foreground of background
  preds = (probs>threshold).float()

  preds = preds.view(-1)
  true_masks = true_masks.view(-1)

  intersection = (preds * true_masks).sum()

  dice = (2. * intersection + 1e-6) / (preds.sum() + true_masks.sum() + 1e-6)

  return dice