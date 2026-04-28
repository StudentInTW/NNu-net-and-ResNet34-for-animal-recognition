import os
import torch
import torch.nn as nn
import torch.optim as optim
from src.oxford_pet import get_dataloaders
from src.models.unet import UNet
from src.models.resnet34_unet import ResNet34_UNet
from src.evaluate import evaluate

class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        
        # 把 [Batch, 1, 256, 256] 攤平成 [Batch, 65536]
        # 每張圖片獨立計算交集，最後再平均
        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)
        
        intersection = (probs * targets).sum(1)
        dice_score = (2. * intersection + 1e-6) / (probs.sum(1) + targets.sum(1) + 1e-6)
        dice_loss = 1.0 - dice_score.mean() # 取整個 Batch 的平均
        
        return bce_loss + dice_loss


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"I'm using {device}")
  
    # Hyperparameters
    batch_size = 16 
    epochs = 100
    learning_rate = 2e-4
    
    # relative path
    data_dir = './dataset/oxford-iiit-pet'
    save_dir = './saved_models'
  
    os.makedirs(save_dir, exist_ok=True) 
  
    # Train UNet (Change it to ResNet34_Unet if you'd like to)
    model = UNet(in_channels=3, out_channels=1).to(device)
    # model = ResNet34_UNet(in_channels=3, out_channels=1).to(device)

    # data prep
    train_loader, val_loader = get_dataloaders(data_dir, batch_size=batch_size)

    # loss and optimizer
    criterion = BCEDiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
  
    # Scheduler: obeserve Val Dice， LR cut half it no improvement within 2 epoch
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    best_val_dice = 0.0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch_idx, (images, masks) in enumerate(train_loader):
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)
        val_dice = evaluate(model, val_loader, device)
        print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {avg_train_loss:.4f} | Val Dice: {val_dice:.4f}")
    
      
        scheduler.step(val_dice)

        if val_dice > best_val_dice:
            best_val_dice = val_dice
           
            save_path = os.path.join(save_dir, 'best_unet.pth') 
            # save_path = os.path.join(save_dir,'resnet34_unet.pth')
            torch.save(model.state_dict(), save_path)
            print(f" => Saved new best model with Val Dice: {best_val_dice:.4f}")

if __name__ == '__main__':
    train()