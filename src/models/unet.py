import torch
import torch.nn as nn

class DoubleConv(nn.Module):
  def __init__(self, in_channels, out_channels):
    super().__init__()
    # I use padding = 1 so skip connnection technique can be use easily
    self.double_conv = nn.Sequential(
      nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
      nn.BatchNorm2d(out_channels),
      nn.ReLU(inplace=True),
      nn.Conv2d(out_channels,out_channels, kernel_size = 3, padding = 1),
      nn.BatchNorm2d(out_channels),
      nn.ReLU(inplace = True)
    )

  def forward(self, x):
      return self.double_conv(x)

class UNet(nn.Module):
  def __init__(self, in_channels=3, out_channels=1):
    super(UNet,self).__init__()
    # encoder => downsampling
    # in_channels => 64
    self.inc = DoubleConv(in_channels, 64)
    # width and height are constantly 0.5 size smaller
    # 256 -> 128 ->64 -> 32->16 ,but with deeper channel
    self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
    self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
    self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
    self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 1024))
    
    # --- Decoder  ---
    #  ConvTranspose2d to make width and height 2 tumes larger
    # width,height 16=>32 
    # but channel from 1024 to 512 ( for skip connection)
    self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
    self.conv_up1 = DoubleConv(1024, 512)
    
    # 512 to 256 for skip connection)
    self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
    self.conv_up2 = DoubleConv(512, 256)
    
    # skip connection)
    self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
    self.conv_up3 = DoubleConv(256, 128)
    
    # skip connection)
    self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
    self.conv_up4 = DoubleConv(128, 64)
    
    # --- Output Layer ---
    # 1 channel means foreground
    self.outc = nn.Conv2d(64, out_channels, kernel_size=1)

  def forward(self, x):
        # Encoder process，x1, x2, x3, x4 are for Skip connection 
        x1 = self.inc(x)         # [B, 64, 256, 256]
        x2 = self.down1(x1)      # [B, 128, 128, 128]
        x3 = self.down2(x2)      # [B, 256, 64, 64]
        x4 = self.down3(x3)      # [B, 512, 32, 32]
        x5 = self.down4(x4)      # Bottleneck: [B, 1024, 16, 16]
        
        # Decoder 過程
        x = self.up1(x5)         # up scale to [B, 512, 32, 32]
        x = torch.cat([x4, x], dim=1) #  concatenate x4 ，dim=1 mean concat channel (batchsize, channel, width, height)
        x = self.conv_up1(x)     #  [B, 512, 32, 32]
        
        x = self.up2(x)
        x = torch.cat([x3, x], dim=1)
        x = self.conv_up2(x)
        
        x = self.up3(x)
        x = torch.cat([x2, x], dim=1)
        x = self.conv_up3(x)
        
        x = self.up4(x)
        x = torch.cat([x1, x], dim=1)
        x = self.conv_up4(x)
        
        # 最後輸出
        logits = self.outc(x)    # [B, 1, 256, 256]
        return logits









