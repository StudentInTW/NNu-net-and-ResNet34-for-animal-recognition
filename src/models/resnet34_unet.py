import torch 
import torch.nn as nn


# CBAM
class ChannelAttention(nn.Module):
  def __init__(self, in_planes,ratio = 16):  
    super(ChannelAttention, self).__init__()
    self.avg_pool = nn.AdaptiveAvgPool2d(1)
    self.max_pool = nn.AdaptiveMaxPool2d(1)
    self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
    self.relu1 = nn.ReLU()
    self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
    self.sigmoid = nn.Sigmoid()

  def forward(self,x):
    avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
    max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
    out= avg_out + max_out
    return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        result = out * self.sa(out)
        return result


class BasicBlock(nn.Module):
  def __init__(self, in_channels,out_channels,stride = 1 ):
    super().__init__()
    # first layer of convolution
    self.conv1= nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=stride,padding=1,bias=False)
    self.bn1 = nn.BatchNorm2d(out_channels)
    self.relu = nn.ReLU(inplace=True)
    
    #second layer of convolution
    # remain same channels
    self.conv2 = nn.Conv2d(out_channels,out_channels,kernel_size=3, stride = 1, padding = 1,bias = False)
    self.bn2 = nn.BatchNorm2d(out_channels)

    # if input shrinks or has more channel after block,  make original input the same shape
    # just for residual connection
    self.shortcut = nn.Sequential()
    if stride !=1 or in_channels != out_channels:
      self.shortcut = nn.Sequential(
        nn.Conv2d(in_channels,out_channels,kernel_size=1,stride = stride, bias = False),
        nn.BatchNorm2d(out_channels)
      )
  

  def forward(self,x):
    identity = self.shortcut(x)
    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)

    # This is where the connection happens
    out+=identity
    out = self.relu(out)

    return out


class DoubleConv(nn.Module):
  def __init__(self, in_channels, out_channels):
    super().__init__()
    self.double_conv = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )
    self.cbam = CBAM(out_channels)

  def forward(self, x):
    x=self.double_conv(x)
    x=self.cbam(x)
    return x

class ResNet34_UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        
        # -----------------------------------------
        # conv1: 7x7 , 64 channels , stride=2 (half of width and height)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        # conv1 for Skip Connection
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # 1/4 width and height
        
        # conv2_x: 3  BasicBlock remain 64 channels
        self.layer1 = self._make_layer(64, 64, blocks=3, stride=1)
        
        # conv3_x: 4  BasicBlock (128 channel  width and height=> 1/8)
        self.layer2 = self._make_layer(64, 128, blocks=4, stride=2)
        
        # conv4_x: 6  BasicBlock (256 channels   1/16)
        self.layer3 = self._make_layer(128, 256, blocks=6, stride=2)
        
        # conv5_x: 3  BasicBlock (512 channels   1/32)
        self.layer4 = self._make_layer(256, 512, blocks=3, stride=2)
        
        # -----------------------------------------
        # Decoder (UNet) - up scaling and concat Encoder 
        
        # 1/32 to 1/16
        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv_up4 = DoubleConv(256 + 256, 256) # left (256) + down(256) = 512 -> conv 256
        
        #  1/16 to 1/8
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(128 + 128, 128)
        
        #  1/8 to 1/4
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(64 + 64, 64)
        
        #  1/4 to 1/2
        self.up1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(64 + 64, 64)
        
        #  1/2 to (1x)
        self.up0 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv_up0 = DoubleConv(32, 32) # no Skip connection just conv
        
        # final output
        self.outc = nn.Conv2d(32, out_channels, kernel_size=1)

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        """a tool that can help us concat mutiple BasicBlock """
        layers = []
        # 第一個 Block downsampling (stride=2)
        layers.append(BasicBlock(in_channels, out_channels, stride))
        # make network deeper (stride=1)
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        # ---------- Encoder 過程 ----------
        # x 假設是 [Batch, 3, 256, 256]
        e1 = self.conv1(x)               # [B, 64, 128, 128]
        
        x_pool = self.maxpool(e1)        # [B, 64, 64, 64]
        e2 = self.layer1(x_pool)         # [B, 64, 64, 64]
        
        e3 = self.layer2(e2)             # [B, 128, 32, 32]
        e4 = self.layer3(e3)             # [B, 256, 16, 16]
        e5 = self.layer4(e4)             # [B, 512, 8, 8]  <- 這是最底層的 Bottleneck
        
        # ---------- Decoder 過程 ----------
        d4 = self.up4(e5)                # 放大成 [B, 256, 16, 16]
        d4 = torch.cat([e4, d4], dim=1)  # 與 e4 拼接 -> [B, 512, 16, 16]
        d4 = self.conv_up4(d4)           # 卷積成 [B, 256, 16, 16]
        
        d3 = self.up3(d4)
        d3 = torch.cat([e3, d3], dim=1)
        d3 = self.conv_up3(d3)
        
        d2 = self.up2(d3)
        d2 = torch.cat([e2, d2], dim=1)
        d2 = self.conv_up2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat([e1, d1], dim=1)
        d1 = self.conv_up1(d1)
        
        # 最後再放大一次，回到 256x256
        d0 = self.up0(d1)
        d0 = self.conv_up0(d0)
        
        logits = self.outc(d0)           # [B, 1, 256, 256]
        return logits












      