import torch
import torch.nn as nn


"""
For Conv2D we have used kernel =4, stride =2 and padding =1
Why if input image =256x256 then ouput feature map will be 
ouput=((input+2padding-kernel_size)/stride)+1=((256+2*1-4)/2)+1=128 perfect halving

In ConvTranspose2D it will be ouput=(input-1)*stride-2*padding+kernel_size 
for input of 128x128 the output will be 256 which is perfect doubling,
Hence these chooson parameters are for perfect halving and perfect doubling
"""

class UpSample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channel = channels
        self.upsample = nn.ConvTranspose2d(in_channels=channels,out_channels=channels,kernel_size=4,stride=2,padding=1)

    def forward(self, x):
        upsampled_tensor = self.upsample(x)
        return upsampled_tensor


class DownSample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channel = channels
        self.downsample = nn.Conv2d(in_channels=channels,out_channels=channels,kernel_size=4,stride=2,padding=1)

    def forward(self, x):
        downsampled_tensor = self.downsample(x)
        return downsampled_tensor