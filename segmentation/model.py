import torch
import torch.nn as nn
import torch.nn.functional as F

class MiniUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoders = nn.ModuleList([
            nn.Sequential(nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1), nn.ReLU()),
            nn.Sequential(nn.MaxPool2d(kernel_size=2), nn.Conv2d(16, 32, 3, padding=1), nn.ReLU()),
            nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(32, 64, 3, padding=1), nn.ReLU()),
            nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(64, 128, 3, padding=1), nn.ReLU()),
            nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(128, 256, 3, padding=1), nn.ReLU()),
        ])
        self.decoders = nn.ModuleList([
            nn.Sequential(nn.Conv2d(128+256, 128, 3, padding=1)),
            nn.Sequential(nn.Conv2d(64+128, 64, 3, padding=1)),
            nn.Sequential(nn.Conv2d(32+64, 32, 3, padding=1)),
            nn.Sequential(nn.Conv2d(16+32, 16, 3, padding=1))
        ])
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv1x1 = nn.Conv2d(16, 6, 1)

    
    def forward(self, x):
        skips = []
        for encoder in self.encoders[:-1]:
            x = encoder(x)
            skips.append(x)
        x = self.encoders[-1](x)
        for idx, decoder in enumerate(self.decoders):
            x = self.upsample(x)
            x = torch.cat([x, skips[-(idx+1)]], dim=1)
            x = F.relu(decoder(x))
        return self.conv1x1(x)
    

if __name__ == '__main__':
    model = MiniUNet()
    input_tensor = torch.rand(1,3,240,320)
    output = model(input_tensor)
    print("output size:", output.shape)
    print(model)


    