from collections import OrderedDict
import torch
import torch.nn as nn


class UNetBlock(nn.Module):
    def __init__(self, in_channels, features):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=features, kernel_size=3, padding=1, bias=False)
        self.dp = nn.Dropout(inplace=False)
        self.conv2 = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, bias=False)
        self.relu = nn.LeakyReLU(inplace=True)
    def forward(self, x):
        out = self.relu(self.dp(self.conv1(x)))
        out = self.relu(self.dp(self.conv2(out)))
        return out

class NestedUNet(nn.Module):

    def __init__(self, in_channels=1, out_channels=1, init_features=32, deep_supervision=True):
        super().__init__()
        self.deep_supervision = deep_supervision
        features = init_features
        
        self.pool = nn.MaxPool2d(2,2)
        self.dp = nn.Dropout(inplace=False)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) ##
        
        self.conv0_0 = UNetBlock(in_channels, features)
        self.conv1_0 = UNetBlock(features, features*2)
        self.conv2_0 = UNetBlock(features*2, features*4)
        self.conv3_0 = UNetBlock(features*4, features*8)
        self.conv4_0 = UNetBlock(features*8, features*16)

        self.conv0_1 = UNetBlock(features+features*2, features)
        self.conv1_1 = UNetBlock(features*2+features*4, features*2)
        self.conv2_1 = UNetBlock(features*4+features*8, features*4)
        self.conv3_1 = UNetBlock(features*8+features*16, features*8)

        self.conv0_2 = UNetBlock(features*2+features*2, features)
        self.conv1_2 = UNetBlock(features*2*2+features*4, features*2)
        self.conv2_2 = UNetBlock(features*4*2+features*8, features*4)

        self.conv0_3 = UNetBlock(features*3+features*2, features)
        self.conv1_3 = UNetBlock(features*2*3+features*4, features*2)

        self.conv0_4 = UNetBlock(features*4+features*2, features)

        if self.deep_supervision:
            self.final1 = nn.Conv2d(features, out_channels, kernel_size=1)
            self.final2 = nn.Conv2d(features, out_channels, kernel_size=1)
            self.final3 = nn.Conv2d(features, out_channels, kernel_size=1)
            self.final4 = nn.Conv2d(features, out_channels, kernel_size=1)
        else:
            self.final = nn.Conv2d(features, out_channels, kernel_size=1)

    def forward(self, x):

        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.dp(self.pool(x0_0)))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.dp(self.pool(x1_0)))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.dp(self.pool(x2_0)))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.dp(self.pool(x3_0)))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]
        else:
            output = self.final(x0_4)
            return output

    
def run_cnn():
    return NestedUNet()

"""
def hook_feature(module, input, output):
    feature_blobs.append(output.cpu().data.numpy())

a = torch.ones(2, 1, 512, 512)
m = run_cnn()
m._modules.get("conv").register_forward_hook(hook_feature)
feature_blobs = []
weights = np.squeeze(list(m.parameters())[-2].cpu().data.numpy())
m(a)
for i in range(2):
    _ = weights.dot(feature_blobs[0][i, :, :, :])
    _ = _.reshape(
print(feature_blobs[0].shape, weights.shape)
"""
