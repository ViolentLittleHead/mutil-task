import torch
from torch import nn
from segmentation_model.CSM import CSM
from torchsummary import summary
import torchvision.models as models
from segmentation_model.mona import Mona
from segmentation_model.AttentionFusion import AttentionFusion
from torchprofile import profile_macs
from ptflops import get_model_complexity_info

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)


class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, decoder):
        x = self.up(decoder)
        return x


class MNSGNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, img_size=256, num_classes=3):
        super(MNSGNet, self).__init__()

        resnet = models.resnet34(weights=None)

        self.conv1 = DoubleConv(in_ch, 64)
        self.Maxpool = nn.MaxPool2d(2)

        self.Conv2 = resnet.layer1  # 64
        self.Conv3 = resnet.layer2  # 128
        self.Conv4 = resnet.layer3  # 256
        self.Conv5 = resnet.layer4  # 512

        self.attention = AttentionFusion(512)

        self.up6 = Up(512, 256)
        self.conv6 = DoubleConv(512, 256)

        self.up7 = Up(256, 128)
        self.conv7 = DoubleConv(256, 128)

        self.up8 = Up(128, 64)
        self.conv8 = DoubleConv(128, 64)

        self.up9 = Up(64, 64)
        self.conv9 = DoubleConv(128, 64)

        self.conv10 = nn.Conv2d(64, out_ch, 1)

        self.cATMambaBlock = CSM(img_size, mamba_dim=128 * 4)

    def forward(self, x):
        c1 = self.conv1(x)
        p1 = self.Maxpool(c1)

        c2 = self.Conv2(p1)
        c3 = self.Conv3(c2)
        c4 = self.Conv4(c3)
        c5 = self.Conv5(c4)

        encoder_out = self.attention(c5) + c5
        ca1, ca2, ca3, ca4 = self.cATMambaBlock(c1, c2, c3, c4)

        up_6 = self.up6(encoder_out)
        merge6 = torch.cat([up_6, ca4], dim=1)
        c6 = self.conv6(merge6)

        up_7 = self.up7(c6)
        merge7 = torch.cat([up_7, ca3], dim=1)
        c7 = self.conv7(merge7)

        up_8 = self.up8(c7)
        merge8 = torch.cat([up_8, ca2], dim=1)
        c8 = self.conv8(merge8)

        up_9 = self.up9(c8)
        merge9 = torch.cat([up_9, ca1], dim=1)
        c9 = self.conv9(merge9)

        seg_pred = torch.sigmoid(self.conv10(c9))

        # 返回中层特征 c4（shape=256 channels）供分类用
        feat_map = c4

        return seg_pred, feat_map


if __name__ == '__main__':
    h, w = 256, 256
    model = MNSGNet(1, 1, img_size=h)
    model = model.to('cuda')
    model.eval()

    summary(model, (1, h, w))
    # 计算 GFLOPs
    # dummy_input = torch.randn(1, 1, h, w).to('cuda')
    # macs = profile_macs(model, dummy_input)
    # gflops = macs / 1e9
    # print(f"G FLOPs: {gflops:.3f}")

    # 使用ptflops 计算GFLOPs
    # with torch.cuda.device(0):
    #     macs, params = get_model_complexity_info(model, (1, h, w), as_strings=True, print_per_layer_stat=True)
    #     print(f'MACs: {macs}')
    #     print(f'Params: {params}')


