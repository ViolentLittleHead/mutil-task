import numpy as np
import torch
from torch import nn
from mamba_ssm import Mamba
from torch.nn.modules.utils import _pair


"""
    input (B, n_patch, mambaChannel)
    output (B, decoderChannel, h , w)
"""
class DATMambaLayer(nn.Module):
    def __init__(self, dim, n_patch, skip_connection_nums=4, d_state=16, d_conv=4, expand=2):
        super().__init__()
        print(f"MambaLayer: dim: {dim}")
        self.dim = dim
        self.n_patch = n_patch
        self.cnorm = nn.LayerNorm(dim)
        self.snorm = nn.LayerNorm(dim // 4)  # 修改为通道维度
        self.channel_mamba = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.space_mamba = Mamba(
            d_model=dim // 4,  # 修改为通道维度
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        # 动态融合模块
        self.fusion = nn.Sequential(
            nn.Conv1d(2 * (dim // 4), 2, kernel_size=1),  # 输入为双倍通道
            nn.Sigmoid()
        )

        self.reconstruct = []
        self.reconstruct.append(Reconstruct(dim // skip_connection_nums, 64, (16, 16)))
        factor = 1
        for _ in range(skip_connection_nums - 1):
            self.reconstruct.append(Reconstruct(dim // skip_connection_nums, 64 * factor, (8 // factor, 8 // factor)))
            factor *= 2
        self.reconstructs = nn.Sequential(*self.reconstruct)

    def forward(self, x1, x2, x3, x4):

        # 通道处理路径
        c1 = torch.cat([x1, x2, x3, x4], dim=2)  # (B, n_patch, dim)
        m1_channel = self.channel_mamba(self.cnorm(c1))
        # 分割并添加层级残差
        cx1, cx2, cx3, cx4 = torch.chunk(m1_channel, 4, dim=2)
        cx1 = cx1 + x1  # 残差连接1
        cx2 = cx2 + x2  # 残差连接2
        cx3 = cx3 + x3  # 残差连接3
        cx4 = cx4 + x4  # 残差连接4
        m1_channel_processed = torch.cat([cx1, cx2, cx3, cx4], dim=1)

        # ------------------- 空间处理路径 + 残差 -------------------
        s1 = torch.cat([x1, x2, x3, x4], dim=1)  # (B, 4*n_patch, dim//4)
        m1_space = self.space_mamba(self.snorm(s1))
        m1_space = m1_space + s1  # 全局残差连接

        # 动态融合
        combined = torch.cat([m1_channel_processed, m1_space], dim=2)  # (B, 4*n_patch, 2*(dim//4))
        combined = combined.transpose(1, 2)  # (B, 2*(dim//4), 4*n_patch)
        fusion_weights = self.fusion(combined)  # (B, 2, 4*n_patch)
        fusion_weights = fusion_weights.transpose(1, 2)  # (B, 4*n_patch, 2)

        fused_feat = (fusion_weights[:, :, 0:1] * m1_channel_processed +
                      fusion_weights[:, :, 1:2] * m1_space)  # (B, 4*n_patch, dim//4)

        # 分割融合结果
        sx1, sx2, sx3, sx4 = torch.chunk(fused_feat, 4, dim=1)

        split_tensors = sx1,sx2,sx3,sx4
        i = 0
        res = []
        for t in split_tensors:
            res.append(self.reconstructs[i](t))
            i += 1
        return (*res,)


class Reconstruct(nn.Module):
    """
    reshape from (B, n_patch, hidden) to (B, hidden, h, w)
    out_channels = [64, 128, 256, 512]
    """

    def __init__(self, in_channels, out_channels, scale_factor, kernel_size=1):
        super(Reconstruct, self).__init__()
        padding = 1 if kernel_size == 3 else 0

        # 可分离卷积替换标准卷积
        self.conv = nn.Sequential(
            # 深度卷积
            nn.Conv2d(in_channels, in_channels,
                      kernel_size=kernel_size,
                      padding=padding,
                      groups=in_channels),
            nn.GELU(),
            nn.BatchNorm2d(in_channels),

            # 逐点扩展
            nn.Conv2d(in_channels, in_channels * 4, 1),
            nn.GELU(),
            nn.BatchNorm2d(in_channels * 4),

            # 逐点收缩
            nn.Conv2d(in_channels * 4, out_channels, 1),
            nn.GELU(),
            nn.BatchNorm2d(out_channels),
        )
        self.scale_factor = scale_factor

    def forward(self, x):
        B, n_patch, hidden = x.size()
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = x.permute(0, 2, 1).contiguous().view(B, hidden, h, w)
        if self.scale_factor[0] > 1:
            x = nn.Upsample(scale_factor=self.scale_factor)(x)
        return self.conv(x)


class Spatial_Embeddings(nn.Module):
    """
    Construct the embeddings from patch, position embeddings.
    """

    def __init__(self, mamba_inchannel, patchsize, img_size, in_channels):
        super().__init__()
        img_size = _pair(img_size)
        patch_size = _pair(patchsize)
        n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])

        # 可分离卷积替换标准卷积
        self.patch_embeddings = nn.Sequential(
            # 深度卷积
            nn.Conv2d(in_channels, in_channels,
                      kernel_size=patch_size,
                      stride=patch_size,
                      groups=in_channels),
            nn.GELU(),
            nn.BatchNorm2d(in_channels),

            # 逐点卷积
            nn.Conv2d(in_channels, mamba_inchannel // 4, 1),
            nn.GELU(),
            nn.BatchNorm2d(mamba_inchannel // 4),
        )
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, mamba_inchannel // 4))

    def forward(self, x):
        x = self.patch_embeddings(x)
        x = x.flatten(2).transpose(-1, -2)
        return x + self.position_embeddings


class CSM(nn.Module):
    def __init__(self, img_size, mamba_dim,channel_num = [64, 64, 128, 256], patchSize = [16, 8, 4, 2],skip_connection_nums=4):
        super().__init__()
        self.patchSize_1 = patchSize[0] # 16
        self.patchSize_2 = patchSize[1] # 8
        self.patchSize_3 = patchSize[2] # 4
        self.patchSize_4 = patchSize[3] # 2
        self.embeddings_1 = Spatial_Embeddings(mamba_dim,self.patchSize_1, img_size=img_size,    in_channels=channel_num[0]) # 64
        self.embeddings_2 = Spatial_Embeddings(mamba_dim,self.patchSize_2, img_size=img_size//2, in_channels=channel_num[1]) # 128
        self.embeddings_3 = Spatial_Embeddings(mamba_dim,self.patchSize_3, img_size=img_size//4, in_channels=channel_num[2]) # 256
        self.embeddings_4 = Spatial_Embeddings(mamba_dim,self.patchSize_4, img_size=img_size//8, in_channels=channel_num[3]) # 512
        n_patch = (img_size // self.patchSize_1) * (img_size // self.patchSize_1)
        self.datmamba = DATMambaLayer(mamba_dim,n_patch,skip_connection_nums)


    def forward(self,en1,en2,en3,en4):
        emb1 = self.embeddings_1(en1)
        emb2 = self.embeddings_2(en2)
        emb3 = self.embeddings_3(en3)
        emb4 = self.embeddings_4(en4)

        o1, o2, o3, o4 = self.datmamba(emb1,emb2,emb3,emb4)

        """
            torch.Size([B, 64, 224, 224])
            torch.Size([B, 128, 112, 112])
            torch.Size([B, 256, 56, 56])
            torch.Size([B, 512, 28, 28])
        """
        return o1, o2, o3, o4


if __name__=='__main__':
    mamba_dim = 128 * 4
    datm = CSM(224,mamba_dim)
    x1 = torch.rand((1, 64, 224, 224))
    x2 = torch.rand((1, 64, 112, 112))
    x3 = torch.rand((1, 128, 56, 56))
    x4 = torch.rand((1, 256, 28, 28))
    emb1, emb2, emb3, emb4 = datm(x1,x2,x3,x4)
    print(emb1.shape)
    print(emb2.shape)
    print(emb3.shape)
    print(emb4.shape)



