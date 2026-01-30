import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
from segmentation_model.mona import Mona


class MambaBlock(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.mamba = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

    def forward(self, x):
        # 输入形状转换: (B, C, H, W) -> (B, L, D)
        B, C, H, W = x.shape
        L = H * W
        x = x.view(B, C, L).permute(0, 2, 1)  # (B, L, D)

        # Mamba处理
        x = self.mamba(x)

        # 恢复形状: (B, L, D) -> (B, C, H, W)
        x = x.permute(0, 2, 1).view(B, C, H, W)
        return x


class AttentionFusion(nn.Module):
    def __init__(self, in_channels, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.in_channels = in_channels

        # 空间信息路径（Mamba）
        self.mamba_path = MambaBlock(
            dim=in_channels,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )

        # 多尺度信息路径（Mona）
        self.mona_path = Mona(
            in_channels=in_channels,
            inner_dim=max(in_channels // 4, 4)
        )

        # 可学习的缩放参数
        self.scale = nn.Parameter(torch.tensor(1.0))

        # 输出层
        self.output_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        L = H * W

        # 1. 双路径处理
        spatial_feat = self.mamba_path(x)  # [B, C, H, W]
        multi_scale_feat = self.mona_path(x)  # [B, C, H, W]

        # 2. 准备Query, Key, Value
        # Query: 原始输入
        query = x.view(B, C, L)  # [B, C, L]

        # Key/Value: 双路径特征的拼接
        key_value = torch.cat([
            spatial_feat.view(B, C, L),
            multi_scale_feat.view(B, C, L)
        ], dim=1)  # [B, 2C, L]

        # 3. 转置维度以匹配矩阵乘法
        query = query  # [B, C, L]
        value = key_value  # [B, 2C, L] (作为Value)

        # 使用矩阵乘法 (需要转置)
        key_value = key_value.permute(0, 2, 1)  # [B, L, 2C]
        attn = torch.matmul(query, key_value)  # [B, C, 2C]

        # 4. 缩放注意力分数
        scale_factor = (key_value.size(1)) ** -0.5  # 1/sqrt(2C)
        attn = self.scale * scale_factor * attn

        # 5. 应用softmax
        attn = F.softmax(attn, dim=-1)  # [B, C, 2C]

        # 6. 特征融合
        fused = torch.matmul(attn, value)  # [B, L, L]

        # 7. 恢复空间维度
        fused = fused.view(B, C, H, W)  # [B, C, H, W]

        # 8. 残差连接 + 输出层
        output = fused
        output = self.output_conv(output)

        return output

# 测试代码
if __name__ == "__main__":
    input_tensor = torch.randn(2, 512, 16, 16)
    model = AttentionFusion(512)
    output = model(input_tensor)
    print(f"输入形状: {input_tensor.shape}")
    print(f"输出形状: {output.shape}")  # 应当保持(2, 256, 16, 16)
