import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba

class Mona(nn.Module):
    def __init__(self, in_channels, inner_dim=64):
        super().__init__()

        # 下投影层 (通道压缩)
        self.down_proj = nn.Conv2d(in_channels, inner_dim, kernel_size=1)

        # 多尺度深度卷积分支
        self.dwconv3 = nn.Conv2d(inner_dim, inner_dim, 3, padding=1, groups=inner_dim)
        self.dwconv5 = nn.Conv2d(inner_dim, inner_dim, 5, padding=2, groups=inner_dim)
        self.dwconv7 = nn.Conv2d(inner_dim, inner_dim, 7, padding=3, groups=inner_dim)

        # 特征聚合层
        self.aggregate = nn.Conv2d(inner_dim, inner_dim, kernel_size=1)

        # 上投影层 (通道恢复)
        self.up_proj = nn.Conv2d(inner_dim, in_channels, kernel_size=1)

    def forward(self, x):
        # 保存原始输入用于最终残差
        identity_high = x

        # 下投影
        x = self.down_proj(x)
        identity_low = x

        # 多尺度特征提取
        branch3 = self.dwconv3(x)
        branch5 = self.dwconv5(x)
        branch7 = self.dwconv7(x)

        # 多尺度特征融合
        fused = (branch3 + branch5 + branch7) / 3.0

        # 第一次残差连接 (低维空间)
        fused += identity_low

        # 通道聚合
        aggregated = self.aggregate(fused)

        # 第二次残差连接 (聚合分支)
        out = aggregated + fused

        # 应用GELU激活
        out = F.gelu(out)

        # 上投影恢复通道
        out = self.up_proj(out)

        # 最终残差连接 (高维空间)
        out += identity_high

        return out


# ------------------------------ 使用示例 ------------------------------
if __name__ == "__main__":
    # 配置参数
    in_channels = 128

    spatial_size = 32

    # 创建测试数据
    x = torch.randn(2, in_channels, spatial_size, spatial_size)

    # 初始化模块
    model = Mona(in_channels)

    # 前向计算
    out = model(x)

    # 验证输出形状
    print(f"输入形状: {x.shape}")  # torch.Size([2, 128, 32, 32])
    print(f"输出形状: {out.shape}")  # torch.Size([2, 128, 32, 32])
    print(f"参数变化: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")