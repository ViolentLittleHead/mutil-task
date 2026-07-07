import torch
import torch.nn as nn
import cv2
import numpy as np
from torchvision.ops import roi_align
import torch.nn.functional as F
from segmentation_model.MNSGNet import MNSGNet


# ========= 基础模块 =========
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class ChannelSelector(nn.Module):
    def __init__(self, in_channels=256, reduction=2, k=16):
        super().__init__()
        mid = in_channels // reduction
        self.k = k

        self.score_mlp = nn.Sequential(
            nn.Linear(in_channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, in_channels),
            nn.Sigmoid()
        )

    def forward(self, feat):
        """
        feat: (B, C, H, W)
        """
        B, C, H, W = feat.shape

        # 1. GAP → (B, C)
        gap = feat.mean(dim=[2, 3])  # global avg pool

        # 2. 得到每个通道的 score
        scores = self.score_mlp(gap)  # (B, C)

        # 3. 获取 top-k 通道索引
        _, idx = torch.topk(scores, self.k, dim=1)  # idx: (B, K)

        # 4. 按 batch 选取通道
        #    构造一个批次选择的 index
        idx = idx.unsqueeze(-1).unsqueeze(-1).expand(B, self.k, H, W)

        selected = torch.gather(feat, dim=1, index=idx)

        return selected  # (B, K, H, W)

class DiscClassifier_CASAB(nn.Module):
    def __init__(self, feat_channels=256, img_channels=1, num_classes=3, topk=16):
        super().__init__()
        self.num_classes = num_classes

        # 简单 attention map（语义权重来自 segmentation features）
        self.att_gen = nn.Sequential(
            nn.Conv2d(feat_channels, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid()
        )

        # 选 ROI 特征中 top-k 通道
        self.selector = ChannelSelector(in_channels=feat_channels, k=topk)

        in_conv = img_channels + topk  # 1 灰度通道 + topk 特征通道

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_conv, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )

        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_classes)
        )

    def forward(self, roi_feat, roi_img):
        # attention map from segmentation features
        att = self.att_gen(roi_feat)
        fused = roi_img * att  # 调制后的图像

        # top-k 通道
        selected_feat = self.selector(roi_feat)

        # 拼接
        x = torch.cat([fused, selected_feat], dim=1)

        x = self.conv_block(x)
        x = x.flatten(1)
        return self.fc(x)

class SoftMorphConv(nn.Module):
    """
    可学习的软形态学模块（Soft Morph Conv）
    思路：
      - 使用 unfold 提取每个像素的 k*k 补丁
      - 每个补丁乘以可学习的 kernel 权重，然后使用温度 softmax 做加权和（近似 max pooling）
      - soft_dilate: 对输入 x 做 soft-dilation
      - soft_erode: -soft_dilate(-x)
      - forward 做一次 dilation + erosion（closing），可以直接用作修复断裂后的小洞/连接
    参数：
      - kernel_size: 卷积核尺寸（奇数）
      - beta: softmax 温度（越大越接近 hard max）
      - edge_decay_init: 用于初始化边缘衰减参数（控制有效核范围）
    """
    def __init__(self, kernel_size=15, beta=20.0, edge_decay_init=2.0):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd"
        self.k = kernel_size
        self.beta = beta

        # kernel 权重（未约束，可以学习出任意形状）
        self.kernel_w = nn.Parameter(torch.randn(self.k * self.k))

        # 边缘衰减参数：通过一个可学习标量控制边缘权重（越大边缘越小）
        # 这里我们用一个可学习的 log-space 参数，保证可正可负
        self.edge_decay_log = nn.Parameter(torch.tensor(float(edge_decay_init)))

        # 生成固定的距离权重矩阵坐标（中心为0，边缘为正）
        coords = []
        half = self.k // 2
        for yi in range(-half, half+1):
            for xi in range(-half, half+1):
                coords.append(xi*xi + yi*yi)
        self.register_buffer("coord_sq", torch.tensor(coords, dtype=torch.float32))  # (k*k,)

    def _soft_dilate_once(self, x):
        """
        x: [B, 1, H, W] 或 [B, C, H, W]
        返回：同形状输出
        """
        B, C, H, W = x.shape
        pad = self.k // 2
        # pad 输入以保尺寸
        x_padded = F.pad(x, (pad, pad, pad, pad), mode='reflect')  # [B, C, H+2p, W+2p]

        # 使用 unfold 取补丁: -> [B, C * k*k, L] where L = H*W
        patches = F.unfold(x_padded, kernel_size=self.k, stride=1)  # [B, C*k*k, L]
        L = patches.shape[-1]
        patches = patches.view(B, C, self.k*self.k, L)  # [B, C, k*k, L]

        # kernel 权重和边缘衰减
        # kernel_w -> (k*k,)
        kw = self.kernel_w  # 可为正负
        # 边缘衰减 mask: coord_sq * decay_factor -> 越远的元素被更强地衰减
        decay = torch.exp(-torch.relu(self.edge_decay_log))  # map to (0,1], 较大 log -> 较小 decay
        edge_mask = torch.exp(-self.coord_sq * decay)  # (k*k,)
        # 组合 kernel 权重与 edge_mask（elementwise）
        effective_w = kw * edge_mask  # (k*k,)

        # 将权重加到补丁值上（模拟 x + K），然后按 k*k 做 softmax pooling
        # patches: [B, C, k*k, L]
        # effective_w: (k*k,)
        # 为 numerical stability，用 beta * (patch + w)
        # expand到 [B,C,k*k,L]
        w_exp = effective_w.view(1, 1, -1, 1)
        val = patches + w_exp  # [B, C, k*k, L]

        # softmax over k*k dim
        weighted = F.softmax(self.beta * val, dim=2) * val  # [B, C, k*k, L]
        out = weighted.sum(dim=2)  # [B, C, L]

        out = out.view(B, C, H, W)
        return out

    def soft_dilate(self, x):
        return self._soft_dilate_once(x)

    def soft_erode(self, x):
        # erosion approximated by -dilate(-x)
        return -self._soft_dilate_once(-x)

    def forward(self, x):
        # closing: dilation followed by erosion (可以根据需要改为开运算等)
        d = self.soft_dilate(x)
        c = self.soft_erode(d)
        return c

class End2EndModel(nn.Module):
    def __init__(self, in_channels=1, num_classes=3, base=32,
                 roi_margin=0.3, min_height_expand=6, min_weight_expand=6,
                 min_area=30, repair_kernel=(2,7),
                 use_learnable_morph=True,
                 morph_kernel_size=15):
        super().__init__()
        # backbone & classifier（请确保 MNSGNet 与 DiscClassifier_CASAB 在别处定义）
        self.backbone = MNSGNet(in_ch=in_channels, out_ch=1, img_size=256)
        self.classifier = DiscClassifier_CASAB(
            feat_channels=256,  # 对应 c4 的通道数，请与你的 backbone 保持一致
            img_channels=in_channels,
            num_classes=num_classes
        )

        # ROI 与预处理参数
        self.roi_margin = roi_margin
        self.min_height_expand = min_height_expand
        self.min_weight_expand = min_weight_expand
        self.min_area = min_area
        self.repair_kernel = repair_kernel  # 仍然保留此字段以便兼容，但当 use_learnable_morph=True 时不使用它

        # 可学习形态学模块（可选）
        self.use_learnable_morph = use_learnable_morph
        if self.use_learnable_morph:
            self.morph_refine = SoftMorphConv(kernel_size=morph_kernel_size, beta=20.0, edge_decay_init=2.0)
        else:
            self.morph_refine = None

    def forward(self, x, gt_mask=None, use_gt_mask=True, return_rois=False):
        """
        return_rois: 是否返回 ROI 坐标（推理 / 可视化时 True）
        """
        seg_pred, feat_map = self.backbone(x)
        B, C_feat, Hf, Wf = feat_map.shape

        all_rois = []
        all_img_patches = []
        num_rois_per_sample = []

        # ===== NEW: 保存每个样本的 ROI bbox（原图坐标）=====
        batched_roi_boxes = [[] for _ in range(B)]

        for b in range(B):

            # -------- mask source --------
            if use_gt_mask and gt_mask is not None:
                mask_src = gt_mask[b, 0]
            else:
                mask_src = (seg_pred[b, 0] > 0.5).float()

            # -------- morphology --------
            if self.use_learnable_morph and self.morph_refine is not None:
                mask_tensor = mask_src.unsqueeze(0).unsqueeze(0)
                mask_refined = self.morph_refine(mask_tensor)
                mask_np = (mask_refined[0, 0].detach().cpu().numpy() > 0.5).astype(np.uint8) * 255
            else:
                mask_np = (mask_src.detach().cpu().numpy() > 0.5).astype(np.uint8) * 255

            # -------- connected components --------
            num_labels, labels = cv2.connectedComponents(mask_np)
            roi_boxes = []

            for i in range(1, num_labels):
                ys, xs = np.where(labels == i)
                if len(xs) == 0:
                    continue

                x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
                w, h = x2 - x1, y2 - y1
                if w * h < self.min_area:
                    continue

                margin_x = max(int(w * self.roi_margin), self.min_weight_expand)
                margin_y = max(int(h * self.roi_margin), self.min_height_expand)

                x1 = max(0, x1 - margin_x)
                x2 = min(mask_np.shape[1] - 1, x2 + margin_x)
                y1 = max(0, y1 - margin_y)
                y2 = min(mask_np.shape[0] - 1, y2 + margin_y)

                roi_boxes.append((x1, y1, x2, y2))

            # -------- 排序（从上到下）--------
            roi_boxes = sorted(roi_boxes, key=lambda b: b[1])
            num_rois_per_sample.append(len(roi_boxes))

            # ===== NEW: 保存原图坐标 ROI =====
            batched_roi_boxes[b] = roi_boxes

            # -------- 构建 roi_align + img_patch --------
            for (x1, y1, x2, y2) in roi_boxes:
                scale_x = Wf / x.shape[-1]
                scale_y = Hf / x.shape[-2]

                x1f, x2f = x1 * scale_x, x2 * scale_x
                y1f, y2f = y1 * scale_y, y2 * scale_y

                all_rois.append(
                    torch.tensor([b, x1f, y1f, x2f, y2f],
                                 dtype=feat_map.dtype,
                                 device=feat_map.device)
                )

                img_patch = x[b:b + 1, :, y1:y2, x1:x2]
                img_patch = F.interpolate(img_patch, size=(32, 32), mode='bilinear', align_corners=False)
                all_img_patches.append(img_patch.squeeze(0))

        # -------- 无 ROI --------
        if len(all_rois) == 0:
            empty_preds = [
                torch.zeros((0, self.classifier.fc[-1].out_features), device=x.device)
                for _ in range(B)
            ]
            if return_rois:
                return seg_pred, empty_preds, batched_roi_boxes
            return seg_pred, empty_preds

        # -------- 分类 --------
        rois = torch.stack(all_rois)
        roi_features = roi_align(feat_map, rois, output_size=(32, 32))
        img_patches = torch.stack(all_img_patches)

        class_preds_all = self.classifier(roi_features, img_patches)

        batched_cls_preds = []
        ptr = 0
        for n in num_rois_per_sample:
            batched_cls_preds.append(class_preds_all[ptr:ptr + n])
            ptr += n

        # ===== NEW: 根据 return_rois 决定输出 =====
        if return_rois:
            return seg_pred, batched_cls_preds, batched_roi_boxes
        else:
            return seg_pred, batched_cls_preds


if __name__ == "__main__":
    model = End2EndModel(in_channels=1, num_classes=3, base=32).cuda()
    x = torch.randn(2, 1, 256, 256).cuda()

    # 构造模拟GT mask（两个椎间盘）
    gt_mask = torch.zeros((2, 1, 256, 256)).float().cuda()
    gt_mask[0, 0, 30:60, 100:150] = 1
    gt_mask[0, 0, 100:130, 100:150] = 1
    gt_mask[1, 0, 50:90, 120:160] = 1
    gt_mask[1, 0, 100:130, 100:150] = 1

    seg_pred, cls_pred = model(x, gt_mask, use_gt_mask=True)

    print("seg_pred:", seg_pred.shape)  # [2,1,H,W]
    print("cls_pred batch lens:", [p.shape[0] for p in cls_pred])  # 每张图的ROI数量
