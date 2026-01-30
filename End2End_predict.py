import torch
import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from torchvision.transforms import ToTensor
from torchvision.ops import roi_align

from compute_metric import compute_seg_metrics
from network.End2End5 import End2EndModel

# ======================= #
# 图像与掩码加载函数
# ======================= #
def load_image_mask(img_path, mask_path, img_size=None):
    """
    加载灰度图和掩码图并转换为 [1,1,H,W] Tensor
    """
    img = np.array(Image.open(img_path).convert("L"), dtype=np.float32)
    mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)

    if img_size is not None:
        img = cv2.resize(img, (img_size[1], img_size[0]), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (img_size[1], img_size[0]), interpolation=cv2.INTER_NEAREST)

    # 归一化并转为 tensor，保持 batch_size=1, channel=1
    img = torch.tensor(img / 255.0).unsqueeze(0).unsqueeze(0)   # [1,1,H,W]
    mask = torch.tensor(mask / 255.0).unsqueeze(0).unsqueeze(0) # [1,1,H,W]
    return img, mask


# ======================= #
# 模型预测函数
# ======================= #
def predict_image(model, img_tensor, device='cuda'):
    """
    使用训练好的模型预测单张图像
    输入: img_tensor [1,1,H,W]
    输出: seg_mask [H,W], cls_preds [num_discs,num_classes]
    """
    model.eval()
    img_tensor = img_tensor.to(device)
    with torch.no_grad():
        seg_pred, cls_preds_list = model(img_tensor, gt_mask=None, use_gt_mask=False)

    seg_mask = seg_pred[0, 0].cpu().numpy()
    cls_preds = cls_preds_list[0].cpu().numpy()
    return seg_mask, cls_preds


if __name__ == "__main__":
    import torch
    import matplotlib.pyplot as plt
    import numpy as np

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ================= 初始化模型 =================
    model = End2EndModel(in_channels=1, num_classes=3, base=32, roi_margin=0.1)
    model.to(device)

    # 加载训练好的权重
    model.load_state_dict(torch.load(
        "./checkpoints/checkpoints_class/best_model_20251113_190931_acc.pth",
        map_location=device
    ))

    # ================= 读取图像与mask =================
    img_tensor, mask_tensor = load_image_mask(
        "./dataset/Spider/images/2_t2_8.png",
        "./dataset/Spider/masks/2_t2_8.png",
        img_size=(256, 256)
    )

    # ================= 预测 =================
    model.eval()
    with torch.no_grad():
        # ---- 确保输入 shape 正确 ----
        if img_tensor.ndim == 3:  # [C,H,W] -> [1,C,H,W]
            img_tensor = img_tensor.unsqueeze(0).to(device)
        else:                     # [B,C,H,W]
            img_tensor = img_tensor.to(device)

        seg_mask, cls_preds = model(img_tensor, use_gt_mask=False)

        # ---- 分割 mask ----
        seg_mask = seg_mask[0, 0].cpu().numpy()              # [H,W]
        seg_mask_bin = (seg_mask > 0.5).astype(np.uint8)    # 二值化 mask 0/1

        # ---- 分类 ----
        cls_probs = []
        cls_labels = []
        for p in cls_preds:   # p: Tensor [num_rois, num_classes]
            if p.numel() == 0:
                cls_probs.append(np.array([]))
                cls_labels.append(np.array([]))
                continue
            p = p.to(device)
            prob = torch.sigmoid(p)             # logits -> 概率
            label = (prob > 0.5).int()          # 概率 -> 0/1
            cls_probs.append(prob.cpu().numpy())
            cls_labels.append(label.cpu().numpy())

    # ================= 计算分割指标 =================
    seg_pred_tensor = torch.tensor(seg_mask).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    dice, iou, f1, hd95 = compute_seg_metrics(seg_pred_tensor, mask_tensor)

    print(f"\nSegmentation Metrics:")
    print(f"  Dice  = {dice:.4f}")
    print(f"  IoU   = {iou:.4f}")
    print(f"  F1    = {f1:.4f}")
    print(f"  HD95  = {hd95:.4f}")

    # ================= 可视化分割结果 =================
    plt.figure(figsize=(6, 6))
    plt.imshow(seg_mask_bin*255, cmap='gray')
    plt.title("Predicted Segmentation Mask (Binary)")
    plt.axis('off')
    plt.show()

    # ================= 打印分类结果 =================
    print("\nPredicted classification per disc:")
    for i, (label_row, prob_row) in enumerate(zip(cls_labels, cls_probs)):
        for j in range(label_row.shape[0]):
            tasks_label = label_row[j]
            tasks_prob  = prob_row[j]
            print(f"Disc {j+1}: Task1={tasks_label[0]}, Task2={tasks_label[1]}, Task3={tasks_label[2]} | "
                  f"Prob={tasks_prob.tolist()}")



