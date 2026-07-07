import os

import torch
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
from network.End2End8_debugROI import End2EndModel

def imread_unicode(path, flags=cv2.IMREAD_COLOR):
    """
    支持中文路径的图像读取
    """
    # 使用 numpy 从文件读取二进制数据
    img_data = np.fromfile(path, dtype=np.uint8)
    # 使用 OpenCV 从内存中解码图像
    img = cv2.imdecode(img_data, flags)
    return img

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
IMG_SIZE = 256
THRESH = 0.5

CLASS_NAMES = [
    "Disc herniation",
    "Disc narrowing",
    "Disc bulging"
]

model = End2EndModel(
    in_channels=1,
    num_classes=3,
    min_area=10,
    morph_kernel_size=15,
    use_learnable_morph=True
).to(DEVICE)

ckpt_path = "checkpoints/checkpoints_seg/best_model_20251219_031945_dice.pth"   # ← 改成你的权重路径
state_dict = torch.load(ckpt_path, map_location=DEVICE)
model.load_state_dict(state_dict)
model.eval()

image_name = "50_t2_SPACE_61"
img_path = f"dataset/Spider/test/images/{image_name}.png"
mask_path = f"dataset/Spider/test/masks/{image_name}.png"

# 设置保存的目录（请替换为你的实际目录）
save_dir = r"dataset\results"

# 使用自定义函数读取
img_gray = imread_unicode(img_path, cv2.IMREAD_GRAYSCALE)

assert img_gray is not None, "Image not found!"

img_gray = cv2.resize(img_gray, (IMG_SIZE, IMG_SIZE))
img_norm = img_gray.astype(np.float32) / 255.0

# GT mask
gt_mask = imread_unicode(mask_path, cv2.IMREAD_GRAYSCALE)
assert gt_mask is not None, "GT Mask not found!"

gt_mask = cv2.resize(
    gt_mask,
    (IMG_SIZE, IMG_SIZE),
    interpolation=cv2.INTER_NEAREST
)

# [1,1,H,W]
x = torch.from_numpy(img_norm).unsqueeze(0).unsqueeze(0).to(DEVICE)

with torch.no_grad():
    seg_pred, cls_preds, roi_boxes = model(
        x,
        gt_mask=None,
        use_gt_mask=False,
        return_rois=True
    )

# segmentation mask
seg_mask = (seg_pred[0, 0].cpu().numpy() > THRESH).astype(np.uint8)

# classification probability
if cls_preds[0].shape[0] > 0:
    probs = torch.sigmoid(cls_preds[0]).cpu().numpy()
else:
    probs = np.zeros((0, len(CLASS_NAMES)))

vis = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)


for i, (x1, y1, x2, y2) in enumerate(roi_boxes[0]):
    # 只画 ROI 框（腰椎间盘）
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

    text_x = max(x1 - 55, 2)  # 向左偏移，防止出界
    text_y = int((y1 + y2) / 2)  # 垂直居中

    cv2.putText(
        vis,
        f"Disc {i + 1}",
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 0, 0),
        1,
        cv2.LINE_AA
    )

print("\n===== ROI Classification Results (Binary) =====")
for i, p in enumerate(probs):
    print(f"ROI {i + 1}:")
    for c, name in enumerate(CLASS_NAMES):
        binary_pred = 1 if p[c] >= THRESH else 0
        print(f"  {name:18s}: {binary_pred}")
    print("-" * 35)

os.makedirs(save_dir, exist_ok=True)

# 1. 保存 Input Image
plt.figure(figsize=(5, 5))
# plt.title("Input Image")
plt.imshow(img_gray, cmap='gray')
plt.axis('off')
# 添加 pad_inches=0 彻底去除白边
plt.savefig(os.path.join(save_dir, f"{image_name}.png"), dpi=150, bbox_inches='tight', pad_inches=0)
plt.close()  # 关闭当前画布，释放内存

# 2. 保存 GT Mask
plt.figure(figsize=(5, 5))
# plt.title("GT Mask")
plt.imshow(gt_mask, cmap='gray')
plt.axis('off')
plt.savefig(os.path.join(save_dir, f"{image_name}_mask.png"), dpi=150, bbox_inches='tight', pad_inches=0)
plt.close()

# 3. 保存 Predicted Mask
plt.figure(figsize=(5, 5))
# plt.title("Predicted Mask")
plt.imshow(seg_mask, cmap='gray')
plt.axis('off')
# 3. 保存 Predicted Mask
# 将 0/1 的掩码严格转换为 0/255 的 uint8 图像
# pred_mask_255 = (seg_mask * 255).astype(np.uint8)
# 直接使用 OpenCV 保存，确保像素值绝对为 0 和 255
# cv2.imwrite(os.path.join(save_dir, f"{image_name}_predicted.png"), pred_mask_255)
plt.close()

# 4. 保存 ROI Detection + Classification
plt.figure(figsize=(5, 5))
# plt.title("ROI Detection + Classification")
plt.imshow(vis)
plt.axis('off')
plt.savefig(os.path.join(save_dir, f"{image_name}_roi_detection.png"), dpi=150, bbox_inches='tight', pad_inches=0)
plt.close()
