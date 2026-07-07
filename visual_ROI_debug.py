# extract_rois_debug.py
import numpy as np
import cv2
import matplotlib.pyplot as plt
from DataSet import LumbarSagittalDataset

def visualize_process(steps, titles):
    """将每一个步骤单独展示为一张图"""
    for img, title in zip(steps, titles):
        plt.figure(figsize=(6, 6))  # 每张图更大
        if img.ndim == 2:
            plt.imshow(img, cmap="gray")
        else:
            plt.imshow(img)
        plt.title(title)
        plt.axis("off")
        plt.show()



def extract_rois_from_mask_debug(
        mask_tensor,
        roi_margin=0.3,
        min_height_expand=6,
        min_weight_expand=6,
        min_area=30,
        repair_kernel=7,
        visualize=False
):
    """
    新增功能：
        除了流程步骤图外，额外展示：
        👉 1. 无闭运算、无膨胀情况下的 ROI 提取结果（用于对比改进方法效果）
        👉 2. 改进方法 ROI（已有）
    """
    mask_np = mask_tensor.squeeze().cpu().numpy().astype(np.uint8)
    mask_bin = (mask_np > 0.5).astype(np.uint8) * 255

    steps = []
    titles = []

    # ================================
    # ★ 0. 原始 mask
    # ================================
    steps.append(mask_bin.copy())
    titles.append("raw mask (resized)")

    # ============================================================
    # ★ A. 不做闭运算、不做膨胀，直接连通域 + ROI（用于对比）
    # ============================================================
    H, W = mask_bin.shape
    num_labels_raw, labels_raw = cv2.connectedComponents(mask_bin)

    raw_rois = []
    for i in range(1, num_labels_raw):
        ys, xs = np.where(labels_raw == i)
        if len(xs) == 0:
            continue

        x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
        w, h = x2 - x1, y2 - y1

        if w * h < min_area:
            continue

        margin_x = max(int(w * roi_margin), min_weight_expand)
        margin_y = max(int(h * roi_margin), min_height_expand)

        raw_rois.append([
            max(x1 - margin_x, 0),
            max(y1 - margin_y, 0),
            min(x2 + margin_x, W - 1),
            min(y2 + margin_y, H - 1)
        ])

    # 排序
    raw_rois.sort(key=lambda r: r[1])

    # ▶ 可视化 raw ROI
    raw_roi_vis = cv2.cvtColor(mask_bin.copy(), cv2.COLOR_GRAY2BGR)
    for (x1, y1, x2, y2) in raw_rois:
        cv2.rectangle(raw_roi_vis, (x1, y1), (x2, y2), (255, 0, 0), 1)  # 黄色框
    steps.append(raw_roi_vis)
    titles.append("Raw ROI (no dilation / no closing)")

    # ============================================================
    # ★ B. 改进方法（先横向膨胀，再闭运算，稳定 ROI）
    # ============================================================
    mask_fix = mask_bin.copy()

    # --- 横向膨胀 ---
    if repair_kernel > 0:
        kernel_h = 2
        kernel_w = 7
        kernel = np.ones((kernel_h, kernel_w), np.uint8)
        mask_fix = cv2.dilate(mask_fix, kernel, iterations=1)
        steps.append(mask_fix.copy())
        titles.append("horizontal dilation")

        # --- 闭运算修复 ---
        kernel_close = np.ones((3, 3), np.uint8)
        mask_fix = cv2.morphologyEx(mask_fix, cv2.MORPH_CLOSE, kernel_close)
        steps.append(mask_fix.copy())
        titles.append("closing operation")

    # --- 连通域 ---
    num_labels, labels = cv2.connectedComponents(mask_fix)
    step_conn = np.zeros_like(mask_fix)
    step_conn[labels > 0] = 255
    steps.append(step_conn)
    titles.append("connected components")

    # --- ROI 提取 ---
    rois = []
    for i in range(1, num_labels):
        ys, xs = np.where(labels == i)
        if len(xs) == 0:
            continue

        x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
        w, h = x2 - x1, y2 - y1

        if w * h < min_area:
            continue

        margin_x = max(int(w * roi_margin), min_weight_expand)
        margin_y = max(int(h * roi_margin), min_height_expand)

        rois.append([
            max(x1 - margin_x, 0),
            max(y1 - margin_y, 0),
            min(x2 + margin_x, W - 1),
            min(y2 + margin_y, H - 1)
        ])

    rois.sort(key=lambda r: r[1])

    # --- ROI 可视化 ---
    roi_vis = cv2.cvtColor(mask_fix.copy(), cv2.COLOR_GRAY2BGR)
    for (x1, y1, x2, y2) in rois:
        cv2.rectangle(roi_vis, (x1, y1), (x2, y2), (255, 0, 0), 1)
    steps.append(roi_vis)
    titles.append("Improved ROI (dilation + closing)")

    # ============================================================
    # ★ 若 visualize=True，则逐张展示
    # ============================================================
    if visualize:
        visualize_process(steps, titles)

    return rois, steps, titles, raw_rois



if __name__ == "__main__":
    import albumentations as A

    # --------------------------
    # 1) Albumentations resize
    # --------------------------
    transform = A.Resize(256, 256)

    # --------------------------
    # 2) 加载你的数据集
    # --------------------------
    dataset = LumbarSagittalDataset(
        image_dir="./dataset/Spider/train/images",
        mask_dir="./dataset/Spider/train/masks",
        label_dir="./dataset/Spider/train/labels",
        transform=transform
    )

    print("样本数:", len(dataset))

    # 指定一个样本
    idx = 253
    img, target = dataset[idx]
    mask = target["mask"]

    print("处理样本 filename:", target["file_name"])
    print("mask shape:", mask.shape)
    print("disc labels:", target["disc_labels"])

    # --------------------------
    # 3) 调用改进版 ROI 提取（含膨胀 + 闭运算）
    # --------------------------
    print("\n===== 改进版 ROI（含修复）=====")
    rois_adv, steps_adv, titles_adv = extract_rois_from_mask_debug(
        mask,
        roi_margin=0.3,
        min_height_expand=6,
        min_weight_expand=6,
        min_area=30,
        repair_kernel=5,   # 启用修复
        visualize=True
    )
    print("最终 ROI 数量:", len(rois_adv))
    print("ROI 坐标：", rois_adv)


