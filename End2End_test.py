import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from DataSet import LumbarSagittalDataset
from End2End_train import custom_collate
from network.End2End8 import End2EndModel
from compute_metric import compute_seg_metrics, compute_cls_metrics, compute_cls_metrics_consistent
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def compute_binary_confusion_stats(all_preds, all_labels, threshold=0.5):
    """
    统计三个二分类任务的 TP / FP / FN / TN
    """
    stats = [
        {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0}
        for _ in range(3)
    ]

    for preds, labels in zip(all_preds, all_labels):
        n = min(preds.shape[0], labels.shape[0])
        preds = preds[:n]
        labels = labels[:n]

        preds_bin = (preds >= threshold).astype(int)

        for c in range(3):
            y_pred = preds_bin[:, c]
            y_true = labels[:, c]

            stats[c]['TP'] += int(((y_pred == 1) & (y_true == 1)).sum())
            stats[c]['FP'] += int(((y_pred == 1) & (y_true == 0)).sum())
            stats[c]['FN'] += int(((y_pred == 0) & (y_true == 1)).sum())
            stats[c]['TN'] += int(((y_pred == 0) & (y_true == 0)).sum())

    return stats


def collect_binary_tasks(all_preds, all_labels):
    """
    将 ROI-level 的预测与标签整理为 task-level（二分类）

    return:
        y_true_list: list of length=3, each [N]
        y_score_list: list of length=3, each [N]
    """
    y_true_list = [[], [], []]
    y_score_list = [[], [], []]

    for preds, labels in zip(all_preds, all_labels):
        n = min(preds.shape[0], labels.shape[0])
        preds = preds[:n]
        labels = labels[:n]

        for c in range(3):
            y_true_list[c].extend(labels[:, c].tolist())
            y_score_list[c].extend(preds[:, c].tolist())

    y_true_list = [np.array(y) for y in y_true_list]
    y_score_list = [np.array(y) for y in y_score_list]

    return y_true_list, y_score_list

import os

def plot_roc_curves(y_true_list, y_score_list, task_names=None, save_dir='./plot'):
    if task_names is None:
        task_names = ['Task-1', 'Task-2', 'Task-3']

    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(6, 6))

    for i in range(3):
        y_true = y_true_list[i]
        y_score = y_score_list[i]

        if len(np.unique(y_true)) < 2:
            print(f"Skip ROC for {task_names[i]} (only one class)")
            continue

        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)

        plt.plot(
            fpr, tpr, lw=2,
            label=f'{task_names[i]} (AUC={roc_auc:.3f})'
        )

    plt.plot([0, 1], [0, 1], linestyle='--', lw=1)

    # ===== 字体大小设置 =====
    plt.xlabel('False Positive Rate', fontsize=18)
    plt.ylabel('True Positive Rate', fontsize=18)
    plt.title('Receiver Operating Characteristic(ROC) Curves', fontsize=16)

    plt.legend(loc='lower right', fontsize=16)
    plt.grid(alpha=0.3)

    # 坐标刻度字体
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.tight_layout()

    save_path = os.path.join(save_dir, 'roc.png')
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"ROC curve saved to: {save_path}")


# =======================
# 测试函数（与验证相同，但不需要 loss ）
# =======================
def test_epoch(model, dataloader, device):
    model.eval()
    cls_error_samples = []  # 存储分类错误的样本名
    all_preds, all_labels = [], []
    seg_preds, seg_gts = [], []
    roi_mismatch_samples = []  # 存储 ROI mismatch 样本名称

    with torch.no_grad():
        for imgs, targets in tqdm(dataloader, desc="Test"):
            imgs = imgs.to(device)
            seg_gt = torch.stack([t['mask'] for t in targets], dim=0).to(device)

            # 测试时永远不使用 GT mask
            seg_pred, cls_pred_list = model(imgs, gt_mask=seg_gt, use_gt_mask=False)

            seg_preds.append(seg_pred.cpu())
            seg_gts.append(seg_gt.cpu())

            # 分类结果收集
            for b in range(len(targets)):
                labels_np = targets[b]['disc_labels'].cpu().numpy()
                preds_np = torch.sigmoid(cls_pred_list[b]).cpu().numpy()
                all_preds.append(preds_np)
                all_labels.append(labels_np)

                file_name = targets[b]['file_name']

                # -------- ROI mismatch --------
                if preds_np.shape[0] != labels_np.shape[0]:
                    roi_mismatch_samples.append(file_name)
                    continue  # 数量不一致时不再做分类错误判断

                # -------- 分类错误判断 --------
                preds_bin = (preds_np >= 0.5).astype(int)

                # 只要有一个 ROI、一个 task 预测错，就记为错误样本
                if not np.array_equal(preds_bin, labels_np):
                    cls_error_samples.append(file_name)

            # for b in range(len(targets)):
            #     labels_np = targets[b]['disc_labels'].cpu().numpy()
            #     preds_np = torch.sigmoid(cls_pred_list[b]).cpu().numpy()
            #     all_preds.append(preds_np)
            #     all_labels.append(labels_np)
            #
            #     # 检查 ROI 数量是否匹配
            #     if preds_np.shape[0] != labels_np.shape[0]:
            #         roi_mismatch_samples.append(targets[b]['file_name'])

    # 分割指标
    seg_pred_all = torch.cat(seg_preds)
    seg_gt_all = torch.cat(seg_gts)
    # 分割指标
    dice, dice_std, iou, iou_std, f1_seg, f1_seg_std, dh95, dh95_std = compute_seg_metrics(seg_pred_all, seg_gt_all)

    # 分类指标
    acc_cls, prec_cls, f1_cls, auc_cls, count_acc, acc_std, prec_std, f1_cls_std, auc_std = compute_cls_metrics_consistent(
        all_preds, all_labels)

    return (
        dice, iou, f1_seg, dh95, dice_std, iou_std, f1_seg_std, dh95_std,
        acc_cls, prec_cls, f1_cls, auc_cls, count_acc, acc_std, prec_std, f1_cls_std, auc_std,
        roi_mismatch_samples,
        cls_error_samples,
        all_preds, all_labels
    )


# =======================
# Test 主函数
# =======================
def test_main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载 checkpoint（你可以改成自己的路径）
    ckpt_path = "./checkpoints/checkpoints_seg/best_model_20260117_014147_dice.pth"

    model = End2EndModel(in_channels=1, num_classes=3, base=32).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    print(f"Loaded model: {ckpt_path}")

    # Test dataset
    test_image_dir = "./dataset/Spider/test/images"
    test_mask_dir  = "./dataset/Spider/test/masks"
    test_label_dir = "./dataset/Spider/test/labels"

    test_dataset = LumbarSagittalDataset(
        test_image_dir,
        test_mask_dir,
        test_label_dir,
        target_size=(256, 256)
    )

    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False,
        num_workers=4, collate_fn=custom_collate
    )

    # ======== 计算测试集指标 ========
    (
        dice, iou, f1_seg, dh95, dice_std, iou_std, f1_seg_std, dh95_std,
        acc_cls, prec_cls, f1_cls, auc_cls, count_acc, acc_std, prec_std, f1_cls_std, auc_std,
        roi_mismatch_samples,
        cls_error_samples,
        all_preds, all_labels
    ) = test_epoch(model, test_loader, device)

    print(f"  ROI mismatch = {count_acc:.0f}")
    if len(roi_mismatch_samples) > 0:
        print("  ROI mismatch samples:")
        for name in roi_mismatch_samples:
            print(f"    {name}")

    print("\n===== Classification Error Samples =====")
    print(f"Total classification error samples: {len(set(cls_error_samples))}")

    for name in sorted(set(cls_error_samples)):
        print(f"  {name}")

    print("\n========== Test Metrics ==========")
    print(f"Segmentation:")
    print(f"  Dice = {dice:.4f} ± {dice_std:.4f}")
    print(f"  IoU  = {iou:.4f} ± {iou_std:.4f}")
    print(f"  F1   = {f1_seg:.4f} ± {f1_seg_std:.4f}")
    print(f"  HD95 = {dh95:.4f} ± {dh95_std:.4f}")

    print(f"\nClassification:")
    print(f"  Acc  = {acc_cls:.4f} ± {acc_std:.4f}")
    print(f"  Prec = {prec_cls:.4f} ± {prec_std:.4f}")
    print(f"  F1   = {f1_cls:.4f} ± {f1_cls_std:.4f}")
    print(f"  AUC  = {auc_cls:.4f} ± {auc_std:.4f}")
    print(f"  ROI mismatch = {count_acc:.0f}")

    print("==================================")

    # ===== TP / FP / FN / TN =====
    confusion_stats = compute_binary_confusion_stats(all_preds, all_labels)

    task_names = [
        'herniation',
        'narrowing',
        'bulging'
    ]

    print("\n===== Binary Confusion Statistics =====")
    for name, stat in zip(task_names, confusion_stats):
        print(
            f"{name}: "
            f"TP={stat['TP']}, "
            f"FP={stat['FP']}, "
            f"FN={stat['FN']}, "
            f"TN={stat['TN']}"
        )
        y_true_list, y_score_list = collect_binary_tasks(all_preds, all_labels)

        plot_roc_curves(y_true_list, y_score_list, task_names)


if __name__ == "__main__":

    # 测试
    test_main()



