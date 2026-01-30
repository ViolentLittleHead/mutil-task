import torch
from sklearn.metrics import accuracy_score, precision_score, f1_score, roc_auc_score
import numpy as np
from skimage import measure
from scipy.spatial.distance import directed_hausdorff

# =======================
# 分割指标计算
# =======================
def compute_seg_metrics(pred_mask, gt_mask, threshold=0.5):
    """
    计算分割指标：Dice, IoU, F1, HD95
    返回平均值和标准差
    """
    if isinstance(pred_mask, torch.Tensor):
        pred_mask = pred_mask.detach().cpu().numpy()
    if isinstance(gt_mask, torch.Tensor):
        gt_mask = gt_mask.detach().cpu().numpy()

    pred_bin = (pred_mask > threshold).astype(np.float32)
    gt_bin = (gt_mask > 0.5).astype(np.float32)

    batch_dice, batch_iou, batch_f1, batch_hd95 = [], [], [], []

    for b in range(pred_bin.shape[0]):
        p = pred_bin[b, 0].flatten()
        g = gt_bin[b, 0].flatten()

        # Dice
        inter = (p * g).sum()
        dice = (2 * inter + 1e-5) / (p.sum() + g.sum() + 1e-5)
        batch_dice.append(dice)

        # IoU
        iou = (inter + 1e-5) / ((p + g - p * g).sum() + 1e-5)
        batch_iou.append(iou)

        # F1
        f1_score_val = 2 * dice * iou / (dice + iou + 1e-5)
        batch_f1.append(f1_score_val)

        # HD95
        def get_surface_points(mask_2d):
            contours = measure.find_contours(mask_2d, 0.5)
            if len(contours) == 0:
                return np.zeros((0, 2))
            return np.concatenate(contours, axis=0)

        pred_points = get_surface_points(pred_bin[b, 0])
        gt_points = get_surface_points(gt_bin[b, 0])

        if len(pred_points) == 0 and len(gt_points) == 0:
            hd95 = 0.0
        elif len(pred_points) == 0 or len(gt_points) == 0:
            h, w = pred_bin.shape[2], pred_bin.shape[3]
            hd95 = np.sqrt(h**2 + w**2)
        else:
            hd_forward = directed_hausdorff(pred_points, gt_points)[0]
            hd_backward = directed_hausdorff(gt_points, pred_points)[0]
            hd95 = max(hd_forward, hd_backward)

        batch_hd95.append(hd95)

    # 平均 + 标准差
    return (
        np.mean(batch_dice), np.std(batch_dice),
        np.mean(batch_iou), np.std(batch_iou),
        np.mean(batch_f1), np.std(batch_f1),
        np.mean(batch_hd95), np.std(batch_hd95)
    )


# =======================
# 分类指标计算函数
# =======================
def compute_cls_metrics(pred_list, label_list):
    accs, precs, f1s, aucs = [], [], [], []
    count_matches = 0

    for preds, labels in zip(pred_list, label_list):
        if preds.shape[0] != labels.shape[0]:
            count_matches += 1
        if preds.shape[0] == 0 or labels.shape[0] == 0:
            continue

        n = min(preds.shape[0], labels.shape[0])
        preds_cut = preds[:n]
        labels_cut = labels[:n]

        preds_bin = (preds_cut > 0.5).astype(int)

        accs.append(accuracy_score(labels_cut.flatten(), preds_bin.flatten()))
        precs.append(precision_score(labels_cut.flatten(), preds_bin.flatten(), zero_division=0))
        f1s.append(f1_score(labels_cut.flatten(), preds_bin.flatten(), average='macro', zero_division=0))

        auc_list = []
        for c in range(preds_cut.shape[1]):
            y_true = labels_cut[:, c]
            y_score = preds_cut[:, c]
            if len(np.unique(y_true)) < 2:
                continue
            try:
                auc_list.append(roc_auc_score(y_true, y_score))
            except ValueError:
                continue
        if len(auc_list) > 0:
            aucs.append(np.mean(auc_list))

    if len(accs) == 0:
        return (0.0, 0.0, 0.0, 0.0, count_matches, 0.0, 0.0, 0.0, 0.0)

    return (
        np.mean(accs), np.mean(precs), np.mean(f1s), np.mean(aucs),
        count_matches,
        np.std(accs), np.std(precs), np.std(f1s), np.std(aucs)
    )

def compute_cls_metrics_consistent(pred_list, label_list):
    accs, precs, f1s, aucs = [], [], [], []

    for preds, labels in zip(pred_list, label_list):
        n = min(preds.shape[0], labels.shape[0])
        preds = preds[:n]
        labels = labels[:n]

        preds_bin = (preds > 0.5).astype(int)

        acc_c, prec_c, f1_c, auc_c = [], [], [], []

        for c in range(preds.shape[1]):
            y_true = labels[:, c]
            y_pred = preds_bin[:, c]
            y_score = preds[:, c]

            acc_c.append(accuracy_score(y_true, y_pred))
            prec_c.append(precision_score(y_true, y_pred, zero_division=0))
            f1_c.append(f1_score(y_true, y_pred, zero_division=0))

            if len(np.unique(y_true)) > 1:
                auc_c.append(roc_auc_score(y_true, y_score))

        accs.append(np.mean(acc_c))
        precs.append(np.mean(prec_c))
        f1s.append(np.mean(f1_c))
        aucs.append(np.mean(auc_c))

    return (
        np.mean(accs), np.mean(precs), np.mean(f1s), np.mean(aucs),
        np.std(accs), np.std(precs), np.std(f1s), np.std(aucs)
    )


