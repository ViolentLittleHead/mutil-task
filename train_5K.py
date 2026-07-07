import datetime
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import logging
from sklearn.model_selection import train_test_split, KFold
from compute_metric import compute_seg_metrics, compute_cls_metrics
from network.End2End8 import End2EndModel
from DataSet import LumbarSagittalDataset  # 你的数据集类

# =======================
# 日志配置函数
# =======================
def setup_logger(log_dir, startTime):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"train_{startTime}.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler()  # 同时输出到控制台
        ]
    )

# =======================
# 自定义 collate_fn
# =======================
def custom_collate(batch):
    imgs = torch.stack([item[0] for item in batch], dim=0)
    targets = [item[1] for item in batch]
    return imgs, targets

# =======================
# Dice Loss
# =======================
def dice_loss(pred, target, smooth=1e-5):
    pred = torch.clamp(pred, 1e-7, 1-1e-7)
    intersection = (pred * target).sum()
    dice = (2 * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return 1 - dice

# =======================
# Multi-task loss
# =======================
def multitask_loss(seg_pred, seg_gt, cls_pred_list, targets, alpha=0.5, beta=1.0, gamma=1.0):
    """
    seg_pred: [B,1,H,W] 分割预测
    seg_gt:   [B,1,H,W] 分割GT
    cls_pred_list: list 长度为 B，每个元素 [num_rois, num_classes]
    targets:  list 长度为 B，每个元素 dict，包含 'disc_labels' [num_discs, num_classes]
    """
    device = seg_pred.device
    seg_loss = dice_loss(seg_pred, seg_gt)  # 返回 tensor (标量)

    batch_cls_loss = torch.tensor(0.0, device=device)  # 在 device 上累加分类损失
    roi_count_loss_batch = torch.tensor(0.0, device=device)  # 在 device 上累加ROI损失
    preds_all, labels_all = [], []

    B = len(targets)
    for b in range(B):
        disc_labels = targets[b]['disc_labels'].to(device)  # [num_discs, num_classes]
        cls_pred = cls_pred_list[b]  # [num_rois, num_classes] (logits)

        # --------------------------
        # (新增) ROI Count Loss
        # --------------------------
        N_pred = cls_pred.shape[0]
        N_gt   = disc_labels.shape[0]
        roi_count_loss_batch += torch.abs(
            torch.tensor(N_pred - N_gt, dtype=torch.float32, device=device)
        )

        if cls_pred.numel() == 0:
            cls_loss = torch.tensor(0.0, device=device)
        else:
            n = min(cls_pred.shape[0], disc_labels.shape[0])
            cls_pred_cut = cls_pred[:n, :]  # logits
            disc_labels_cut = disc_labels[:n, :].float()

            # - 使用 BCEWithLogitsLoss（内部包含 sigmoid，更稳定）
            # - batch_cls_loss 以 tensor 在 device 上累加，不使用 .item()
            # - 对没有 ROI 的样本用 0 tensor 占位
            # - 返回时把用于 logging 的数值用 .item()
            bce_logits_loss_fn = nn.BCEWithLogitsLoss(reduction='mean')
            cls_loss = bce_logits_loss_fn(cls_pred_cut, disc_labels_cut)

            # 保存用于指标计算（把 logits 转成概率再阈值）
            with torch.no_grad():
                probs = torch.sigmoid(cls_pred).detach().cpu().numpy()
                preds_all.append(probs)
                labels_all.append(disc_labels.detach().cpu().numpy())

        batch_cls_loss = batch_cls_loss + cls_loss

    avg_cls_loss = batch_cls_loss / float(B)
    avg_cnt_loss = roi_count_loss_batch / float(B)
    total_loss = alpha * seg_loss + beta * avg_cls_loss + gamma * avg_cnt_loss

    # 返回：loss（tensor，可直接 backward），以及用于 logging 的数值
    return total_loss, seg_loss.item(), avg_cls_loss.item(), avg_cnt_loss.item(), preds_all, labels_all


# =======================
# 训练函数
# =======================
def train_epoch(model, dataloader, optimizer, device, use_gt_mask=True):
    model.train()
    total_loss, total_seg, total_cls = 0,0,0
    all_preds, all_labels = [], []
    seg_preds, seg_gts = [], []

    for imgs, targets in tqdm(dataloader, desc=f"Train(use_gt_mask={use_gt_mask})"):
        imgs = imgs.to(device)
        seg_gt = torch.stack([t['mask'] for t in targets], dim=0).to(device)

        optimizer.zero_grad()
        seg_pred, cls_pred_list = model(imgs, gt_mask=seg_gt, use_gt_mask=use_gt_mask)

        loss, seg_l, cls_l, roi_l, preds_list, labels_list = multitask_loss(seg_pred, seg_gt, cls_pred_list, targets)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_seg += seg_l
        total_cls += cls_l

        seg_preds.append(seg_pred.detach().cpu())
        seg_gts.append(seg_gt.cpu())
        all_preds.extend(preds_list)
        all_labels.extend(labels_list)

    seg_pred_all = torch.cat(seg_preds)
    seg_gt_all = torch.cat(seg_gts)
    dice, iou, f1_seg, dh95 = compute_seg_metrics(seg_pred_all, seg_gt_all)
    acc, prec, f1_cls, auc_cls, count_acc = compute_cls_metrics(all_preds, all_labels)

    return total_loss/len(dataloader), dice, iou, f1_seg, dh95, acc, prec, f1_cls, auc_cls, count_acc

# =======================
# 验证函数
# =======================
def validate_epoch(model, dataloader, device):
    model.eval()
    total_loss, total_seg, total_cls = 0,0,0
    all_preds, all_labels = [], []
    seg_preds, seg_gts = [], []

    with torch.no_grad():
        for imgs, targets in tqdm(dataloader, desc="Validation"):
            imgs = imgs.to(device)
            seg_gt = torch.stack([t['mask'] for t in targets], dim=0).to(device)

            seg_pred, cls_pred_list = model(imgs, gt_mask=seg_gt, use_gt_mask=False)
            loss, seg_l, cls_l, cls_l, preds_list, labels_list = multitask_loss(seg_pred, seg_gt, cls_pred_list, targets)

            total_loss += loss.item()
            total_seg += seg_l
            total_cls += cls_l

            seg_preds.append(seg_pred.detach().cpu())
            seg_gts.append(seg_gt.cpu())
            all_preds.extend(preds_list)
            all_labels.extend(labels_list)

    seg_pred_all = torch.cat(seg_preds)
    seg_gt_all = torch.cat(seg_gts)
    dice, iou, f1_seg, dh95 = compute_seg_metrics(seg_pred_all, seg_gt_all)
    acc, prec, f1_cls, auc_cls, count_acc = compute_cls_metrics(all_preds, all_labels)

    return total_loss/len(dataloader), dice, iou, f1_seg, dh95, acc, prec, f1_cls, auc_cls, count_acc

# =======================
# 主函数
# =======================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    startTime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # ======================= #
    # 日志 & 保存根目录
    # ======================= #
    log_root = f"logs/SuZhou/{startTime}"
    ckpt_root = f"checkpoints/SuZhou/{startTime}"
    os.makedirs(log_root, exist_ok=True)
    os.makedirs(ckpt_root, exist_ok=True)

    setup_logger(log_root, startTime)
    logging.info(f"5-Fold Training start at {startTime}")

    image_dir = "../SuZhou-End2End/images"
    mask_dir  = "../SuZhou-End2End/masks"
    label_excel = "../SuZhou-End2End/labels"

    full_dataset = LumbarSagittalDataset(
        image_dir, mask_dir, label_excel, target_size=(256, 256)
    )

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    num_epochs = 100
    batch_size = 4
    pretrained = True
    fold_results = []

    # ======================= #
    # 5-Fold
    # ======================= #
    for fold, (train_idx, val_idx) in enumerate(kfold.split(range(len(full_dataset)))):

        fold_id = fold + 1
        logging.info("=" * 90)
        logging.info(f"Fold {fold_id}/5")
        logging.info("=" * 90)

        train_dataset = torch.utils.data.Subset(full_dataset, train_idx)
        val_dataset   = torch.utils.data.Subset(full_dataset, val_idx)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=custom_collate
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=custom_collate
        )

        model = End2EndModel(in_channels=1, num_classes=3, base=32).to(device)
        if pretrained :
            ckpt_path = "./checkpoints/checkpoints_seg/best_model_20251219_115709_dice.pth"
            model.load_state_dict(torch.load(ckpt_path))
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        # 保存目录
        fold_ckpt_dir = f"{ckpt_root}/fold{fold_id}"
        os.makedirs(fold_ckpt_dir, exist_ok=True)

        best_metrics = {
            "dice": -1,
            "acc": -1,
            "iou": 0,
            "f1_seg": 0,
            "dh95": 999,
            "prec": 0,
            "f1_cls": 0,
            "auc": 0
        }

        # ======================= #
        # Epoch loop
        # ======================= #
        for epoch in range(num_epochs):
            logging.info(f"[Fold {fold_id}] Epoch {epoch+1}/{num_epochs}")

            use_gt = epoch < 10

            train_loss, *_ = train_epoch(
                model, train_loader, optimizer, device, use_gt_mask=use_gt
            )

            val_loss, dice, iou, f1_seg, dh95, acc, prec, f1_cls, auc, roi_err = \
                validate_epoch(model, val_loader, device)

            logging.info(
                f"[Val] Loss={val_loss:.4f} | "
                f"Seg(Dice={dice:.4f}, IoU={iou:.4f}, F1={f1_seg:.4f}, DH={dh95:.4f}) | "
                f"Cls(Acc={acc:.4f}, Prec={prec:.4f}, F1={f1_cls:.4f}, AUC={auc:.4f}, ROI err={roi_err:.1f})"
            )

            # -------- 分类最优 --------
            if acc > best_metrics["acc"]:
                best_metrics.update({
                    "acc": acc,
                    "prec": prec,
                    "f1_cls": f1_cls,
                    "auc": auc
                })
                torch.save(model.state_dict(), f"{fold_ckpt_dir}/best_acc.pth")

            # -------- 分割最优 --------
            if dice > best_metrics["dice"]:
                best_metrics.update({
                    "dice": dice,
                    "iou": iou,
                    "f1_seg": f1_seg,
                    "dh95": dh95
                })
                torch.save(model.state_dict(), f"{fold_ckpt_dir}/best_dice.pth")

        best_metrics["fold"] = fold_id
        fold_results.append(best_metrics)

    # ======================= #
    # 5-Fold 汇总
    # ======================= #
    logging.info("=" * 90)
    logging.info("5-Fold Cross Validation Summary")

    def avg(key):
        return sum(f[key] for f in fold_results) / 5

    for f in fold_results:
        logging.info(
            f"Fold {f['fold']} | "
            f"Seg(Dice={f['dice']:.4f}, IoU={f['iou']:.4f}, F1={f['f1_seg']:.4f}, DH={f['dh95']:.4f}) | "
            f"Cls(Acc={f['acc']:.4f}, Prec={f['prec']:.4f}, F1={f['f1_cls']:.4f}, AUC={f['auc']:.4f})"
        )

    logging.info("-" * 90)
    logging.info(
        f"AVG | "
        f"Seg(Dice={avg('dice'):.4f}, IoU={avg('iou'):.4f}, "
        f"F1={avg('f1_seg'):.4f}, DH={avg('dh95'):.4f}) | "
        f"Cls(Acc={avg('acc'):.4f}, Prec={avg('prec'):.4f}, "
        f"F1={avg('f1_cls'):.4f}, AUC={avg('auc'):.4f})"
    )


if __name__ == "__main__":
    main()
