import datetime
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import logging
from sklearn.model_selection import train_test_split
from compute_metric import compute_seg_metrics, compute_cls_metrics
from network.End2End7 import End2EndModel
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

    # 训练开始时间
    startTime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # 设置日志
    log_dir = "logs"
    setup_logger(log_dir, startTime)
    logging.info(f"Training start at {startTime}")

    image_dir = "dataset/Spider/train/images"
    mask_dir = "dataset/Spider/train/masks"
    label_excel = "dataset/Spider/train/labels"

    full_dataset = LumbarSagittalDataset(image_dir, mask_dir, label_excel, target_size=(256, 256))
    train_idx, val_idx = train_test_split(list(range(len(full_dataset))), test_size=0.2, random_state=42, shuffle=True)
    train_dataset = torch.utils.data.Subset(full_dataset, train_idx)
    val_dataset = torch.utils.data.Subset(full_dataset, val_idx)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4, collate_fn=custom_collate)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4, collate_fn=custom_collate)

    model = End2EndModel(in_channels=1, num_classes=3, base=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    best_acc = 0.0
    best_dice = 0.0

    # 保存目录
    cls_save_dir = "checkpoints/checkpoints_class"
    seg_save_dir = "checkpoints/checkpoints_seg"
    os.makedirs(cls_save_dir, exist_ok=True)
    os.makedirs(seg_save_dir, exist_ok=True)

    num_epochs = 50
    for epoch in range(num_epochs):
        logging.info(f"\nEpoch {epoch+1}/{num_epochs}")
        use_gt = epoch < 10
        # use_gt = True

        train_loss, dice, iou, f1_seg, dh95, acc, prec, f1_cls, auc_cls, count_matches = train_epoch(model, train_loader, optimizer, device, use_gt_mask=use_gt)
        val_loss, val_dice, val_iou, val_f1_seg, val_dh95, val_acc, val_prec, val_f1_cls, val_auc_cls, val_count_matches = validate_epoch(model, val_loader, device)

        logging.info(f"[Train] Loss={train_loss:.4f} | Seg(Dice={dice:.4f}, IoU={iou:.4f}, F1={f1_seg:.4f}, DH={dh95:.4f}) | Cls(Acc={acc:.4f}, Prec={prec:.4f}, F1={f1_cls:.4f}, AUC={auc_cls:.4f}, ROI error ={count_matches:.1f})")
        logging.info(f"[Val]   Loss={val_loss:.4f} | Seg(Dice={val_dice:.4f}, IoU={val_iou:.4f}, F1={val_f1_seg:.4f}, DH={val_dh95:.4f}) | Cls(Acc={val_acc:.4f}, Prec={val_prec:.4f}, F1={val_f1_cls:.4f}, AUC={val_auc_cls:.4f}), ROI error ={val_count_matches:.1f})")


        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(cls_save_dir, f"best_model_{startTime}_acc.pth"))
            logging.info(f"Classification model saved with ACC={best_acc:.3f}")

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), os.path.join(seg_save_dir, f"best_model_{startTime}_dice.pth"))
            logging.info(f"Segmentation model saved with DSC={best_dice:.3f}")

if __name__ == "__main__":
    main()
