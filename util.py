import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from skimage.segmentation import mark_boundaries
import torch.nn as nn
from torchvision.transforms.functional import to_pil_image
from metric import calculate_iou,calculate_f1

# 损失函数
"""
    计算切块损失值
    pred：一个张量 ，形状（batch，1，height，width），与预测相对应
    target：一个形状（1，h，w）的张量，对应于GT
"""
def dice_loss(pred,target,smooth = 1e-5):
    intersection = (pred * target).sum(dim=(3,2))
    union = pred.sum(dim=(2,3)) + target.sum(dim=(2,3))

    # dice 在[0,1]之间，其中值1表示预测和基准值之间的完美重叠。
    dice = 2.0 * (intersection + smooth) / (union + smooth)
    # 我们想要最小化这个loss
    loss = 1.0 - dice
    # 返回切块损失值
    return loss.sum(),dice.sum()


"""
    计算每个batch的合并损失
"""
def loss_func(pred_seg,pred_clas,target_seg,target_clas):
    # segmentation loss function
    # 网络输出的channels大于1
    if pred_seg.shape[1] > 1:
        celoss = torch.nn.CrossEntropyLoss()
        # target为[8, 1, 224, 224] 需要移除尺度1
        bce = celoss(pred_seg, np.squeeze(target_seg).long())
    else:
        # 二元交叉熵损失
        bce = F.binary_cross_entropy_with_logits(pred_seg,target_seg,reduction='mean')
    # 切块损失
    pred = torch.sigmoid(pred_seg)
    dlv,_ = dice_loss(pred,target_seg)

    # classification loss function
    criterion = nn.BCEWithLogitsLoss()
    clas_loss = criterion(pred_clas, target_clas)

    # 总损失
    loss = bce + dlv + clas_loss

    return loss

"""
    计算每个batch的度量。
"""
def metrics_batch(pred,target):
    pred = torch.sigmoid(pred)
    _,metric = dice_loss(pred,target)
    return metric

"""
    计算每个batch的损失和度量值dice。在训练时，优化器对象被传递给helper函数
    因此，使用opt.step() 更新参数模型
"""
def loss_batch(loss_func,output,target,opt=None):
    loss = loss_func(output,target)

    with torch.no_grad():
        pred = torch.sigmoid(output)
        _,metric_b = dice_loss(pred,target)
        meaniou = calculate_iou(pred,target)
        meanf1 = calculate_f1(pred, target)

    if opt is not None:
        opt.zero_grad()
        loss.backward()
        opt.step()

    return loss.item(),metric_b.item(),meaniou.item(),meanf1.item()

def show_img_mask(img,mask):
    if torch.is_tensor(img):
        img = to_pil_image(img)
        mask = to_pil_image(mask)
    # 将mask覆盖到图像上
    img_mask = mark_boundaries(np.array(img),np.array(mask),outline_color=(0,1,0),color=(0,1,0))
    plt.imshow(img_mask)

