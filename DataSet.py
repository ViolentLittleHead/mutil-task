import os
import torch
import numpy as np
import json
import glob
import cv2
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor
from PIL import Image
import matplotlib.pyplot as plt


class LumbarSagittalDataset(Dataset):
    def __init__(self, image_dir, mask_dir, label_dir,
                 transform=None, target_size=(256, 256)):
        """
        image_dir: 原图目录
        mask_dir: mask 目录
        label_dir: txt/json 标签目录
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.label_dir = label_dir
        self.transform = transform
        self.target_size = target_size

        # 所有 png 文件
        self.image_files = sorted(glob.glob(os.path.join(image_dir, "*.png")))

        # 保证文件名一致（不包含路径，仅名字）
        self.file_names = [os.path.basename(f).replace(".png", "") for f in self.image_files]

    def __len__(self):
        return len(self.file_names)

    def _load_label(self, label_path):
        """
        label txt/json 格式如下：
        {
            "1": {"Disc herniation":0, ... },
            "2": {...},
            ...
        }
        """
        with open(label_path, "r") as f:
            data = json.load(f)

        # 按 key 从小到大排序（"1","2"..."6"）
        labels_ordered = []
        for key in sorted(data.keys(), key=lambda x: int(x)):
            row = data[key]
            labels_ordered.append([
                row["Disc herniation"],
                row["Disc narrowing"],
                row["Disc bulging"],
            ])

        # 转 tensor
        labels = torch.tensor(labels_ordered, dtype=torch.float32)
        return labels

    def __getitem__(self, idx):
        file_name = self.file_names[idx]

        # 路径
        img_path = os.path.join(self.image_dir, file_name + ".png")
        mask_path = os.path.join(self.mask_dir, file_name + ".png")
        label_path = os.path.join(self.label_dir, file_name + ".txt")

        # 读取图像
        img = np.array(Image.open(img_path))
        mask = np.array(Image.open(mask_path))

        # resize
        img = cv2.resize(img, (self.target_size[1], self.target_size[0]), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (self.target_size[1], self.target_size[0]), interpolation=cv2.INTER_NEAREST)

        # 数据增强
        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img, mask = augmented['image'], augmented['mask']

        # 转 tensor
        img_tensor = to_tensor(img)
        mask_tensor = to_tensor(mask / 255.0)

        # load label
        labels = self._load_label(label_path)

        target = {
            "mask": mask_tensor,
            "disc_labels": labels,
            "file_name": file_name
        }

        return img_tensor, target

    def visualize(self, idx):
        img, target = self[idx]
        file_name = self.file_names[idx]

        img_np = img[0].numpy()
        mask = target["mask"][0].numpy()
        labels = target["disc_labels"].numpy()

        plt.figure(figsize=(6, 10))
        plt.imshow(img_np, cmap="gray")
        plt.imshow((mask > 0.5).astype(np.uint8), cmap="jet", alpha=0.3)

        # 从上到下显示
        text_lines = [
            f"{i+1}: {labels[i].astype(int)}"
            for i in range(labels.shape[0])
        ]
        text_str = "\n".join(text_lines)

        H, W = img_np.shape
        plt.text(W - 10, H - 10, text_str, color="red", fontsize=12,
                 verticalalignment="bottom", horizontalalignment="right")

        plt.title(file_name)
        plt.axis("off")
        plt.show()


# ================== 示例测试 ==================
if __name__ == "__main__":
    import albumentations as A
    transform = A.Resize(256, 256)

    dataset = LumbarSagittalDataset(
        image_dir="./dataset/Spider/images",
        mask_dir="./dataset/Spider/masks",
        label_dir="./dataset/Spider/labels",
        transform=transform
    )

    print("样本数:", len(dataset))
    img, target = dataset[50]
    print("图像尺寸:", img.shape)
    print("mask尺寸:", target["mask"].shape)
    print("disc_labels:", target["disc_labels"])

    dataset.visualize(50)
