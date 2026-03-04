# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torch.optim.lr_scheduler import StepLR
import os
import json


# ===================== 1. 定义ResNet18模型（无预训练权重，避免下载错误） =====================
class AnimalModel(nn.Module):
    def __init__(self, num_classes=90, device="cuda" if torch.cuda.is_available() else "cpu"):
        super(AnimalModel, self).__init__()
        self.num_classes = num_classes
        self.device = device

        # 核心修改：不用预训练权重（weights=None），避免下载/哈希错误
        self.backbone = models.resnet18(weights=None)
        # 适配90类动物，修改最后一层
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes)
        )
        # 不冻结层（从头训练，适配无预训练的情况）
        self.to(self.device)

    def forward(self, x):
        return self.backbone(x)


# ===================== 2. 数据集加载 =====================
from torch.utils.data import Dataset
from PIL import Image


class AnimalDS(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        # 标注文件路径
        self.label_path = r"D:\PythonProject4\动物识别\animal_labels.json"
        if not os.path.exists(self.label_path):
            raise FileNotFoundError(f"标注文件不存在：{self.label_path}")

        with open(self.label_path, "r", encoding="utf-8") as f:
            self.img_labels = json.load(f)

        # 生成动物映射
        animal_names = list(set([info["label"] for info in self.img_labels.values()]))
        animal_names.sort()
        self.animal_map = {name: idx for idx, name in enumerate(animal_names)}
        # 过滤有效图片
        self.valid_data = self._filter_valid()

    def _filter_valid(self):
        """过滤不存在/损坏的图片"""
        valid = []
        for info in self.img_labels.values():
            img_path = info["path"]
            if os.path.exists(img_path) and os.path.isfile(img_path):
                try:
                    Image.open(img_path).convert("RGB")
                    valid.append({"path": img_path, "label": info["label"]})
                except:
                    continue
        return valid

    def __len__(self):
        return len(self.valid_data)

    def __getitem__(self, idx):
        data = self.valid_data[idx]
        img_path = data["path"]
        label_str = data["label"]

        # 读取图片
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        # 标签转数字
        label = self.animal_map[label_str]
        return img, label


# ===================== 3. 训练配置（适配CPU+无预训练） =====================
# 超参数（无预训练需调高学习率，增加轮数）
EPOCH = 20  # CPU训练20轮足够
BATCH_SIZE = 16  # CPU专属批次大小
LEARNING_RATE = 0.001  # 无预训练需调高学习率（原0.0005→0.001）
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"训练设备：{DEVICE}")
print(f"⚠️ 无预训练权重，训练轮数20轮，学习率0.001")

# 数据预处理（增强保留，提升特征学习）
train_transform = transforms.Compose([
    transforms.Resize([100, 100], antialias=True),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载数据集（CPU专属配置）
dataset = AnimalDS(transform=train_transform)
train_loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,  # CPU必须设0
    pin_memory=False
)
print(f"✅ 数据集加载完成：{len(dataset)} 条数据，{len(dataset.animal_map)} 类动物")

# 初始化模型
num_classes = len(dataset.animal_map)
model = AnimalModel(num_classes=num_classes, device=DEVICE)

# 优化器（AdamW，适配无预训练）
optimizer = optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4
)

# 损失函数
criterion = nn.CrossEntropyLoss()

# 学习率调度器
scheduler = StepLR(optimizer, step_size=8, gamma=0.5)  # 每8轮学习率减半

# ===================== 4. 训练循环 =====================
print("🚀 开始训练（无预训练权重，CPU训练约2~3小时）...")
best_loss = float('inf')
for epoch in range(EPOCH):
    model.train()
    total_loss = 0.0
    for batch_idx, (imgs, labels) in enumerate(train_loader):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        # 前向传播
        outputs = model(imgs)
        loss = criterion(outputs, labels)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # 打印进度
        if batch_idx % 10 == 0:
            avg_loss = total_loss / (batch_idx + 1)
            print(f"Epoch [{epoch + 1}/{EPOCH}] | Batch [{batch_idx + 1}/{len(train_loader)}] | Loss: {avg_loss:.4f}")

    # 更新学习率
    scheduler.step()
    epoch_avg_loss = total_loss / len(train_loader)
    print(f"📌 Epoch {epoch + 1} 平均损失：{epoch_avg_loss:.4f} | 当前学习率：{scheduler.get_last_lr()[0]}")

    # 保存最优模型
    if epoch_avg_loss < best_loss:
        best_loss = epoch_avg_loss
        torch.save({
            "model_state_dict": model.state_dict(),
            "num_classes": num_classes,
            "animal_map": dataset.animal_map,
            "epoch": epoch + 1,
            "best_loss": best_loss
        }, "AnimalModel_Best.pt")
        print(f"✅ 保存最优模型（损失：{best_loss:.4f}）")

# 保存最终模型
torch.save({
    "model_state_dict": model.state_dict(),
    "num_classes": num_classes,
    "animal_map": dataset.animal_map,
    "final_loss": epoch_avg_loss
}, "AnimalModel_Final.pt")

print("\n🎉 训练完成！")
print(f"📁 最优模型：AnimalModel_Best.pt（损失：{best_loss:.4f}）")
print(f"📁 最终模型：AnimalModel_Final.pt")