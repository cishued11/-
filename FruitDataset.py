from torch.utils.data import DataLoader, Dataset
from PIL import Image
import json
import os
from pathlib import Path


# ===================== 生成/加载JSON标注文件 =====================
def generate_animal_json():
    ANIMAL_ROOT = r"D:\PythonProject4\动物识别\animals\animals"
    OUTPUT_JSON = r"D:\PythonProject4\动物识别\animal_labels.json"
    if os.path.exists(OUTPUT_JSON):
        print(f"✅ 检测到已存在{OUTPUT_JSON}，跳过生成")
        return

    labels = {}
    idx = 0
    for class_dir in os.scandir(ANIMAL_ROOT):
        if class_dir.is_dir():
            animal_name = class_dir.name
            for img_file in os.scandir(class_dir.path):
                if img_file.is_file() and Path(img_file.path).suffix.lower() in (".jpg", ".jpeg", ".png"):
                    labels[str(idx)] = {
                        "path": os.path.abspath(img_file.path),
                        "label": animal_name
                    }
                    idx += 1

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)
    print(f"✅ JSON生成完成：{OUTPUT_JSON}，共{len(labels)}条数据")


# 自动生成JSON（如果不存在）
generate_animal_json()

# ===================== 加载JSON并生成动物映射 =====================
labels = {}
try:
    with open(r"D:\PythonProject4\动物识别\animal_labels.json", "r", encoding="utf-8") as file:
        labels = json.load(file)
except Exception as e:
    raise RuntimeError(f"❌ 加载JSON失败：{e}")

# 生成动物标签→数字映射
animal_names = list(set([info["label"] for info in labels.values()]))
animal_names.sort()
animal_map = {name: idx for idx, name in enumerate(animal_names)}
print(f"✅ 动物类别映射：{animal_map}")


# ===================== 定义AnimalDS类 =====================
class AnimalDS(Dataset):
    def __init__(self, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform  # 保留参数但不使用（兼容接口）
        self.img_labels = labels
        self.animal_map = animal_map
        self.valid_labels = self._filter_valid_data()

    def _filter_valid_data(self):
        valid = {}
        for idx, info in self.img_labels.items():
            img_path = info["path"]
            if os.path.exists(img_path) and os.path.isfile(img_path):
                valid[idx] = info
            else:
                print(f"❌ 过滤无效数据：{idx} → {img_path}")
        return valid

    def __len__(self):
        return len(self.valid_labels)

    def __getitem__(self, item):
        idx = str(item)
        if idx not in self.valid_labels:
            raise IndexError(f"索引{item}超出范围（0~{len(self.valid_labels) - 1}）")

        info = self.valid_labels[idx]
        img_path = info["path"]
        raw_label = info["label"]  # 动物名字符串（如'gorilla'）

        # 读取并处理图片
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"读取图片失败：{img_path} → {e}")
        if self.transform is not None:
            img = self.transform(img)

        # 仅做一次：字符串→数字（核心！不再重复转换）
        label = self.animal_map[raw_label]  # gorilla → 31

        # 保留target_transform但注释（避免重复转换）
        # if self.target_transform is not None:
        #     label = self.target_transform(label)

        return img, label


# 测试（可选）
if __name__ == "__main__":
    dataset = AnimalDS()
    print(f"✅ 有效数据集数量：{len(dataset)}")
    if len(dataset) > 0:
        img, label = dataset[0]
        animal_name = [k for k, v in dataset.animal_map.items() if v == label][0]
        print(f"第一条数据：图片尺寸{img.size}，标签{label}（{animal_name}）")