import json
import os
from pathlib import Path

# ===================== 配置项 =====================
# 你的动物数据集根目录（对应字典中的path前缀）
ANIMAL_ROOT = "animals/animals"
# 生成的合法JSON标注文件保存路径
OUTPUT_JSON = "animal_labels.json"
# 支持的图片格式
ALLOWED_FORMATS = (".jpg", ".jpeg", ".png", ".bmp")


# ===================== 遍历文件夹生成标注 =====================
def generate_animal_labels(root_dir):
    """
    遍历动物文件夹，为每个图片生成标注信息
    返回：{索引: {"path": 图片绝对路径, "label": 具体动物名}}
    """
    labels = {}
    idx = 0  # 全局索引

    # 遍历每个动物类别文件夹
    for class_dir in os.scandir(root_dir):
        if not class_dir.is_dir():
            continue
        animal_name = class_dir.name  # 具体动物名（如antelope、badger）
        print(f"正在处理：{animal_name}")

        # 遍历该类别下的所有图片
        for img_file in os.scandir(class_dir.path):
            if (img_file.is_file() and
                    Path(img_file.path).suffix.lower() in ALLOWED_FORMATS):
                # 保存绝对路径（避免相对路径问题）
                abs_path = os.path.abspath(img_file.path)
                labels[str(idx)] = {
                    "path": abs_path,
                    "label": animal_name
                }
                idx += 1

    return labels


# 生成标注并保存为合法JSON
animal_labels = generate_animal_labels(ANIMAL_ROOT)
# 保存为JSON（ensure_ascii=False支持特殊字符，indent格式化）
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(animal_labels, f, ensure_ascii=False, indent=2)

print(f"\n✅ 已生成合法JSON标注文件：{OUTPUT_JSON}")
print(f"📊 共收集 {len(animal_labels)} 张动物图片")