# -*- coding: utf-8 -*-
import sys
import os
import torch
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog, QMessageBox
# ===================== 1. 模型加载与推理函数 =====================
class AnimalRecognition:
    def __init__(self, model_path="AnimalModel.pt"):
        # 检查模型文件是否存在
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ 未找到模型文件：{model_path}，请先训练模型！")

        # 加载模型和映射
        self.checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
        self.num_classes = self.checkpoint["num_classes"]
        self.animal_map = self.checkpoint["animal_map"]
        self.idx2animal = {v: k for k, v in self.animal_map.items()}  # 数字→动物名

        # 加载模型结构（和训练时一致）
        self.model = self._build_model()
        self.model.load_state_dict(self.checkpoint["model_state_dict"])
        self.model.eval()  # 推理模式

        # 图片预处理（和训练时完全一致）
        self.transform = Compose([
            Resize([100, 100], antialias=True),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _build_model(self):
        """构建和训练时一致的CNN模型"""
        from torch import nn
        class AnimalModel(nn.Module):
            def __init__(self, num_classes=90):
                super().__init__()
                self.backbone = nn.Sequential(
                    nn.Conv2d(3, 10, 5),
                    nn.BatchNorm2d(10),
                    nn.ReLU(),
                    nn.MaxPool2d(2, 2),
                    nn.Conv2d(10, 20, 5),
                    nn.BatchNorm2d(20),
                    nn.ReLU(),
                    nn.MaxPool2d(2, 2),
                    nn.Conv2d(20, 30, 5),
                    nn.BatchNorm2d(30),
                    nn.ReLU(),
                    nn.MaxPool2d(2, 2),
                    nn.Dropout(p=0.5),
                    nn.Flatten(),
                    nn.Linear(2430, 300),
                    nn.ReLU(),
                    nn.Linear(300, 150),
                    nn.ReLU(),
                    nn.Linear(150, 80),
                    nn.ReLU(),
                    nn.Linear(80, num_classes)
                )

            def forward(self, x):
                return self.backbone(x)

        return AnimalModel(num_classes=self.num_classes)

    def predict(self, img_path):
        """
        输入图片路径，返回识别结果
        :return: 动物名, 置信度(%)
        """
        try:
            # 读取并预处理图片
            img = Image.open(img_path).convert("RGB")
            img_tensor = self.transform(img).unsqueeze(0)  # 加batch维度

            # 推理（禁用梯度，加速）
            with torch.no_grad():
                outputs = self.model(img_tensor)
                prob = torch.softmax(outputs, dim=1)  # 转概率分布
                pred_idx = torch.argmax(prob, dim=1).item()  # 预测类别索引
                pred_prob = prob[0][pred_idx].item()  # 置信度

            # 转换为动物名
            pred_animal = self.idx2animal[pred_idx]
            return pred_animal, round(pred_prob * 100, 2)

        except Exception as e:
            raise RuntimeError(f"❌ 图片识别失败：{str(e)}")


# ===================== 2. PyQt5界面类 =====================
class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        # 窗口基础设置
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(600, 550)
        MainWindow.setMinimumSize(QtCore.QSize(600, 550))
        MainWindow.setWindowTitle("动物识别系统")

        # 中央控件
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        # 选择图片按钮
        self.chooseImgBtn = QtWidgets.QPushButton(self.centralwidget)
        self.chooseImgBtn.setGeometry(QtCore.QRect(180, 40, 240, 60))
        self.chooseImgBtn.setStyleSheet("font-size: 14px; font-weight: bold;")
        self.chooseImgBtn.setText("选择待识别的动物图片")

        # 图片显示标签
        self.imgLab = QtWidgets.QLabel(self.centralwidget)
        self.imgLab.setGeometry(QtCore.QRect(80, 120, 440, 280))
        self.imgLab.setAlignment(QtCore.Qt.AlignCenter)
        self.imgLab.setStyleSheet("border: 1px solid #cccccc;")
        self.imgLab.setText("请选择图片")

        # 结果提示标签
        self.lab = QtWidgets.QLabel(self.centralwidget)
        self.lab.setGeometry(QtCore.QRect(100, 430, 180, 50))
        self.lab.setStyleSheet("font-size: 14px; color: #333333;")
        self.lab.setText("动物识别结果：")

        # 结果显示标签（突出显示）
        self.resLab = QtWidgets.QLabel(self.centralwidget)
        self.resLab.setGeometry(QtCore.QRect(280, 430, 200, 50))
        self.resLab.setStyleSheet("font-size: 16px; color: #e64340; font-weight: bold;")
        self.resLab.setText("未识别")

        # 置信度标签
        self.probLab = QtWidgets.QLabel(self.centralwidget)
        self.probLab.setGeometry(QtCore.QRect(100, 480, 400, 30))
        self.probLab.setStyleSheet("font-size: 12px; color: #666666;")
        self.probLab.setText("置信度：--")

        MainWindow.setCentralWidget(self.centralwidget)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)


# ===================== 3. 界面交互逻辑 =====================
class AnimalRecoApp(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # 初始化识别模型（加载训练好的模型）
        try:
            self.recognizer = AnimalRecognition(model_path="AnimalModel.pt")
            QMessageBox.information(self, "成功", "✅ 动物识别模型加载完成！")
        except Exception as e:
            QMessageBox.critical(self, "错误", str(e))
            sys.exit(1)

        # 绑定按钮点击事件
        self.ui.chooseImgBtn.clicked.connect(self.choose_image)

    def choose_image(self):
        """选择图片并识别"""
        # 打开文件选择对话框
        img_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择动物图片",
            "",
            "图片文件 (*.jpg *.jpeg *.png *.bmp)"
        )

        if not img_path:  # 未选择图片
            return

        try:
            # 显示选中的图片（按比例缩放）
            pixmap = QtGui.QPixmap(img_path)
            pixmap = pixmap.scaled(self.ui.imgLab.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
            self.ui.imgLab.setPixmap(pixmap)

            # 调用识别函数
            animal_name, prob = self.recognizer.predict(img_path)

            # 显示结果
            self.ui.resLab.setText(animal_name)
            self.ui.probLab.setText(f"置信度：{prob}%")

        except Exception as e:
            QMessageBox.warning(self, "识别失败", f"❌ {str(e)}")
            self.ui.resLab.setText("识别失败")
            self.ui.probLab.setText("置信度：--")


# ===================== 4. 主函数 =====================
if __name__ == "__main__":
    # 解决PyQt5中文显示问题
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    app = QtWidgets.QApplication(sys.argv)

    # 设置中文字体（避免乱码）
    font = QtGui.QFont()
    font.setFamily("SimHei")  # 黑体
    app.setFont(font)

    # 启动界面
    window = AnimalRecoApp()
    window.show()
    sys.exit(app.exec_())