'''
Author: Howard
Date: 2025-01-07 17:20:36
LastEditors: warmheartHoward 1366194556@qq.com
LastEditTime: 2025-01-15 14:43:37
FilePath: \QT\calibration_window.py
Description: 

Copyright (c) 2025 by ${git_name_email}, All Rights Reserved. 
'''
import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel, QLineEdit,
    QTextEdit, QVBoxLayout, QHBoxLayout, QFileDialog, QGroupBox,
    QGridLayout, QDialog, QFormLayout, QDialogButtonBox, QSpacerItem, QSizePolicy
)
from PyQt5.QtWidgets import QApplication, QMainWindow
from pyqtgraph import ErrorBarItem

from PyQt5.QtCore import Qt
from base_window import BaseWindow
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtGui import QMovie, QFont
import cv2
import pyqtgraph as pg
from utils.calibration_methods import calibration_process
import os
import numpy as np
# from PyQt5.QtWidgets import QGraphicsLineItem

 
# 自定义的 FigureCanvas 类
class CustomPlotWidget(pg.PlotWidget):
    def __init__(self, parent=None, title=""):
        super().__init__(parent)
        self.setBackground('w')  # 设置背景为白色
        if title:
            # 使用 plotItem 的 setTitle 方法设置标题
            self.plotItem.setTitle(title, color='black', size='12pt')
        self.showGrid(x=True, y=True)
        self.addLegend()

class ProcessingThread(QThread):
    result_ready = pyqtSignal(object)

    def __init__(self, algorithm):
        super().__init__()
        self.algorithm = algorithm

    def run(self):
        # 在这里调用 self.algorithm 来执行特定的算法
        result = self.algorithm()
        self.result_ready.emit(result)

class CalibrationWindow(BaseWindow):
    """Calibration window"""
    def __init__(self):
        super().__init__("系统标定")
        self.initial_settings = {
            'excel_file_path': None,
            'image_data_col': "F",
            'balance_data_col': "J"
        }
        self.excel_file_path = self.initial_settings['excel_file_path']
        self.image_data_col = self.initial_settings['image_data_col']
        self.balance_data_col = self.initial_settings['balance_data_col']
        # self.add_content(self.layout)
        # self.excel_file_path = None
        # self.image_data_col = "F"
        # self.balance_data_col = "J"
        
        
    def add_content(self, layout):
        # 主布局
        self.dpi = 100
        main_layout = QHBoxLayout()
        # 左侧布局
        left_layout = QVBoxLayout()
        # 测量模型标定区域
        calibration_group = QGroupBox("测量模型线性标定")
        calibration_group.setStyleSheet("""
        QGroupBox {
            border: 2px solid gray;
            border-radius: 5px;
            margin-top: 10px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top center;                          
            color: black;
        }
        """)
        # 设置整个QGroupBox的字体
        font = calibration_group.font()
        font.setPointSize(12)  # 设置字体大小
        font.setBold(True)     # 字体加粗
        calibration_group.setFont(font)
        calibration_group.setFixedSize(400, 400)
     

        excel_select_btn = QPushButton("原始数据Excel文件")
        excel_select_btn.clicked.connect(self.select_file)
        excel_select_btn.setStyleSheet("""
            QPushButton {
                font-size: 14px; 
                padding: 8px; 
                border-radius: 5px; 
                background-color: green; 
                color: white;
                border: none;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)

        calibration_layout = QVBoxLayout()
        calibration_layout.addWidget(excel_select_btn)

        # 图像特征所在Excel的列索引
        image_feature_layout = QHBoxLayout()
        image_feature_label = QLabel("图像特征计算Excel列索引:")
        image_feature_label.setStyleSheet("""
            QLabel {
                background: transparent;
                color: black;
                font-size: 12px;
                border-radius: 2px; 
                border: none;
            }
        """)

        self.image_feature_input = QLineEdit(self)
        self.image_feature_input.setText("F")
        self.image_feature_input.textChanged.connect(self.on_return_pressed_image)
        
        image_feature_layout.addWidget(image_feature_label)
        image_feature_layout.addWidget(self.image_feature_input)
        calibration_layout.addLayout(image_feature_layout)

        # 天平数据所在Excel的列索引
        balance_data_layout = QHBoxLayout()
        balance_data_label = QLabel("天平称量结果Excel列索引:")
        balance_data_label.setStyleSheet("""
            QLabel {
                background: transparent;
                color: black;
                font-size: 12px;
                border-radius: 2px; 
                border: none;
            }
        """)
        self.balance_data_input = QLineEdit(self)
        self.balance_data_input.setText("J")
        self.balance_data_input.textChanged.connect(self.on_return_pressed_balance)
        balance_data_layout.addWidget(balance_data_label)
        balance_data_layout.addWidget(self.balance_data_input)
        calibration_layout.addLayout(balance_data_layout)

        # 一键标定按钮
        self.calibration_button = QPushButton("一键标定")
        self.calibration_button.clicked.connect(lambda: self.start_task(self.calibration_algorithm, self.calibration_result_callback))
        self.calibration_button.setStyleSheet("""
            QPushButton {
                font-size: 14px; 
                padding: 8px; 
                border-radius: 5px; 
                background-color: #2196F3; 
                color: white;
                border: none;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        calibration_layout.addWidget(self.calibration_button)

        calibration_group.setLayout(calibration_layout)
        left_layout.addWidget(calibration_group)
        ###########################################################################################
        # 单应矩阵计算区域
        homography_group = QGroupBox("单应矩阵计算")
        homography_group.setStyleSheet("""
        QGroupBox {
            border: 2px solid gray;
            border-radius: 5px;
            margin-top: 10px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top center;                          
            color: black;
        }
        """)
        # 设置整个QGroupBox的字体
        font = homography_group.font()
        font.setPointSize(14)  # 设置字体大小
        font.setBold(True)     # 字体加粗
        homography_group.setFont(font)
        
        homography_layout = QVBoxLayout()
        homography_select_layout = QHBoxLayout()
        # 图像选择按钮
        self.image_select_button = QPushButton("图像选择")
        self.image_select_button.setStyleSheet("""
            QPushButton {
                font-size: 14px; 
                padding: 8px; 
                border-radius: 5px; 
                background-color: #FF5722; 
                color: white;
                border: none;
            }
            QPushButton:hover {
                background-color: #e64a19;
            }
        """)
        self.image_select_button.clicked.connect(self.select_image)
        homography_select_layout.addWidget(self.image_select_button)

        # 图像畸变参数输入按钮
        self.distortion_params_button = QPushButton("图像畸变参数手动输入")
        self.distortion_params_button.setStyleSheet("""
            QPushButton {
                font-size: 14px; 
                padding: 8px; 
                border-radius: 5px; 
                background-color: #9C27B0; 
                color: white;
                border: none;
            }
            QPushButton:hover {
                background-color: #7B1FA2;
            }
        """)
        self.distortion_params_button.clicked.connect(self.input_distortion_params)
        homography_select_layout.addWidget(self.distortion_params_button)
        # 一键计算按钮
        self.compute_button = QPushButton("一键计算")
        self.compute_button.setStyleSheet("""
            QPushButton {
                font-size: 16px; 
                padding: 10px; 
                border-radius: 20px; 
                background-color: #2196F3; 
                color: white;
                border: none;
            }
            QPushButton:hover {
                background-color: #1e88e5;
            }
        """)

        homography_layout.addLayout(homography_select_layout)
        homography_layout.addWidget(self.compute_button)
        homography_group.setLayout(homography_layout)
        left_layout.addWidget(homography_group)
        ##################################################################################
        # 输出文本
        self.text_display = QTextEdit()
        self.text_display.setFixedSize(400, 150)
        self.text_display.setReadOnly(True)
        self.text_display.setPlaceholderText("输出文本显示区域")
        self.text_display.setStyleSheet("""
            font-size: 14px; 
            border: 2px #9E9E9E; 
            border-radius: 5px;
        """)

        left_layout.addWidget(self.text_display)
        ######################################################################################

        # 右侧布局
        right_layout = QVBoxLayout()

        # 标定结果图像显示区域
        # Create a Matplotlib canvas
        # sc = MplCanvas(self, width=2, height=2, dpi=100)


        self.calibration_image_layout = QHBoxLayout()
        self.left_image = QLabel("线性拟合结果图像显示区域")
        self.left_image.setAlignment(Qt.AlignCenter)
        self.left_image.setStyleSheet("""
            QLabel {
                background: #333;
                color: #fff;
                font-size: 20px;
                border: 2px solid #555;
                min-width: 300px;
                min-height: 300px;
            }
        """)
        self.right_image = QLabel("拟合结果分组误差条图")
        self.right_image.setAlignment(Qt.AlignCenter)
        self.right_image.setStyleSheet("""
            QLabel {
                background: #333;
                color: #fff;
                font-size: 20px;
                border: 2px solid #555;
                min-width: 300px;
                min-height: 300px;
            }
        """)
            # 创建两个 FigureCanvas，并添加到布局中
        # 使用 PyQtGraph 的 PlotWidget 代替 Matplotlib Canvas
        self.left_plot = CustomPlotWidget(title="线性拟合结果图像显示区域")
        self.right_plot = CustomPlotWidget(title="拟合结果分组误差条图")
        self.calibration_image_layout.addWidget(self.left_plot)
        self.calibration_image_layout.addWidget(self.right_plot)
        

        # 单应矩阵计算示意图
        # image_display = QHBoxLayout()
        self.image_display = QLabel("标记点检测图像")
        # self.image_display.setFixedSize(800, 300)
        self.image_display.setAlignment(Qt.AlignCenter)
        self.image_display.setStyleSheet("""
            QLabel {
                background: #333;
                color: #fff;
                font-size: 20px;
                border: 2px solid #555;
                min-width: 400px;
                min-height: 350px;
            }
        """)
        # self.calibration_image_display = QLabel("标定板图像示例")
        # self.calibration_image_display.setAlignment(Qt.AlignCenter)
        # self.calibration_image_display.setStyleSheet("""
        #     QLabel {
        #         background: #333;
        #         color: #fff;
        #         font-size: 20px;
        #         border: 2px solid #555;
        #         min-width: 400px;
        #         min-height: 350px;
        #     }
        # """)
        # image_display.addWidget(self.image_display)
        # image_display.addWidget()

        



        ###################################################################################################################
        # 右下角加载动画
        self.otherfunction = QHBoxLayout()
        self.statusLabel = QLabel(self)
        self.statusLabel.setFixedSize(50, 50)  # 增大控件尺寸
        self.statusLabel.setStyleSheet("background: transparent;")  # 添加边框便于调试
        self.statusLabel.raise_()  # 确保在最上层
        self.movie = QMovie("image/loading.gif")
        self.movie.setScaledSize(self.statusLabel.size())  # 缩放GIF尺寸
        spacer1 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        # 一键清除功能
        self.clearall = QPushButton("一键清除")
        self.clearall.clicked.connect(self.clearAll)
        self.clearall.setFixedSize(120, 30)
        self.clearall.setStyleSheet("""
                QPushButton {
                    background: #4CAF50;
                    color: white;
                    font-size: 13px;
                    border: none;
                    border-radius: 4px;
                    min-width: 200px;
                }
                QPushButton:hover {
                    background: #45a049;
                }
        """)

        self.otherfunction.addStretch(4)
        self.otherfunction.addWidget(self.clearall)
        self.otherfunction.addStretch(3)
        self.otherfunction.addWidget(self.statusLabel)


        right_layout.addLayout(self.calibration_image_layout)

        right_layout.addWidget(self.image_display)
        right_layout.addLayout(self.otherfunction)


        # right_layout.addWidget(self.text_display)

        # 将左侧和右侧布局添加到主布局
        main_layout.addLayout(left_layout, 1)
        main_layout.addLayout(right_layout, 2)
        layout.addLayout(main_layout)

    def select_file(self):
        # File dialog for selecting Excel files
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "选择Excel文件", "", "Excel Files (*.xls *.xlsx);;All Files (*)", options=options)
        self.excel_file_path = file_name
        if file_name:
            print(f"Selected file: {file_name}")
            self.text_display.append(f"选择的Excel路径: {file_name}")

    def on_return_pressed_image(self):
        # 更新对象中的变量
        self.image_data_col = self.image_feature_input.text()
        # 打印以验证
        print(f"Input value: {self.image_data_col}")
        # 锁定QLineEdit，使其无法输入
        self.image_feature_input.setReadOnly(True)

    def on_return_pressed_balance(self):
        # 更新对象中的变量
        self.balance_data_col = self.balance_data_input.text()
        # 打印以验证
        print(f"Input value: {self.balance_data_col}")
        # 锁定QLineEdit，使其无法输入
        self.balance_data_input.setReadOnly(True)

    def select_image(self):
        """选择图像并显示"""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图像", "", "图像文件 (*.png *.jpg *.bmp);;所有文件 (*)", options=options
        )
        if file_path:
            pixmap = QPixmap(file_path)
            pixmap = pixmap.scaled(self.image_display.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.image_display.setPixmap(pixmap)
            self.text_display.append(f"选择的图像路径: {file_path}")

    def input_distortion_params(self):
        """弹出窗口输入畸变参数"""
        dialog = DistortionParamsDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            params = dialog.get_params()
            self.text_display.append(f"输入的畸变参数: {params}")

    def cv2pixmap(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # 获取图像的维度
        height, width, channel = image_rgb.shape
        bytes_per_line = 3 * width
        # 创建 QImage，然后转换为 QPixmap
        qimage = QImage(image_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)
        return pixmap
    
    def start_task(self, algorithm, result_callback):
        self.statusLabel.setMovie(self.movie)
        self.movie.start()
        
        self.thread = ProcessingThread(algorithm)
        self.thread.result_ready.connect(lambda result:self.on_result_ready(result, result_callback))
        self.thread.start()

        # 结束movie的动画
    def on_result_ready(self, result, result_callback):
        self.movie.stop()
        self.statusLabel.clear()
        result_callback(result)
        


    def calibration_algorithm(self):
        if self.excel_file_path != None:
            model_str, x2, real_weight, y_fit, bin_centers, bin_means, bin_errors, R2 = calibration_process(self.excel_file_path, self.image_data_col, self.balance_data_col, "H")
            return [model_str, x2, y_fit, real_weight, bin_centers, bin_means, bin_errors, R2]

            
        else:
            self.text_display.clear()
            self.text_display.append("请先打开采集的excel数据文件")
            return None


    def calibration_result_callback(self, result):
        
        model_str, x2, y_fit, real_weight, bin_centers, bin_means, bin_errors, R2 = result
        self.text_display.clear()
        self.text_display.append(model_str)

        # 清除之前的绘图
        self.left_plot.clear()
        self.right_plot.clear()
        # 确保数据是numpy数组
        x2 = np.array(x2)
        y_fit = np.array(y_fit)
        real_weight = np.array(real_weight)
        bin_centers = np.array(bin_centers)
        bin_means = np.array(bin_means)
        bin_errors = np.array(bin_errors)
   
        # 绘制线性拟合图
        self.left_plot.plot(x2, y_fit, pen=pg.mkPen(color='r', width=2), name='Calibration')
        # 绘制实际数据点
        scatter = pg.ScatterPlotItem(
            x=x2,
            y=real_weight,
            pen=pg.mkPen(color='b'),
            brush=pg.mkBrush(color='b'),
            symbol='o',
            size=5,
            name='Real Volume'
        )
        self.left_plot.addItem(scatter)
        # 添加 R² 文本
        text_item = pg.TextItem(f"R² = {R2:.3f}", anchor=(0,1), color='r')
        if len(x2) > 0 and len(y_fit) > 0:
            x_max = np.max(x2)
            y_max = np.max(y_fit)
            text_item.setPos(x_max * 0.3, y_max * 0.6)
        self.left_plot.addItem(text_item)
        self.left_plot.setLabel('left', "Real Volume (mL)")
        self.left_plot.setLabel('bottom', "Liquid Segment Length in the Pixel Coordinate System (pixels)")
        self.left_plot.setTitle("Linear Fitting Results of the Calibration Measurement Method",  size="9pt")


        # 绘制误差条图，使用 ErrorBarItem
        error_bars = ErrorBarItem(
            x=bin_centers,
            y=bin_means,
            height=bin_errors,
            beam=0.01,  # 设置“帽子”的宽度
            pen=pg.mkPen(color='g', width =1)
        )
        self.right_plot.addItem(error_bars)

        # 绘制测量数据点（散点，不连接线）
        measurement_scatter = pg.ScatterPlotItem(
            x=bin_centers,
            y=bin_means,
            pen=pg.mkPen(color='b'),
            brush=pg.mkBrush(color='b'),
            symbol='o',
            size=4,
            name='Measurement'
        )
        self.right_plot.addItem(measurement_scatter)

        # 绘制真实值参考线（连续线条）
        if len(bin_centers) > 0:
            x_min = np.min(bin_centers)
            x_max = np.max(bin_centers)
            self.right_plot.plot(
                [x_min, x_max],
                [x_min, x_max],
                pen=pg.mkPen(color='r', style=Qt.DashLine, width=2),
                name='True Value'
            )
        # 添加图例
        self.right_plot.setLabel('left', "Calibration-Based Volume Measurement/mL")
        self.right_plot.setLabel('bottom', "Real Volume/mL")
        self.right_plot.setTitle('Error Bar Chart Of the Calibration Measurement Method', size="9pt")
        # 锁定纵横比为 1:1
        self.right_plot.setAspectLocked(True, ratio=1.0)

        # 添加图例
        self.left_plot.addLegend()
        self.right_plot.addLegend()


        # 设置 PlotWidget 的大小策略为扩展，以自适应布局
        self.left_plot.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.right_plot.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # 自动调整坐标轴范围
        self.left_plot.enableAutoRange()
        self.right_plot.enableAutoRange()

    def clearAll(self):
        """清空所有图表内容，重置标签和变量到初始化状态。"""
        # 1. 清空图表
        self.left_plot.clear()
        self.right_plot.clear()
        # 2. 重置标题
        self.left_plot.setTitle("线性拟合结果图像显示区域", color="black", size="14pt")
        self.right_plot.setTitle("拟合结果分组误差条图", color="black", size="14pt")
        # 5. 清空文本显示区域
        self.text_display.clear()
        # 6. 重置输入字段
        self.image_feature_input.setText(self.initial_settings['image_data_col'])
        self.balance_data_input.setText(self.initial_settings['balance_data_col'])
        self.image_feature_input.setReadOnly(False)  # 允许重新输入
        self.balance_data_input.setReadOnly(False)  # 允许重新输入
        # 7. 重置相关变量
        self.excel_file_path = self.initial_settings['excel_file_path']
        self.image_data_col = self.initial_settings['image_data_col']
        self.balance_data_col = self.initial_settings['balance_data_col']








        


class DistortionParamsDialog(QDialog):
    """畸变参数输入对话框"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("输入畸变参数")
        self.setModal(True)
        self.init_ui()

    def init_ui(self):
        layout = QFormLayout()

        self.k1_input = QLineEdit()
        self.k2_input = QLineEdit()
        self.p1_input = QLineEdit()
        self.p2_input = QLineEdit()
        self.k3_input = QLineEdit()

        layout.addRow("k1:", self.k1_input)
        layout.addRow("k2:", self.k2_input)
        layout.addRow("p1:", self.p1_input)
        layout.addRow("p2:", self.p2_input)
        layout.addRow("k3:", self.k3_input)

        # 确认和取消按钮
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.setLayout(layout)

    def get_params(self):
        """获取输入的参数"""
        return {
            "k1": self.k1_input.text(),
            "k2": self.k2_input.text(),
            "p1": self.p1_input.text(),
            "p2": self.p2_input.text(),
            "k3": self.k3_input.text(),
        }