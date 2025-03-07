'''
Author: Howard
Date: 2025-01-07 17:20:36
LastEditors: warmheartHoward 1366194556@qq.com
LastEditTime: 2025-01-15 22:41:08
FilePath: \QT\calibration_window.py
Description: 

Copyright (c) 2025 by ${git_name_email}, All Rights Reserved. 
'''
import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel, QLineEdit,
    QTextEdit, QVBoxLayout, QHBoxLayout, QFileDialog, QGroupBox,
    QGridLayout, QDialog, QFormLayout, QDialogButtonBox, QSpacerItem, QSizePolicy, QMessageBox
)
from PyQt5.QtWidgets import QApplication, QMainWindow

from PyQt5.QtCore import Qt
from base_window import BaseWindow
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtGui import QMovie
import cv2
import matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from matplotlib.figure import Figure
from utils.calibration_methods import calibration_process
from utils.image_processing import distort_image, homography_matrix_cal
import os
matplotlib.use('Agg')  # 使用非交互式后端
 
# 自定义的 FigureCanvas 类
# 自定义的 FigureCanvas 类
class MplCanvas(FigureCanvas):
    def __init__(self, fig=None, parent=None, width=4, height=4, dpi=100):
        if fig is None:
            self.figure = Figure(figsize=(width, height), dpi=dpi)
            self.figure = Figure()
        else:
            self.figure = fig
        super(MplCanvas, self).__init__(self.figure)

        # 设置 SizePolicy 以允许自动调整大小
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.updateGeometry()
        self.setParent(parent)
        self.ax = self.figure.add_subplot(111)
        self.figure.tight_layout()
        
    def update_with_figure(self, new_fig: Figure):
        self.figure.clf()
        self.figure = new_fig
        self.draw()

    def clear(self):
        self.figure.clf()
        self.ax = self.figure.add_subplot(111)
        self.draw()
        # self.figure.tight_layout()

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
        self.excel_file_path = None
        self.image_data_col = "F"
        self.balance_data_col = "J"
        self.image_path = None
        self.distribution = None
        self.distance = None
        
        
    def add_content(self, layout):
        # 主布局
        self.dpi = 125
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
        font.setPointSize(14)  # 设置字体大小
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
        self.calibration_button.clicked.connect(lambda: self.start_task(self.calibration_algorithm))
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
        self.distortion_params_button = QPushButton("图像畸变矫正")
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
        self.distortion_params_button.clicked.connect(lambda: self.start_task(self.distortion_image_algorithm))
        # self.distortion_params_button.clicked.connect(self.distortion_image_algorithm)
        homography_select_layout.addWidget(self.distortion_params_button)
        # 一键计算按钮
        self.compute_button = QPushButton("单应矩阵计算")
        self.compute_button.clicked.connect(self.open_input_dialog)
        self.compute_button.setStyleSheet("""
            QPushButton {
                font-size: 14px; 
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
        # 创建两个 FigureCanvas，并添加到布局中
        self.left_canvas = MplCanvas(width=4, height=4, dpi=self.dpi)
        self.left_canvas.setFixedWidth(400)
        self.left_canvas.setFixedHeight(400)
        self.right_canvas = MplCanvas(width=4, height=4, dpi=self.dpi)
        self.right_canvas.setFixedWidth(400)
        self.right_canvas.setFixedHeight(400)
        self.calibration_image_layout.addWidget(self.left_canvas)
        self.calibration_image_layout.addWidget(self.right_canvas)
        

        # 单应矩阵计算示意图
        self.image_display = QLabel("图像显示区域")
        # self.image_display.setFixedSize(800, 300)
        self.image_display.setAlignment(Qt.AlignCenter)
        self.image_display.setStyleSheet("""
            QLabel {
                background: #333;
                color: #fff;
                font-size: 20px;
                border: 2px solid #555;
                min-width: 800px;
                min-height: 350px;
            }
        """)

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
        self.clearall.clicked.connect(self.on_clear_all_call)
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
            self.image_path = file_path
            pixmap = QPixmap(file_path)
            pixmap = pixmap.scaled(self.image_display.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.image_display.setPixmap(pixmap)
            self.text_display.append(f"选择的图像路径: {file_path}")

    def distortion_image_algorithm(self):
        if self.image_path is not None:
            if self.intrinsic_parameter is not None and self.distortion_parameter is not None:
                image = cv2.imread(self.image_path)
                dist_img = distort_image(image, self.intrinsic_parameter, self.distortion_parameter)
                dist_img_p = self.cv2pixmap(dist_img)
                
                dist_img_p = dist_img_p.scaled(self.image_display.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.image_display.clear()
                self.image_display.setPixmap(dist_img_p)
            else:
                self.text_display.clear()
                self.text_display.append(f"请先到图像处理单元进行相机参数标定")

        else:
            self.text_display.append(f"请先打开带有标记点的图片")

            
    def homography_cal_algorithm(self):
        if self.image_path is None:
            self.text_display.clear()
            self.text_display.append(f"请先打开带有标记点的图片")
            return
        
        if self.distribution is not None and self.circle_distance is not None: 
            image = cv2.imread(self.image_path)
            H, dst_ = homography_matrix_cal(image, self.intrinsic_parameter, self.distortion_parameter, self.distribution, self.circle_distance)
            dst_p = self.cv2pixmap(dst_)
            dst_p = dst_p.scaled(self.image_display.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.image_display.clear()
            self.image_display.setPixmap(dst_p)
            self.text_display.append(f"图像坐标系与世界坐标系之间的单应矩阵为：{H}")
        else:
             self.text_display.append(f"每一进行标记点的输入")


    def cv2pixmap(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # 获取图像的维度
        height, width, channel = image_rgb.shape
        bytes_per_line = 3 * width
        # 创建 QImage，然后转换为 QPixmap
        qimage = QImage(image_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)
        return pixmap
    
    def start_task(self, algorithm):
        self.statusLabel.setMovie(self.movie)
        self.movie.start()
        
        self.thread = ProcessingThread(algorithm)
        self.thread.result_ready.connect(self.on_result_ready)
        self.thread.start()

    # 结束movie的动画
    def on_result_ready(self):
        self.movie.stop()
        self.statusLabel.clear()


    def calibration_algorithm(self):
        if self.excel_file_path != None:
            model_str, x2, real_weight, y_fit, bin_centers, bin_means, bin_errors, R2 = calibration_process(self.excel_file_path, self.image_data_col, self.balance_data_col, "H")
            sc1 = Figure(figsize=(4, 4), dpi=self.dpi)
            axes1 =sc1.add_subplot(111)
            axes1.clear()
            # 绘制线性拟合图
            axes1.plot(x2, y_fit, label='Calibration', linestyle='-', color = "r")
            axes1.scatter(x2, real_weight, label='Real Volume', marker='.', color = "b")
            axes1.set_title("Linear Fitting Results of the Calibration Measurement Method", fontsize = 10)
            axes1.set_xlabel("Liquid Segment Length in the Pixel Coordinate System (pixels)")
            axes1.set_ylabel("Real Volume (mL)")
            axes1.grid(True, linestyle='--', alpha=0.6)
            axes1.text(2000, 0.5, f"$R^2 = {R2:.3f}$", fontsize=12, color='k')
            axes1.legend()
            self.left_canvas.update_with_figure(sc1)
            sc2 = Figure(figsize=(4, 4), dpi=self.dpi)
            axes2 = sc2.add_subplot(111)
            axes2.errorbar(bin_centers, bin_means, yerr=bin_errors, fmt='o', label='Measurement',
                     color='b', ecolor='g', elinewidth=2, capsize=4, markersize = 8, markerfacecolor='none', markeredgewidth=0.8)
            axes2.plot(real_weight, real_weight, 'r--', linewidth=1, label='True Value')
            axes2.grid(True, linestyle='--', alpha=0.6)
            axes2.set_title('Error Bar Chart Of the Calibration Measurement Method', fontsize = 10)
            axes2.set_xlabel('Real Volume/mL')
            axes2.set_ylabel('Calibration-Based Volume Measurement/mL')
            axes2.legend(loc='upper left')
            self.right_canvas.update_with_figure(sc2)

        else:
            self.text_display.clear()
            self.text_display.append("请先打开采集的excel数据文件")
            return None
        

    def open_input_dialog(self):
        if self.image_path is not None:
            dialog = InputDialog(self)
            if dialog.exec_() == QDialog.Accepted:
                self.distribution = dialog.distribution
                self.circle_distance = dialog.circle_distance
                self.text_display.append(
                    f"特征点分布: {self.distribution}\n圆心距离: {self.circle_distance}"
                )
                self.start_task(self.homography_cal_algorithm)
            else:
                self.text_display.append("用户取消了输入。")
        else:
            self.text_display.append(f"请先打开带有标记点的图片")
    
    def on_clear_all_call(self):
        self.text_display.clear()
        self.start_task(self.clear_all)
    
    def clear_all(self):
        self.left_canvas.clear()
        self.right_canvas.clear()
        self.image_display.clear()
        self.image_display.setText("图像显示区域")
        self.excel_file_path = None
        self.image_data_col = "F"
        self.balance_data_col = "J"
        self.image_path = None
        self.distribution = None
        self.distance = None
        
        

        



class InputDialog(QDialog):
    def __init__(self, parent=None):
        super(InputDialog, self).__init__(parent)
        self.setWindowTitle("输入特征点分布和圆心距离")
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()

        # 特征点分布输入
        distribution_layout = QHBoxLayout()
        distribution_label = QLabel("特征点分布（例如 [8,1,1,8]）：")
        self.distribution_input = QLineEdit()
        self.distribution_input.setPlaceholderText("[8,1,1,8]")
        self.distribution_input.setText("[8,1,1,8]")  # 设置默认值
        distribution_layout.addWidget(distribution_label)
        distribution_layout.addWidget(self.distribution_input)
        layout.addLayout(distribution_layout)

        # 圆心距离输入
        distance_layout = QHBoxLayout()
        distance_label = QLabel("圆心距离 (circle_distance, 单位: cm)：")
        self.distance_input = QLineEdit()
        self.distance_input.setPlaceholderText("请输入圆心距离，例如 1")
        self.distance_input.setText("1")  # 设置默认值
        distance_layout.addWidget(distance_label)
        distance_layout.addWidget(self.distance_input)
        layout.addLayout(distance_layout)

        # 按钮
        button_layout = QHBoxLayout()
        self.ok_button = QPushButton("确定")
        self.cancel_button = QPushButton("取消")
        button_layout.addStretch()
        button_layout.addWidget(self.ok_button)
        button_layout.addWidget(self.cancel_button)
        layout.addLayout(button_layout)

        self.setLayout(layout)

        # 连接信号
        self.ok_button.clicked.connect(self.validate_and_accept)
        self.cancel_button.clicked.connect(self.reject)

    def validate_and_accept(self):
        distribution_text = self.distribution_input.text().strip()
        distance_text = self.distance_input.text().strip()

        # 验证特征点分布格式
        if not (distribution_text.startswith('[') and distribution_text.endswith(']')):
            QMessageBox.warning(self, "输入错误", "特征点分布应为类似 [8,1,1,8] 的格式。")
            return

        try:
            # 转换为列表
            distribution = [int(x.strip()) for x in distribution_text[1:-1].split(',')]
        except ValueError:
            QMessageBox.warning(self, "输入错误", "特征点分布中的所有元素应为整数。")
            return

        # 验证圆心距离为正数
        try:
            circle_distance = float(distance_text)
            if circle_distance <= 0:
                raise ValueError
        except ValueError:
            QMessageBox.warning(self, "输入错误", "圆心距离应为一个正数。")
            return

        # 如果验证通过，保存数据并关闭对话框
        self.distribution = distribution
        self.circle_distance = circle_distance
        self.accept()