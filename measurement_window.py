# @CreateTime: Jan 12, 2025 5:10 PM 
# @Author: Howard 
# @Contact: wangh22@mails.tsinghua.edu.cn 
# @Last Modified By: Howard
# @Last Modified Time: Jan 12, 2025 7:58 PM
# @Description: Modify Here, Please 

from PyQt5.QtWidgets import (QPushButton, QLabel, QHBoxLayout, QVBoxLayout, 
                            QFileDialog, QWidget, QGroupBox, QSpacerItem, QSizePolicy, QTextEdit)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QMovie
from base_window import BaseWindow
import os
import sys
import numpy as np
from image_process_utils.image_processing_c import preprocessing, liquidSegemntation
import cv2

class ProcessingThread(QThread):
    result_ready = pyqtSignal()

    def __init__(self, algorithm):
        super().__init__()
        self.algorithm = algorithm

    def run(self):
        # 在这里调用 self.algorithm 来执行特定的算法
        result = self.algorithm()
        self.result_ready.emit()


class MeasurementWindow(BaseWindow):
    """Measurement window"""
    def __init__(self):
        super().__init__("测量界面")
        self.image_path = None
        self.save_path = None
        
    def add_content(self, layout):
      
        # Image selection button
        self.image_select_btn = QPushButton("选择图像文件")
        self.image_select_btn.clicked.connect(self.load_image)
        self.image_select_btn.setStyleSheet("""
        QPushButton {
            font-size: 14px;
            padding: 8px;
            border-radius: 4px;
            background-color: #2196F3;
            color: white;
            border: none;
        }
        QPushButton:hover {
            background-color: #45a049;
        }
        """)

        # Image display areas
        image_layout = QHBoxLayout()
        # Original image display
        self.original_image_label = QLabel("待测图像数据")
        self.original_image_label.setAlignment(Qt.AlignCenter)
        self.original_image_label.setStyleSheet("""
            QLabel {
                background: #333;
                color: #fff;
                font-size: 20px;
                border: 2px solid #555;
                min-width: 400px;
                min-height: 300px;
            }
        """)
        
    
        self.original_image_label.setScaledContents(True)
        
        # Processed image display
        self.processed_image_label = QLabel("液段分割结果")
        self.processed_image_label.setAlignment(Qt.AlignCenter)
        self.processed_image_label.setStyleSheet("""
            QLabel {
                background: #333;
                color: #fff;
                font-size: 20px;
                border: 2px solid #555;
                min-width: 400px;
                min-height: 300px;
            }
        """)

        
        
        # Measurement buttons
        self.calibration_measure_btn = QPushButton("标定测量")
        self.calibration_measure_btn.clicked.connect(lambda: self.start_task(self.calibration_measurement_algorithm))
        self.calibration_measure_btn.setStyleSheet("""
        QPushButton {
            font-size: 14px;
            padding: 8px;
            border-radius: 5px;
            background-color: #2196F3;
            color: white;
            border: none;
        }
        QPushButton:hover {
            background-color: #1e88e5;
        }
        """)
        
        self.homography_measure_btn = QPushButton("单应矩阵测量")
        self.homography_measure_btn.clicked.connect(lambda: self.start_task(self.homography_measurement_algorithm))
        self.homography_measure_btn.setStyleSheet("""
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
        
        # Text display area
        self.text_display = QLabel("测量结果显示处")
        self.text_display.setStyleSheet("""
            font-size: 14px; 
            border: 2px solid #9E9E9E; 
            border-radius: 5px;
        """)
        # 设置文本左对齐
        self.text_display.setAlignment(Qt.AlignCenter)

        # 右下角加载动画
        self.otherfunction = QHBoxLayout()
        self.statusLabel = QLabel(self)
        self.statusLabel.setFixedSize(50, 50)  # 增大控件尺寸
        self.statusLabel.setStyleSheet("background: transparent;")  # 添加边框便于调试
        self.statusLabel.raise_()  # 确保在最上层
        self.movie = QMovie("image/loading.gif")
        self.movie.setScaledSize(self.statusLabel.size())  # 缩放GIF尺寸
        spacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.otherfunction.addItem(spacer)
        self.otherfunction.addWidget(self.statusLabel)

        # 一键清除功能
        self.clearall = QPushButton("一键清除")
        self.clearall.clicked.connect(self.clear_all)
        self.clearall.setFixedSize(120, 30)
        self.clearall.setStyleSheet("""
                QPushButton {
                    background: #4CAF50;
                    color: white;
                    font-size: 13px;
                    border: none;
                    border-radius: 4px;
                }
                QPushButton:hover {
                    background: #45a049;
                }
        """)
        spacer = QSpacerItem(50, 20, QSizePolicy.Fixed, QSizePolicy.Minimum)
        
        # Layout setup
        vbox = QVBoxLayout()
        vbox.addWidget(self.image_select_btn)
        vbox.addItem(spacer)
        image_layout.addWidget(self.original_image_label)
        image_layout.addWidget(self.processed_image_label)
        vbox.addLayout(image_layout)

        button_layout = QVBoxLayout()
        # 创建固定宽度的占位组件（例如 50 像素宽）
        spacer_button = QSpacerItem(50, 50, QSizePolicy.Fixed, QSizePolicy.Minimum)
        button_layout.addWidget(self.calibration_measure_btn)
        button_layout.addItem(spacer_button)
        button_layout.addWidget(self.homography_measure_btn)
        button_layout.addItem(spacer_button)
        button_layout.addWidget(self.clearall)
        
        main_layout = QHBoxLayout()
        main_layout.addLayout(button_layout)
        main_layout.addWidget(self.text_display)
        vbox.addItem(spacer)
        vbox.addLayout(main_layout)
        vbox.addLayout(self.otherfunction)
        layout.addLayout(vbox)
        
        
    def load_image(self):
        """Handle image loading"""
        file_name, _ = QFileDialog.getOpenFileName(
            self, "选择图像文件", "", "Images (*.png *.jpg *.bmp)")
        if file_name:
            self.image_path = file_name
            parent_directory = os.path.dirname(self.image_path)
            self.save_path = parent_directory
            pixmap = QPixmap(file_name)
            self.original_image_label.setPixmap(
                pixmap.scaled(
                self.processed_image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
            )

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

    def homography_measurement_algorithm(self):
    # 获取对应的路径
        if self.image_path:
            parent_directory = os.path.dirname(self.image_path)
            grand_directory = os.path.dirname(parent_directory)
            public_line_pth = os.path.join(grand_directory, f"Preprocess_public_line.npz")
            # 首先判断需不需要进行前处理
            if os.path.exists(public_line_pth):
                spline_points, radial_vector, tangent_vectors = np.load(public_line_pth)
                data = np.load(public_line_pth)
                spline_points = data["spline_points"]
                radial_vector = data["radial_vector"]
                tangent_vectors = data["tangent_vectors"]
                image = cv2.imread(self.image_path)
                image_seg, adthresh_image, length, length_world =  liquidSegemntation(image, spline_points=spline_points, radial_vector= radial_vector, tangent_vectors=tangent_vectors)
                pixmap_image_seg = self.cv2pixmap(image_seg)
                self.processed_image_label.setPixmap(
                    pixmap_image_seg.scaled(self.processed_image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                )
                # 直接测量法
                D = 1.6 # mm
                Real_weight = length_world * (D**(2))/400*np.pi

                result_text = f"""图片中的液段像素长度为{round(length, 3)} pixel, 转换到世界坐标系下的液段长度为{round(length_world, 3)} cm,
                单应矩阵尺度转换测量法的体积测量结果为{round(Real_weight, 3)}毫升"""
                self.text_display.setText(result_text)
            else:
                    self.text_display.setText("请前往图像处理窗口进行图像预处理")
        else:
            self.text_display.setText("请打开图片")

    def calibration_measurement_algorithm(self):
    # 获取对应的路径
        if self.image_path:
            parent_directory = os.path.dirname(self.image_path)
            grand_directory = os.path.dirname(parent_directory)
            public_line_pth = os.path.join(grand_directory, f"Preprocess_public_line.npz")
            # 首先判断需不需要进行前处理
            if os.path.exists(public_line_pth):
                spline_points, radial_vector, tangent_vectors = np.load(public_line_pth)
                data = np.load(public_line_pth)
                spline_points = data["spline_points"]
                radial_vector = data["radial_vector"]
                tangent_vectors = data["tangent_vectors"]
                image = cv2.imread(self.image_path)
                image_seg, adthresh_image, length, length_world =  liquidSegemntation(image, spline_points=spline_points, radial_vector= radial_vector, tangent_vectors=tangent_vectors)
                pixmap_image_seg = self.cv2pixmap(image_seg)
                self.processed_image_label.setPixmap(
                    pixmap_image_seg.scaled(self.processed_image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                )
                Real_weight = 0.00013*length+0.00498
                print(Real_weight)

                result_text = f"""图片中的液段像素长度为{round(length, 3)} pixel, 使用的标定模型为y = 0.00013x+0.00498,
                标定测量法的体积测量结果为{round(Real_weight, 3)}毫升"""
                self.text_display.setText(result_text)
            else:
                    self.text_display.setText("请前往图像处理窗口进行图像预处理")
        else:
            self.text_display.setText("请打开图片")
        

    def clear_all(self):
        self.image_path = None
        self.save_path = None
        self.original_image_label.clear()
        self.original_image_label.setText("待测图像数据")
        self.processed_image_label.clear()
        self.processed_image_label.setText("液段分割结果")
        self.text_display.clear()
        self.text_display.setText("测量结果显示处")

