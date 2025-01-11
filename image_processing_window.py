
from PyQt5.QtWidgets import (QPushButton, QLabel, QHBoxLayout, QVBoxLayout, 
                            QFileDialog, QWidget, QGroupBox, QSpacerItem, QSizePolicy)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QMovie
from base_window import BaseWindow
import sys
# from image_process_utils.image_processing import imageProcessing
from image_process_utils.image_processing_c import preprocessing, liquidSegemntation
import cv2
import os
import numpy as np

class ProcessingThread(QThread):
    result_ready = pyqtSignal()

    def __init__(self, algorithm):
        super().__init__()
        self.algorithm = algorithm

    def run(self):
        # 在这里调用 self.algorithm 来执行特定的算法
        result = self.algorithm()
        self.result_ready.emit()


class ImageProcessingWindow(BaseWindow):
    """Image processing window with new layout"""
    def __init__(self):
        super().__init__("图像处理")
        self.image_path = None
        self.processed_image = None
        self.save_path = None
        self.init_ui()
        
    def init_ui(self):
        """Initialize UI components"""
        self.setMinimumSize(1600, 1000)  # 增加窗口最小尺寸
        
    def add_content(self, main_layout):
        # Main vertical layout
        vbox = QVBoxLayout()
        
        # File selection button at top
        file_select_layout = QHBoxLayout()
        self.select_file_btn = QPushButton("选择图片")
        self.select_file_btn.setFixedSize(120, 30)
        self.select_file_btn.setStyleSheet("""
            QPushButton {
                background: #2196F3;
                color: white;
                font-size: 13px;
                border: none;
                border-radius: 4px;
            }
            QPushButton:hover {
                background: #1e88e5;
            }
        """)
        self.select_file_btn.clicked.connect(self.load_image)
        file_select_layout.addWidget(self.select_file_btn)
        file_select_layout.addStretch(1)
        vbox.addLayout(file_select_layout)

        # result save path
        # Save location
        save_layout = QVBoxLayout()
        self.save_path_btn = QPushButton("结果保存路径")
        self.save_path_btn.clicked.connect(self.save_path_select)
        self.save_path_btn.setFixedSize(120, 30)
        self.save_path_label = QLabel("未选择保存位置, 默认为选择图片路径")
        self.save_path_label.setStyleSheet("""
            QLabel {
                color: #666;
                font-size: 12px;
                padding: 2px;
            }
        """)
        save_layout.addWidget(self.save_path_btn)
        save_layout.addWidget(self.save_path_label)

        vbox.addLayout(save_layout)


        
        # Image display area
        image_display_layout = QHBoxLayout()
        
        # Original image display
        self.original_image_label = QLabel("原图")
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
        self.processed_image_label = QLabel("特征提取结果")
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
        
        image_display_layout.addWidget(self.original_image_label)
        image_display_layout.addWidget(self.processed_image_label)
        vbox.addLayout(image_display_layout)
        
        # Large horizontal image display
        # self.h_layout = QHBoxLayout()
        self.large_result_label = QLabel("自适应阈值分割示意图")
        self.large_result_label.setAlignment(Qt.AlignCenter)
        # self.large_result_label.setFixedSize(600, 300)
        self.large_result_label.setStyleSheet("""
            QLabel {
                background: #333;
                color: #fff;
                font-size: 20px;
                border: 2px solid #555;
                min-width: 400px;
                min-height: 300px;                     
            }
        """)
        # self.h_layout.addStretch()
        # self.h_layout.addWidget(self.large_result_label)
        # self.h_layout.addStretch()
        # vbox.addLayout(self.h_layout)
        vbox.addWidget(self.large_result_label)
        
        # Control buttons at bottom
        control_layout = QHBoxLayout()
        
        # Preprocessing button
        self.preprocess_btn = QPushButton("图像前处理")
        self.preprocess_btn.clicked.connect(lambda: self.start_task(self.preprocess_algorithm))
        self.preprocess_btn.setFixedSize(120, 30)
        
        # Segmentation button
        self.segment_btn = QPushButton("液段分割")
        self.segment_btn.clicked.connect(lambda: self.start_task(self.liquid_segmentation_algorithm))
        self.segment_btn.setFixedSize(120, 30)
        
        
        # Style buttons
        for btn in [self.preprocess_btn, self.segment_btn, self.save_path_btn]:
            btn.setStyleSheet("""
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
        
        control_layout.addWidget(self.preprocess_btn)
        control_layout.addWidget(self.segment_btn)
        # control_layout.addLayout(save_layout)
        vbox.addLayout(control_layout)
        # vbox.addLayout(save_layout)
        
        # Feature extraction results display
        self.results_label = QLabel("特征提取结果将显示在此处")
        self.results_label.setStyleSheet("""
            QLabel {
                color: #333;
                font-size: 14px;
                padding: 10px;
                border: 1px solid #ddd;
                background: #f8f8f8;
            }
        """)
        vbox.addWidget(self.results_label)
        

        # 右下角加载动画
        self.otherfunction = QHBoxLayout()
        self.statusLabel = QLabel(self)
        self.statusLabel.setFixedSize(50, 50)  # 增大控件尺寸
        self.statusLabel.setStyleSheet("background: transparent;")  # 添加边框便于调试
        self.statusLabel.raise_()  # 确保在最上层

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
        spacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.otherfunction.addWidget(self.clearall)
        self.otherfunction.addItem(spacer)
        self.otherfunction.addWidget(self.statusLabel)
        #vbox.addWidget(self.clearall, alignment=Qt.AlignBottom | Qt.AlignLeft)
        vbox.addLayout(self.otherfunction)
        # vbox.addWidget(self.statusLabel, alignment=Qt.AlignBottom | Qt.AlignRight)
        # 使用绝对路径
        self.movie = QMovie("image/loading.gif")
        self.movie.setScaledSize(self.statusLabel.size())  # 缩放GIF尺寸
        # self.statusLabel.setMovie(self.movie)  # 立即设置movie
        main_layout.addLayout(vbox)
        

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
        
    def save_path_select(self):
        """选择保存处理完后的图片路径"""
        # 打开文件选择对话框，并获取用户选择的文件路径
        file_path  = QFileDialog.getExistingDirectory(
                    None,  # 父窗口
                    "Select Direatory",  # 对话框标题
                    "",  # 默认路径
                    )
        
        if file_path:
            self.save_path = file_path
            self.save_path_label.setText(file_path)
    
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

        

    def preprocess_algorithm(self):
            # 获取对应的路径
        if self.image_path is not None:
            parent_directory = os.path.dirname(self.image_path)
            grand_directory = os.path.dirname(parent_directory)
            ex_mask_directory = os.path.join(parent_directory, "Exiting-Liquid_Entering_Mask.jpg")
            print("=================")
            if os.path.exists(ex_mask_directory) and self.image_path:
                image = cv2.imread(self.image_path)
                ex_mask = cv2.imread(ex_mask_directory)
                spline_points, radial_vector, tangent_vectors, image_line_extract= preprocessing(image, ex_mask)
                img_result_save_path = os.path.join(self.save_path, "preprocessing_tube_line_image.jpg")
                cv2.imwrite(img_result_save_path, image_line_extract)
                pixmap = self.cv2pixmap(image_line_extract)
                self.processed_image_label.setPixmap(
                    pixmap.scaled(self.processed_image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                )
                self.results_label.setText("管道中心线提取的结果如图所示")
                public_line_pth = os.path.join(grand_directory, f"Preprocess_public_line.npz")
                np.savez(public_line_pth, spline_points = spline_points, radial_vector = radial_vector, tangent_vectors= tangent_vectors )

            else:
                self.results_label.setText("数据采集不完整")
        else:
                self.results_label.setText("请打开图片")


    def liquid_segmentation_algorithm(self):
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
                image_seg, adthresh_image, length =  liquidSegemntation(image, spline_points=spline_points, radial_vector= radial_vector, tangent_vectors=tangent_vectors)
                pixmap_image_seg = self.cv2pixmap(image_seg)
                self.processed_image_label.setPixmap(
                    pixmap_image_seg.scaled(self.processed_image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                )
                pixmap_image_seg = self.cv2pixmap(adthresh_image)
                self.large_result_label.setPixmap(
                    pixmap_image_seg.scaled(self.large_result_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                )
                result_text = f"图片中的液段像素长度为{round(length, 2)} pixel"
                self.results_label.setText(result_text)
            else:
                 self.results_label.setText("请进行一次图像的预处理")
        
        else:
            self.results_label.setText("请打开图片")


    

    # 结束movie的动画
    def on_result_ready(self):
        self.movie.stop()
        self.statusLabel.clear()


    def clear_all(self):
        self.image_path = None
        self.processed_image = None
        self.save_path = None
        self.original_image_label.clear()
        self.original_image_label.setText("原图")
        self.processed_image_label.clear()
        self.processed_image_label.setText("特征提取结果")
        self.large_result_label.clear()
        self.large_result_label.setText("自适应阈值分割示意图")
        self.results_label.clear()
        self.results_label.setText("特征提取结果将显示在此处")
       


    