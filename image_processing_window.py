from PyQt5.QtWidgets import (QPushButton, QLabel, QHBoxLayout, QVBoxLayout, 
                            QFileDialog, QWidget, QGroupBox)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from base_window import BaseWindow

class ImageProcessingWindow(BaseWindow):
    """Image processing window with new layout"""
    def __init__(self):
        super().__init__("图像处理")
        self.image_path = None
        self.processed_image = None
        self.init_ui()
        
    def init_ui(self):
        """Initialize UI components"""
        self.setMinimumSize(1000, 800)
        
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
        self.large_result_label = QLabel("特征提取详细结果")
        self.large_result_label.setAlignment(Qt.AlignCenter)
        self.large_result_label.setStyleSheet("""
            QLabel {
                background: #333;
                color: #fff;
                font-size: 20px;
                border: 2px solid #555;
                min-height: 200px;
            }
        """)
        vbox.addWidget(self.large_result_label)
        
        # Control buttons at bottom
        control_layout = QHBoxLayout()
        
        # Preprocessing button
        self.preprocess_btn = QPushButton("图像前处理")
        self.preprocess_btn.setFixedSize(120, 30)
        
        # Segmentation button
        self.segment_btn = QPushButton("液段分割")
        self.segment_btn.setFixedSize(120, 30)
        
        # Save location
        save_layout = QVBoxLayout()
        self.save_path_btn = QPushButton("选择保存位置")
        self.save_path_btn.setFixedSize(120, 30)
        self.save_path_label = QLabel("未选择保存位置")
        self.save_path_label.setStyleSheet("""
            QLabel {
                color: #666;
                font-size: 12px;
                padding: 2px;
            }
        """)
        save_layout.addWidget(self.save_path_btn)
        save_layout.addWidget(self.save_path_label)
        
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
        control_layout.addLayout(save_layout)
        vbox.addLayout(control_layout)
        
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
        
        main_layout.addLayout(vbox)
        
    def load_image(self):
        """Handle image loading"""
        file_name, _ = QFileDialog.getOpenFileName(
            self, "选择图像文件", "", "Images (*.png *.jpg *.bmp)")
        if file_name:
            self.image_path = file_name
            pixmap = QPixmap(file_name)
            self.original_image_label.setPixmap(
                pixmap.scaled(
                    self.original_image_label.width(),
                    self.original_image_label.height(),
                    Qt.KeepAspectRatio
                )
            )
