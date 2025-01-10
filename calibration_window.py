'''
Author: Howard
Date: 2025-01-07 17:20:36
LastEditors: warmheartHoward 1366194556@qq.com
LastEditTime: 2025-01-07 19:41:46
FilePath: \Date conversion\calibration_window.py
Description: 

Copyright (c) 2025 by ${git_name_email}, All Rights Reserved. 
'''
from PyQt5.QtWidgets import QPushButton, QVBoxLayout, QLabel, QFileDialog, QWidget, QHBoxLayout, QTextEdit
from PyQt5.QtGui import QPixmap
from base_window import BaseWindow

class CalibrationWindow(BaseWindow):
    """Calibration window"""
    def __init__(self):
        super().__init__("系统标定")
        
    def add_content(self, layout):
        # File selection button
        self.file_select_btn = QPushButton("选择Excel文件")
        self.file_select_btn.clicked.connect(self.select_file)
        self.file_select_btn.setStyleSheet("""
            font-size: 14px; 
            padding: 8px; 
            border-radius: 5px; 
            background-color: #4CAF50; 
            color: white;
            border: none;
        }
        QPushButton:hover {
            background-color: #45a049;
        }
        """)
        
        # Calibration and Homography buttons
        self.calibration_fit_btn = QPushButton("标定拟合")
        self.calibration_fit_btn.setStyleSheet("""
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
        self.homography_calc_btn = QPushButton("单应矩阵计算")
        self.homography_calc_btn.setStyleSheet("""
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
        
        # Image and Text display areas
        self.image_display = QLabel()
        self.image_display.setFixedSize(300, 300)  # Example size
        self.image_display.setStyleSheet("""
            border: 2px solid #9E9E9E; 
            background-color: #f0f0f0; 
            border-radius: 5px;
        """)
        self.image_display.setFixedSize(300, 300)  # Example size
        self.text_display = QTextEdit("文本显示区域")
        self.text_display.setStyleSheet("""
            font-size: 14px; 
            border: 2px solid #9E9E9E; 
            border-radius: 5px;
        """)
        
        # Layout setup
        layout.addWidget(self.file_select_btn)
        layout.addWidget(self.calibration_fit_btn)
        layout.addWidget(self.homography_calc_btn)
        
        # Horizontal layout for display areas
        display_layout = QHBoxLayout()
        display_layout.addWidget(self.image_display)
        display_layout.addWidget(self.text_display)
        
        layout.addLayout(display_layout)
        
    def display_image(self, image_path):
        # Load and display image
        pixmap = QPixmap(image_path)
        self.image_display.setPixmap(pixmap.scaled(self.image_display.size(), aspectRatioMode=1))

    def select_file(self):
        # File dialog for selecting Excel files
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "选择Excel文件", "", "Excel Files (*.xls *.xlsx);;All Files (*)", options=options)
        if file_name:
            print(f"Selected file: {file_name}")
