from PyQt5.QtWidgets import QPushButton, QVBoxLayout, QLabel, QFileDialog, QWidget, QHBoxLayout, QTextEdit
from PyQt5.QtGui import QPixmap
from base_window import BaseWindow

class MeasurementWindow(BaseWindow):
    """Measurement window"""
    def __init__(self):
        super().__init__("测量界面")
        
    def add_content(self, layout):
        # Image selection button
        self.image_select_btn = QPushButton("选择图像文件")
        self.image_select_btn.clicked.connect(self.select_image)
        self.image_select_btn.setStyleSheet("""
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
        
        # Image display areas
        self.left_image_display = QLabel()
        self.left_image_display.setFixedSize(300, 300)
        self.left_image_display.setStyleSheet("""
            border: 2px solid #9E9E9E; 
            background-color: #f0f0f0; 
            border-radius: 5px;
        """)
        
        self.right_image_display = QLabel()
        self.right_image_display.setFixedSize(300, 300)
        self.right_image_display.setStyleSheet("""
            border: 2px solid #9E9E9E; 
            background-color: #f0f0f0; 
            border-radius: 5px;
        """)
        
        # Measurement buttons
        self.calibration_measure_btn = QPushButton("标定测量")
        self.calibration_measure_btn.setStyleSheet("""
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
        self.homography_measure_btn.setStyleSheet("""
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
        self.text_display = QTextEdit("测量结果")
        self.text_display.setStyleSheet("""
            font-size: 14px; 
            border: 2px solid #9E9E9E; 
            border-radius: 5px;
        """)
        
        # Layout setup
        layout.addWidget(self.image_select_btn)
        
        image_layout = QHBoxLayout()
        image_layout.addWidget(self.left_image_display)
        image_layout.addWidget(self.right_image_display)
        
        button_layout = QVBoxLayout()
        button_layout.addWidget(self.calibration_measure_btn)
        button_layout.addWidget(self.homography_measure_btn)
        
        main_layout = QHBoxLayout()
        main_layout.addLayout(button_layout)
        main_layout.addWidget(self.text_display)
        
        layout.addLayout(image_layout)
        layout.addLayout(main_layout)
        
    def select_image(self):
        # File dialog for selecting image files
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "选择图像文件", "", "Image Files (*.png *.jpg *.bmp);;All Files (*)", options=options)
        if file_name:
            pixmap = QPixmap(file_name)
            self.left_image_display.setPixmap(pixmap.scaled(self.left_image_display.size(), aspectRatioMode=1))
