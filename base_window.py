'''
Author: Howard
Date: 2025-01-07 17:19:42
LastEditors: warmheartHoward 1366194556@qq.com
LastEditTime: 2025-01-07 17:19:51
FilePath: \Date conversion\base_window.py
Description: 

Copyright (c) 2025 by ${git_name_email}, All Rights Reserved. 
'''
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt5.QtCore import Qt

class BaseWindow(QWidget):
    """Base class for all menu windows"""
    def __init__(self, title):
        super().__init__()
        self.distortion_parameter = [[-4.89724302e-01, 5.38995757e-02, -1.70527295e-03,  1.71884255e-04, 3.58879812e-01]]
        self.intrinsic_parameter =   [[2.68796219e+03, 0.00000000e+00, 9.78010473e+02],
                                      [0.00000000e+00, 2.68879367e+03, 5.36347421e+02],
                                      [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
        self.title = title
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle(self.title)
        self.resize(800, 600)
        
        # Main layout
        self.layout = QVBoxLayout()
        
        # Title label
        # title_label = QLabel(f"{self.title} 功能界面")
        # title_label.setStyleSheet("font-size: 24px; font-weight: bold;")
        # self.layout.addWidget(title_label)
        
        # Add specific content
        self.add_content(self.layout)
        
        self.setLayout(self.layout)
        
    def get_content(self):
        """Return self to be added to main window"""
        return self
