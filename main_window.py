'''
Author: Howard
Date: 2025-01-07 13:02:45
LastEditors: warmheartHoward 1366194556@qq.com
LastEditTime: 2025-01-10 16:17:51
FilePath: \QT\main_window.py
Description: 

Copyright (c) 2025 by ${git_name_email}, All Rights Reserved. 
'''
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QLabel, QMenuBar
from PyQt5.QtCore import Qt
import sys

from menu_windows import (
    DataCollectionWindow,
    ImageProcessingWindow,
    CalibrationWindow,
    MeasurementWindow
)

class MainWindow(QMainWindow):
    def __init__(self, username):
        super().__init__()
        self.username = username
        self.init_ui()
        self.menu_windows = {
            "data": None,
            "image": None,
            "calibration": None,
            "measurement": None
        }

    def init_ui(self):
        self.setWindowTitle('基于机器视觉的微液量进液计量系统')
        self.resize(1200, 800)
        
        # Create menu bar
        menubar = self.menuBar()
        
        # Data collection menu
        data_menu = menubar.addMenu('数据采集')
        data_action = data_menu.addAction('打开数据采集')
        data_action.triggered.connect(
            lambda: self.show_content(DataCollectionWindow))
        
        # Image processing menu
        image_menu = menubar.addMenu('图像处理')
        image_action = image_menu.addAction('打开图像处理')
        image_action.triggered.connect(
            lambda: self.show_content(ImageProcessingWindow))
        
        # Calibration menu
        calibration_menu = menubar.addMenu('计量模型标定')
        calibration_action = calibration_menu.addAction('打开标定')
        calibration_action.triggered.connect(
            lambda: self.show_content(CalibrationWindow))
        
        # Measurement menu
        measure_menu = menubar.addMenu('实际测量')
        measure_action = measure_menu.addAction('打开测量')
        measure_action.triggered.connect(
            lambda: self.show_content(MeasurementWindow))
        
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)
        
        # Welcome message
        welcome_label = QLabel(f"欢迎, {self.username}!")
        welcome_label.setObjectName('welcomeLabel')
        layout.addWidget(welcome_label)
        
        # Set layout
        central_widget.setLayout(layout)
        self.setStyleSheet(open('styles.css').read())

    def show_content(self, window_class):
        """Show content in main window with proper resource management"""
        # Get window type name
        window_type = window_class.__name__.lower()
        
        # Release resources of current window
        current_window_type = next(
            (k for k, v in self.menu_windows.items() if v and v.isVisible()),
            None
        )
        if current_window_type:
            current_window = self.menu_windows[current_window_type]
            if hasattr(current_window, 'release_resources'):
                current_window.release_resources()
        
        # Reuse existing window instance if available
        if window_type in self.menu_windows and self.menu_windows[window_type]:
            window = self.menu_windows[window_type]
        else:
            window = window_class()
            self.menu_windows[window_type] = window
            
        # Clear existing content
        if self.centralWidget().layout():
            while self.centralWidget().layout().count():
                item = self.centralWidget().layout().takeAt(0)
                if item.widget():
                    item.widget().setParent(None)  # Detach instead of delete
        
        # Create container widget and layout
        container = QWidget()
        layout = QVBoxLayout()
        container.setLayout(layout)
        
        # Add window content
        window.add_content(layout)
        
        # Add container to main window
        self.centralWidget().layout().addWidget(container)
        
        # Ensure camera resources are properly initialized
        if window_type == 'datacollectionwindow':
            window.init_camera()

    def closeEvent(self, event):
        """Clean up windows when closing"""
        for window in self.menu_windows.values():
            if window:
                window.close()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)  # QApplication是一个应用程序对象，sys.argv参数是来自命令行的参数列表。
    native_window = MainWindow("Howard")
    native_window.show()
    sys.exit(app.exec())