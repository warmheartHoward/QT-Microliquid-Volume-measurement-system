'''
Author: Howard
Date: 2025-01-07 12:59:29
LastEditors: warmheartHoward 1366194556@qq.com
LastEditTime: 2025-01-10 13:40:00
FilePath: \QT\login_window.py
Description: 

Copyright (c) 2025 by ${git_name_email}, All Rights Reserved. 
'''
from PyQt5.QtWidgets import (
    QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout, 
    QMessageBox, QHBoxLayout, QSpacerItem, QSizePolicy
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt, pyqtSignal
from database import initialize_db, register_user, validate_user
from main_window import MainWindow

class LoginWindow(QWidget):
    login_success = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        initialize_db()
        self.init_ui()

    def init_ui(self):
        # Set window properties
        self.setWindowTitle('基于机器视觉的微液量进液计量系统')
        self.resize(400, 500)
        self.setObjectName('loginWindow')
        self.center_window()

        # Create main layout
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(40, 40, 40, 40)
        main_layout.setSpacing(20)

        # Add system title
        title_label = QLabel('基于机器视觉的微液量进液计量系统')
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet('''
            QLabel {
                font-family: "Microsoft YaHei";
                font-size: 24px;
                font-weight: bold;
                color: #2c3e50;
                padding: 20px 0;
            }
        ''')
        main_layout.addWidget(title_label)

        # Add header image
        header_label = QLabel(self)
        try:
            pixmap = QPixmap("image/1.png")
            print("Image loaded successfully.")
            if pixmap.isNull():
                print("Image is null, possibly corrupted or unsupported format.")
                raise FileNotFoundError
            else:
                header_label.setPixmap(pixmap.scaledToWidth(400, Qt.SmoothTransformation))
                print("Image set to QLabel successfully.")
            header_label.setAlignment(Qt.AlignCenter)
            main_layout.addWidget(header_label)
        except:
            print("Failed to load image.")
            error_label = QLabel('无法加载图片')
            error_label.setAlignment(Qt.AlignCenter)
            error_label.setStyleSheet('color: red;')
            main_layout.addWidget(error_label)

        # Add form layout
        form_layout = QVBoxLayout()
        form_layout.setSpacing(15)

        # Username field
        self.username_input = QLineEdit()
        self.username_input.setPlaceholderText('Enter your username')
        form_layout.addWidget(self.username_input)

        # Password field
        self.password_input = QLineEdit()
        self.password_input.setPlaceholderText('Enter your password')
        self.password_input.setEchoMode(QLineEdit.Password)
        form_layout.addWidget(self.password_input)

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.setSpacing(20)
        
        self.login_button = QPushButton('Login')
        self.register_button = QPushButton('Register')
        
        button_layout.addWidget(self.login_button)
        button_layout.addWidget(self.register_button)

        # Add form and buttons to main layout
        main_layout.addLayout(form_layout)
        main_layout.addSpacerItem(QSpacerItem(20, 20, QSizePolicy.Minimum, QSizePolicy.Expanding))
        main_layout.addLayout(button_layout)

        self.setLayout(main_layout)

        # Connect signals
        self.login_button.clicked.connect(self.on_login)
        self.register_button.clicked.connect(self.on_register)

    def on_login(self):
        username = self.username_input.text()
        password = self.password_input.text()

        if not username or not password:
            QMessageBox.warning(self, 'Error', 'Please enter both username and password')
            return

        if validate_user(username, password):
            self.login_success.emit()
            self.close()
        else:
            QMessageBox.warning(self, 'Error', 'Invalid username or password')

    def center_window(self):
        """Center the window on the screen"""
        screen = self.screen().availableGeometry()
        size = self.geometry()
        self.move(
            (screen.width() - size.width()) // 2,
            (screen.height() - size.height()) // 2
        )

    def on_register(self):
        username = self.username_input.text()
        password = self.password_input.text()

        if not username or not password:
            QMessageBox.warning(self, 'Error', 'Please enter both username and password')
            return

        if register_user(username, password):
            QMessageBox.information(self, 'Success', 'Registration successful!')
        else:
            QMessageBox.warning(self, 'Error', 'Username already exists')
