from PyQt5.QtWidgets import (QPushButton, QLabel, QGridLayout, QHBoxLayout, 
                            QVBoxLayout, QFileDialog, QWidget)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QTimer
import cv2
from base_window import BaseWindow

class DataCollectionWindow(BaseWindow):
    """Data collection window with new layout"""
    def __init__(self):
        super().__init__("数据采集")
        self.init_camera()
        
    def init_camera(self):
        """Initialize camera and timer"""
        self.cap = cv2.VideoCapture(0)
        print("Initializing camera...")
        if not self.cap.isOpened():
            print("Failed to open camera.")
            self.camera_label.setText("无法打开摄像头")
            self.camera_active = False
            return
            
        self.camera_active = True
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # Update every 30ms (~33fps)
        
    def update_frame(self):
        """Capture and display camera frame"""
        try:
            if not self.camera_active or not self.camera_label:
                return
                
            ret, frame = self.cap.read()
            print("Capturing frame...")
            if ret:
                print("Frame captured successfully.")
                # Convert frame to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Convert to QImage
                h, w, ch = frame.shape
                bytes_per_line = ch * w
                q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                
                # Convert to QPixmap and display
                self.camera_label.setPixmap(QPixmap.fromImage(q_img).scaled(
                    self.camera_label.width(),
                    self.camera_label.height(),
                    Qt.KeepAspectRatio
                ))
        except RuntimeError:
            # Clean up if UI elements are already destroyed
            self.release_resources()
            
    def release_resources(self):
        """Release camera resources when switching windows"""
        if self.camera_active:
            self.timer.stop()
            self.cap.release()
            self.camera_active = False
            
    def closeEvent(self, event):
        """Clean up camera resources"""
        self.release_resources()
        super().closeEvent(event)
        
    def add_content(self, main_layout):
        # Main horizontal layout
        hbox = QHBoxLayout()
        
        # Left side - Camera display (2/3 width)
        self.camera_label = QLabel("相机画面")
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setStyleSheet("""
            QLabel {
                background: #333;
                color: #fff;
                font-size: 20px;
                border: 2px solid #555;
                min-width: 600px;
            }
        """)
        hbox.addWidget(self.camera_label, stretch=2)
        
        # Right side - Controls (1/3 width)
        right_panel = QVBoxLayout()
        
        # Save path section with fixed width
        save_path_widget = QWidget()
        save_path_layout = QHBoxLayout(save_path_widget)
        save_path_layout.setContentsMargins(0, 0, 0, 0)
        
        self.save_path_btn = QPushButton("选择路径")
        self.save_path_btn.setFixedSize(100, 30)
        self.save_path_btn.setStyleSheet("""
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
        self.save_path_btn.clicked.connect(self.set_save_path)
        
        self.path_label = QLabel("未选择路径")
        self.path_label.setFixedWidth(200)
        self.path_label.setStyleSheet("""
            QLabel {
                color: #666;
                font-size: 13px;
                padding: 5px;
                border: 1px solid #ddd;
                border-radius: 4px;
                background: #f8f8f8;
            }
        """)
        self.path_label.setAlignment(Qt.AlignCenter)
        
        save_path_layout.addWidget(self.save_path_btn)
        save_path_layout.addWidget(self.path_label)
        save_path_layout.addStretch(1)
        right_panel.addWidget(save_path_widget)
        
        # Pump control grid
        pump_grid = QGridLayout()
        
        # Pump 1 controls
        self.pump1_start = QPushButton("泵1 启动")
        self.pump1_stop = QPushButton("泵1 停止")
        
        # Pump 2 controls
        self.pump2_start = QPushButton("泵2 启动") 
        self.pump2_stop = QPushButton("泵2 停止")
        
        # Add to grid
        pump_grid.addWidget(self.pump1_start, 0, 0)
        pump_grid.addWidget(self.pump1_stop, 0, 1)
        pump_grid.addWidget(self.pump2_start, 1, 0)
        pump_grid.addWidget(self.pump2_stop, 1, 1)
        
        # Style pump buttons
        for btn in [self.pump1_start, self.pump1_stop, 
                   self.pump2_start, self.pump2_stop]:
            btn.setStyleSheet("""
                QPushButton {
                    background: #2196F3;
                    color: white;
                    padding: 8px;
                    font-size: 13px;
                    border: none;
                    min-width: 80px;
                    min-height: 40px;
                }
                QPushButton:hover {
                    background: #1e88e5;
                }
            """)
        
        right_panel.addLayout(pump_grid)
        
        # Add stretch to push collection controls to bottom
        right_panel.addStretch(1)
        
        # Collection control buttons at bottom
        collection_controls = QHBoxLayout()
        collection_controls.setSpacing(10)
        
        self.inlet_btn = QPushButton("进液")
        self.outlet_btn = QPushButton("排液")
        self.stop_collect_btn = QPushButton("停止采集")
        
        # Style collection buttons
        for btn in [self.inlet_btn, self.outlet_btn, self.stop_collect_btn]:
            btn.setFixedSize(100, 40)
            btn.setStyleSheet("""
                QPushButton {
                    background: #f44336;
                    color: white;
                    font-size: 14px;
                    border: none;
                    border-radius: 4px;
                }
                QPushButton:hover {
                    background: #e53935;
                }
            """)
        
        collection_controls.addStretch(1)
        collection_controls.addWidget(self.inlet_btn)
        collection_controls.addWidget(self.outlet_btn)
        collection_controls.addWidget(self.stop_collect_btn)
        collection_controls.addStretch(1)
        
        right_panel.addLayout(collection_controls)
        
        hbox.addLayout(right_panel, stretch=1)
        main_layout.addLayout(hbox)
        
    def set_save_path(self):
        """Handle save path selection"""
        path = QFileDialog.getExistingDirectory(self, "选择保存路径")
        if path:
            # Show shortened path if too long
            if len(path) > 30:
                path = "..." + path[-27:]
            self.path_label.setText(path)
            self.path_label.setStyleSheet("""
                QLabel {
                    color: #4CAF50;
                    font-weight: bold;
                    background: #f8f8f8;
                    border: 1px solid #ddd;
                    border-radius: 4px;
                }
            """)
