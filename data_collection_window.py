from PyQt5.QtWidgets import (QPushButton, QLabel, QGridLayout, QHBoxLayout, 
                            QVBoxLayout, QFileDialog, QWidget, QGroupBox, QLineEdit, QSpacerItem, QSizePolicy, QTextEdit)
from PyQt5.QtGui import QPixmap, QImage, QMovie
from PyQt5.QtCore import Qt, QTimer
import cv2
from base_window import BaseWindow
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from utils.data_collection_utils_qthread import HardwareControlThread, water_tracker 
from utils.hardware_control import Pump_MultiValve_Control
import time
import datetime
import os
from utils.image_processing import preprocessing, liquidSegemntation

class ProcessingThread(QThread):
    result_ready = pyqtSignal()

    def __init__(self, algorithm):
        super().__init__()
        self.algorithm = algorithm

    def run(self):
        # 在这里调用 self.algorithm 来执行特定的算法
        result = self.algorithm()
        self.result_ready.emit()



class DataCollectionWindow(BaseWindow):
    stop_pump_signal = pyqtSignal() # 当检测到液段首端到达标识线位置时，反馈停止进液
    # image_capture_signal = pyqtSignal() # 记录将空气推入一点后的图像

    """Data collection window with new layout"""
    def __init__(self):
        super().__init__("数据采集")
        self.pump_1_RS = 30
        self.pump_2_RS = 50
        self.init_camera()
        self.hardware_thread = None # 初始化硬件控制线程
        self.liquid_tracker = water_tracker()
   
        self.cam_id = 0
        self.saving_dir = None
        self.entering_processing_frame = False # 开始进行进液过程的图像处理信号
        self.existing_processing_frame = False # 开始进行排液过程的图像处理信号
        self.image_capture_signal = False # 用于保存对应的图像

        # 获取标识线起始点与重点
        self.label_begin = (1350,920)
        self.label_end = (1350,1050)
        self.label_color = (0, 0, 255)
        self.Label_points = self.liquid_tracker.generate_line_points(self.label_begin,self.label_end, self.frame) # 后续需要改成检测标识物
        self.operation = None
        self.frame_number = 0

   

        
    def init_camera(self):
        """Initialize camera and timer"""
        # self.video_path = ""
        # self.cap = cv2.VideoCapture(self.cam_id)
        # self.cap.set(cv2.CAP_PROP_FPS, 120) # 设置帧率
        # self.size = (1920, 1080)
        # _, self.frame =self.cap.read()
        
        if self.cam_id == 1:
            self.cap = cv2.VideoCapture(self.cam_id, cv2.CAP_DSHOW)
            self.cap.set(6, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G')) # 将视频编码格式改为MJPG
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,1900) # 将帧的高度设置为1900才能无延迟流畅运行？？？
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            # video.set(cv2.CAP_PROP_FRAME_WIDTH, 2500)
            # video.set(cv2.CAP_PROP_FRAME_HEIGHT, 1944) # 设置帧的高度
        else:
            self.cap = cv2.VideoCapture(self.cam_id)
            self.cap.set(cv2.CAP_PROP_FPS, 120) # 设置帧率
            self.size = (1920, 1080)
        _, self.frame =self.cap.read()
        self.update_status(f"打开摄像头，时间：{time.ctime(time.time())}")

        # print("Initializing camera...")
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
            ret, self.frame = self.cap.read()
            # 先获取标识线点集
            if ret:
                # Convert frame to RGB
                if self.entering_processing_frame:
                    if self.frame_number == 0:
                        Begining_path =os.path.join(self.saving_entering_dir,f"{self.operation}-Begining.jpg")
                        cv2.imwrite(Begining_path, self.frame) 
                        self.update_status(f"已保存进液初始状态图片至路径：{Begining_path}")
                    frame_Draw = self.frame.copy()
                    T_frame= self.liquid_tracker.moving_water_extraction(frame_Draw)
                    cv2.line(T_frame, self.label_begin, self.label_end, self.label_color, 3)

                    if self.frame_number >= 10:
                        contour_points = self.liquid_tracker.get_contour_points()
                        intersecting_Flag = self.liquid_tracker.is_intersecting(contour_points, self.Label_points,1)
                        frame_pixmap = self.cv2pixmap(T_frame)
                        self.camera_label.setPixmap(
                            frame_pixmap.scaled(self.camera_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                        )
                        # 录制视频
                        self.out.write(T_frame)
                        ROI_mask = cv2.cvtColor(self.liquid_tracker.ROI_mask.copy(),cv2.COLOR_GRAY2BGR)
                        self.out_mask.write(ROI_mask)
                        if intersecting_Flag:
                            self.label_color = (0,255,0)  
                            self.out.release()
                            self.out_mask.release()
                            self.update_status(f"已保存进液流程视频至：{self.saving_entering_dir}")
                            self.stop_pump_signal.emit()
                    
                elif self.image_capture_signal:
                     cv2.imwrite(os.path.join(self.saving_entering_dir, f"{self.operation}-push.jpg"), self.frame)
                     self.image_capture_signal = False
               
               
                # 排液流程显示
                elif self.existing_processing_frame:
                    if self.frame_number == 0:
                        Begining_path =os.path.join(self.saving_entering_dir,f"{self.operation}-Begining.jpg")
                        cv2.imwrite(Begining_path, self.frame) 
                        self.update_status(f"已保存排液初始状态图片至路径：{Begining_path}")
                    if self.frame_number > 10:
                        frame_Draw = self.frame.copy()
                        T_frame= self.liquid_tracker.moving_water_extraction(frame_Draw)
                        contour_points = self.liquid_tracker.get_contour_points()
                        self.out_exit.write(self.frame)
                        ROI_mask = cv2.cvtColor(self.liquid_tracker.ROI_mask.copy(),cv2.COLOR_GRAY2BGR)
                        self.out_mask_exit.write(ROI_mask)


                else:
                    frame_pixamp = self.cv2pixmap(self.frame)
                    self.camera_label.setPixmap(
                            frame_pixamp.scaled(self.camera_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                        )
                    
                    # Convert to QPixmap and display
                    # self.camera_label.setPixmap(QPixmap.fromImage(q_img).scaled(
                    #     self.camera_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
                    # ))
                self.frame_number += 1
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
        vbox = QVBoxLayout()
        hbox = QHBoxLayout()
        ##################################################################################################
        # 左边的按键
        left_side = QVBoxLayout()
        self.camera_label = QLabel("相机画面")
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setStyleSheet("""
            QLabel {
                background: #333;
                color: #fff;
                font-size: 20px;
                min-width: 800px;
            }
        """)
        self.camera_label.setFixedSize(800, 600)  # 根据需要调整大小
        # 设置三个按键
        collection_controls = QHBoxLayout()
        self.inlet_btn = QPushButton("进液")
        self.inlet_btn.setFixedHeight(50)
        self.inlet_btn.setStyleSheet("""
            QPushButton {
                background: #4CAF50;
                color: white;
                font-size: 13px;
            }
            QPushButton:hover {
                background: #45a049;
            }
        """)
        self.outlet_btn = QPushButton("排液")
        self.outlet_btn.setFixedHeight(50)
        self.outlet_btn.setStyleSheet("""
            QPushButton {
                background: #2196F3;
                color: white;
                font-size: 13px;
            }
            QPushButton:hover {
                background: #45a049;
            }
        """)
        self.stop_collect_btn = QPushButton("停止采集")
        self.stop_collect_btn.setFixedHeight(50)
        self.stop_collect_btn.setStyleSheet("""
            QPushButton {
                background: #f44336;
                color: white;
                font-size: 14px;
                border: none;
                border-radius: 4px
            }
            QPushButton:hover {
                background: #e53935;
            }
        """)
        
        collection_controls.addWidget(self.inlet_btn)
        collection_controls.addWidget(self.outlet_btn)
        collection_controls.addWidget(self.stop_collect_btn)
        left_side.addWidget(self.camera_label)
        left_side.addLayout(collection_controls)
        hbox.addLayout(left_side)
        #################################################################################################
        # 右边的布局
        right_panel = QVBoxLayout()
        # 数据保存按键
        self.save_path_btn = QPushButton("选择数据保存路径")
        self.save_path_btn.setFixedHeight(40)
        self.save_path_btn.setStyleSheet("""
            QPushButton {
                background: #4CAF50;
                color: white;
                font-size: 13px;
            }
            QPushButton:hover {
                background: #45a049;
            }
        """)
        self.save_path_btn.clicked.connect(self.set_save_path)
        right_panel.addWidget(self.save_path_btn)
        #######################################################################Pump1##################################################################
        pump1_group = QGroupBox("泵1控制")
        pump1_group.setStyleSheet("""
        QGroupBox {
            border: 2px solid gray;
            border-radius: 5px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top center;                          
            color: black;
        }
        """)
        # 设置整个QGroupBox的字体
        font = pump1_group.font()
        font.setPointSize(12)  # 设置字体大小
        font.setBold(True)     # 字体加粗
        pump1_group.setFont(font)
        # Pump control grid
        pump_1_layout = QVBoxLayout()
        pump_1_velocity_layout = QHBoxLayout()
        self.pump1_velocity_label = QLabel("设置泵1的转速(r/min):")
        self.pump1_velocity_label.setStyleSheet("""
            QLabel {
                background: transparent;
                color: black;
                font-size: 14px;
                border-radius: 2px; 
                border: none;
            }
        """)
        self.pump1_velocity_input = QLineEdit(self)
        # self.pump1_velocity_input.setFixedHeight(50)
        self.pump1_velocity_input.setText("20")
        self.pump1_velocity_input.textChanged.connect(self.on_return_pressed_pump1)
        pump_1_velocity_layout.addWidget(self.pump1_velocity_label)
        pump_1_velocity_layout.addWidget(self.pump1_velocity_input)
        pump_1_layout.addLayout(pump_1_velocity_layout)
        # Pump 1 controls
        pump_1_control_layout = QHBoxLayout()
        self.pump1_positive_start = QPushButton("泵1 正转")
        self.pump1_negativa_start = QPushButton("泵1 反转")
        self.pump1_stop = QPushButton("泵1 停止")
        pump_1_control_layout.addWidget(self.pump1_positive_start)
        pump_1_control_layout.addWidget(self.pump1_negativa_start)
        pump_1_layout.addLayout(pump_1_control_layout)
        pump_1_layout.addWidget(self.pump1_stop)
        pump1_group.setLayout(pump_1_layout)
        right_panel.addWidget(pump1_group)
        
        ############################################################################################################################################
        pump2_group = QGroupBox("泵1控制")
        pump2_group.setStyleSheet("""
        QGroupBox {
            border: 2px solid gray;
            border-radius: 5px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top center;                          
            color: black;
        }
        """)
        # 设置整个QGroupBox的字体
        font = pump2_group.font()
        font.setPointSize(12)  # 设置字体大小
        font.setBold(True)     # 字体加粗
        pump2_group.setFont(font)
        # Pump control grid
        pump_2_layout = QVBoxLayout()
        pump_2_velocity_layout = QHBoxLayout()
        self.pump2_velocity_label = QLabel("设置泵1的转速(r/min):")
        self.pump2_velocity_label.setStyleSheet("""
            QLabel {
                background: transparent;
                color: black;
                font-size: 14px;
                border-radius: 2px; 
                border: none;
            }
        """)
        self.pump2_velocity_input = QLineEdit(self)
        self.pump2_velocity_input.setText("20")
        self.pump2_velocity_input.textChanged.connect(self.on_return_pressed_pump2)
        pump_2_velocity_layout.addWidget(self.pump2_velocity_label)
        pump_2_velocity_layout.addWidget(self.pump2_velocity_input)
        pump_2_layout.addLayout(pump_2_velocity_layout)
        # Pump 2 controls
        pump_2_control_layout = QHBoxLayout()
        self.pump2_positive_start = QPushButton("泵2 正转")
        self.pump2_negativa_start = QPushButton("泵2 反转")
        self.pump2_stop = QPushButton("泵2 停止")
        pump_2_control_layout.addWidget(self.pump2_positive_start)
        pump_2_control_layout.addWidget(self.pump2_negativa_start)
        pump_2_layout.addLayout(pump_2_control_layout)
        pump_2_layout.addWidget(self.pump2_stop)
        pump2_group.setLayout(pump_2_layout)
        right_panel.addWidget(pump2_group)

        # Style pump buttons
        for btn in [self.pump1_positive_start, self.pump1_negativa_start, self.pump1_stop, 
                   self.pump2_positive_start, self.pump2_negativa_start, self.pump2_stop]:
            btn.setFixedHeight(30)
            btn.setStyleSheet("""
                QPushButton {
                    background: #2196F3;
                    color: white;
                    font-size: 13px;
                    border: none;
                    border-radius: 2px;
                }
                QPushButton:hover {
                    background: #45a049;
                }
        """)
        
        # 添加一键数据采集与一键测量的按钮
        self.data_collection = QPushButton("一键数据采集")
        self.data_collection.setFixedHeight(40)
        self.data_collection.setStyleSheet("""
            QPushButton {
                background: #4CAF50;
                color: white;
                font-size: 13px;
            }
            QPushButton:hover {
                background: #45a049;
            }
        """)

        self.measurement = QPushButton("一键测量")
        self.measurement.setFixedHeight(40)
        self.measurement.setStyleSheet("""
            QPushButton {
                background: #4CAF50;
                color: white;
                font-size: 13px;
            }
            QPushButton:hover {
                background: #45a049;
            }
        """)

        #right_panel.addWidget(self.data_collection)
        right_panel.addWidget(self.measurement)

        # Add stretch to push collection controls to bottom
        # right_panel.addStretch(1)

        self.results_label = QTextEdit()
        self.results_label.setFixedHeight(200)
        self.results_label.setReadOnly(True)
        self.results_label.setPlaceholderText("输出文本显示区域")
        self.results_label.setStyleSheet("""
            font-size: 14px; 
            border: 2px #9E9E9E; 
            border-radius: 5px;
        """)
        # self.results_label = QLabel("信息显示")
        # self.results_label.setStyleSheet("""
        #     QLabel {
        #         color: #333;
        #         font-size: 14px;
        #         padding: 10px;
        #         border: 1px solid #ddd;
        #         background: #f8f8f8;
        #     }
        # """)
        # self.results_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        # self.results_label.setWordWrap(True)  # Enable word wrap
        right_panel.addWidget(self.results_label)

        


        # 右下角加载动画
        self.otherfunction = QHBoxLayout()
        self.statusLabel = QLabel(self)
        self.statusLabel.setFixedSize(50, 50)  # 增大控件尺寸
        self.statusLabel.setStyleSheet("background: transparent;")  # 添加边框便于调试
        self.statusLabel.raise_()  # 确保在最上层

        # 一键清除功能
        self.clearall = QPushButton("一键清除")
        # self.clearall.clicked.connect(lambda: self.start_task(self.clear_all))
        self.clearall.setFixedSize(120,40)
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
        # 使用绝对路径
        self.movie = QMovie("image/loading.gif")
        self.movie.setScaledSize(self.statusLabel.size())  # 缩放GIF尺寸
        #vbox.addWidget(self.clearall, alignment=Qt.AlignBottom | Qt.AlignLeft)
        

        hbox.addLayout(right_panel)
        vbox.addLayout(hbox)
        vbox.addLayout(self.otherfunction)
        main_layout.addLayout(vbox)

        # 连接按钮事件：
        self.pump1_positive_start.clicked.connect(lambda: self.start_pump("Positive", 1))
        self.pump1_negativa_start.clicked.connect(lambda: self.start_pump("Negative", 1))
        self.pump1_stop.clicked.connect(lambda: self.stop_pump(1))
        self.pump2_positive_start.clicked.connect(lambda: self.start_pump("Positive", 2))
        self.pump2_negativa_start.clicked.connect(lambda: self.start_pump("Negative", 2))
        self.pump2_stop.clicked.connect(lambda: self.stop_pump(2))
        self.inlet_btn.clicked.connect(lambda: self.start_task(self.start_entering_algorithm))
        self.outlet_btn.clicked.connect(lambda: self.start_task(self.start_exiting_algorithm))
        self.stop_collect_btn.clicked.connect(lambda:self.start_task(self.stop_all_algorithm))
        self.clearall.clicked.connect(lambda:self.start_task(self.on_clear_all_call))

        
    def set_save_path(self):
        """Handle save path selection"""
        path = QFileDialog.getExistingDirectory(self, "选择保存路径")
        if path:
            self.saving_dir = path
            self.results_label.append(f"保存采集数据的路径为{path}")

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

    def update_status(self, status):
        """更新状态标签"""
        # current_text = self.results_label.text()
        self.results_label.append( status)

    def on_return_pressed_pump1(self):
        # 更新对象中的变量
        self.pump_1_RS = self.pump1_velocity_input.text()
        status = f"泵1的转速被设置为 {self.pump_1_RS} r/min"
        self.update_status(status)
        # print(f"Input value: {self.image_data_col}")
        # 锁定QLineEdit，使其无法输入
        self.pump1_velocity_input.setReadOnly(True)

    def on_return_pressed_pump2(self):
        # 更新对象中的变量
        self.pump_2_RS = self.pump2_velocity_input.text()
        status = f"泵2的转速被设置为 {self.pump_2_RS} r/min"
        self.update_status(status)
        # print(f"Input value: {self.image_data_col}")
        # 锁定QLineEdit，使其无法输入
        self.pump2_velocity_input.setReadOnly(True)

    def start_pump(self, status, pump_number):
        """启动指定的泵
        status: "Positive" | "Negative"
        """
        self.results_label.append(f"启动泵{pump_number}...")
        try:
            if pump_number == 1:
                T1 = Pump_MultiValve_Control("COM6")
                T1.open_pump(status, self.pump_1_RS)
                self.update_status(f"泵1 启动，转动方向: {status}, 转速: {self.pump_1_RS} r/min")
            elif pump_number == 2:
                T2 = Pump_MultiValve_Control("COM7")
                T2.open_pump(status, self.pump_2_RS)
                self.update_status(f"泵2 启动，转动方向: {status}, 转速: {self.pump_2_RS} r/min")
        except Exception as e:
            self.update_status(f"启动泵{pump_number}失败: {str(e)}")

    def stop_pump(self, pump_number):
        """停止指定的泵"""
        self.results_label.append(f"停止泵{pump_number}...")
        try:
            if pump_number == 1:
                T1 = Pump_MultiValve_Control("COM6")
                T1.close_pump()
                self.update_status("泵1 已停止")
            elif pump_number == 2:
                T2 = Pump_MultiValve_Control("COM7")
                T2.close_pump()
                self.update_status("泵2 已停止")
        except Exception as e:
            self.update_status(f"停止泵{pump_number}失败: {str(e)}")
    

    def start_entering_algorithm(self):
        """开始进液操作"""
        if self.saving_dir is not None:
            try:
                self.results_label.append("开始进液操作...")
                self.operation = "Entering"

                # self.begining_image_save = True
                if self.hardware_thread is None or not self.hardware_thread.isRunning():
                    self.hardware_thread = HardwareControlThread(operation="Entering", push_twice=True)
                    # 图像处理与显示变化
                    self.entering_processing_frame = True
                    # 处理图像初始化：
                    self.frame_number = 0
                    
                    # 设置视频录制的参数
                    self.video_output_name = f"{self.operation}-.avi"
                    self.mask_video_output_name = f"{self.operation}-Mask.avi"
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
                    self.saving_entering_dir = os.path.join(self.saving_dir, current_date)
                    os.mkdir(self.saving_entering_dir)
                    self.out = cv2.VideoWriter(os.path.join(self.saving_entering_dir,self.video_output_name), fourcc, 25.0, self.size, True)
                    self.out_mask = cv2.VideoWriter(os.path.join(self.saving_entering_dir,self.mask_video_output_name), fourcc, 25.0, self.size, True)
                    # 设置硬件线程的信号链接
                    self.hardware_thread.control_finished.connect(self.on_control_finished)
                    self.hardware_thread.status_update.connect(self.update_status)
                    self.stop_pump_signal.connect(self.hardware_thread.stop)
                    self.hardware_thread.image_save.connect(self.image_save_signal)
                    # 开始硬件线程
                    self.hardware_thread.start()
            except Exception as e:
                self.update_status(f"进液失败: {str(e)}")
        else:
            self.update_status("请先选定对应的保存路径")
    
    def image_save_signal(self, signal):
        if signal:
            self.image_capture_signal = True


    def start_exiting_algorithm(self):
        """开始排液操作"""
        if self.saving_dir is not None:
            if self.saving_entering_dir is not None:
                try:
                    print("启动排液操作")
                    self.results_label.append("开始排液操作...")
                    self.operation = "Exiting"
                    #self.entering_processing_frame = True

                    if self.hardware_thread is None or not self.hardware_thread.isRunning():
                        self.hardware_thread = HardwareControlThread(operation="Exiting", push_twice=True)
                        # 连接信号
                        self.existing_processing_frame = True
                        # 处理图像初始化：
                        self.frame_number = 0
                        # 设置视频录制的参数
                        self.video_output_name = f"{self.operation}-.avi"
                        self.mask_video_output_name = f"{self.operation}-Mask.avi"
                        fourcc = cv2.VideoWriter_fourcc(*'XVID')
                        #current_date = datetime.datetime.now().strftime("%Y-%m-%d")
                        #self.saving_entering_dir = os.path.join(self.saving_dir, current_date)
                        self.out_exit = cv2.VideoWriter(os.path.join(self.saving_entering_dir,self.video_output_name), fourcc, 25.0, self.size, True)
                        self.out_mask_exit = cv2.VideoWriter(os.path.join(self.saving_entering_dir,self.mask_video_output_name), fourcc, 25.0, self.size, True)

                        self.hardware_thread.control_finished.connect(self.on_control_finished)
                        self.hardware_thread.status_update.connect(self.update_status)
                        self.stop_pump_signal.connect(self.hardware_thread.stop)
                        self.hardware_thread.start()
                        # print("HardwareControlThread 已启动")
                except Exception as e:
                    print(f"排液失败: {str(e)}")
                    self.update_status(f"排液失败: {str(e)}")
            else:
                self.update_status("请先进行进液操作")
        else:
            self.update_status("请先选定对应的保存路径")

    def stop_all_algorithm(self):
        """停止所有操作"""
        try:
            print("停止所有操作")
            self.results_label.append("停止所有操作...")
            self.entering_processing_frame = False
            self.existing_processing_frame = False

            if self.hardware_thread and self.hardware_thread.isRunning():
                self.stop_pump_signal.emit()
                self.hardware_thread.control_finished.disconnect(self.on_control_finished)
                self.hardware_thread.status_update.disconnect(self.update_status)
                self.hardware_thread = None
                self.update_status("所有操作已停止")
        except Exception as e:
            print(f"停止操作失败: {str(e)}")
            self.update_status(f"停止操作失败: {str(e)}")

    def on_control_finished(self, operation):
        """处理控制线程完成的信号"""
        self.results_label.append(f"{operation} 操作完成。")
        self.hardware_thread = None
        self.entering_processing_frame = False
        self.existing_processing_frame = False
        self.image_capture_signal = True

    def on_clear_all_call(self):
        self.results_label.clear()
        self.start_task(self.clear_all)

    def clear_all(self):
        self.entering_processing_frame = False
        self.existing_processing_frame = False
        self.operation = None
        self.image_capture_signal = False # 用于保存对应的图像
        self.saving_dir = None

    def direct_measure_algorithm(self):
        
        pass


        

    


