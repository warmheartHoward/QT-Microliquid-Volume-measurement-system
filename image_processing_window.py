# @CreateTime: Jan 15, 2025 6:16 PM 
# @Author: Howard 
# @Contact: wangh22@mails.tsinghua.edu.cn 
# @Last Modified By: Howard
# @Last Modified Time: Jan 15, 2025 10:21 PMM
# @Description: Modify Here, Please 

from PyQt5.QtGui import QPixmap, QImage, QDoubleValidator
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QMovie
from base_window import BaseWindow
import sys
# from image_process_utils.image_processing import imageProcessing
from utils.image_processing import preprocessing, liquidSegemntation, generate_world_points, enhance_image
import cv2
import os
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel, QLineEdit,
    QTextEdit, QVBoxLayout, QHBoxLayout, QFileDialog, QGroupBox,
    QGridLayout, QDialog, QFormLayout, QDialogButtonBox, QSpacerItem, QSizePolicy, QMessageBox
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import glob


# 自定义的 FigureCanvas 类
class MplCanvas(FigureCanvas):
    def __init__(self, fig=None, parent=None, width=4, height=4, dpi=100):


        if fig is None:
            self.figure = Figure(figsize=(width, height), dpi=dpi)
            self.figure = Figure()
        else:
            self.figure = fig
        super(MplCanvas, self).__init__(self.figure)

        # 设置 SizePolicy 以允许自动调整大小
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.updateGeometry()
        self.setParent(parent)
        self.ax = self.figure.add_subplot(111)
        self.figure.tight_layout()
        
    def update_with_figure(self, new_fig: Figure):
        self.figure.clf()
        self.figure = new_fig
        self.draw()

    def clear(self):
        self.figure.clf()
        self.ax = self.figure.add_subplot(111)
        self.draw()  # 刷新画布






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
        self.cali_image_path = None
        self.pattern_size = (15, 15)
        self.circle_spacing = 3 # 毫米
        self.init_ui()
        
    def init_ui(self):
        """Initialize UI components"""
        self.setMinimumSize(1600, 1000)  # 增加窗口最小尺寸
        
    def add_content(self, main_layout):
        # Main vertical layout
        vbox = QVBoxLayout()
        button_layout = QHBoxLayout()
        calibration_image = QGroupBox("相机参数标定")
        calibration_image.setStyleSheet("""
        QGroupBox {
            border: 2px solid gray;
            border-radius: 5px;
            margin-top: 10px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top center;                          
            color: black;
        }
        """)
        # 设置整个QGroupBox的字体
        font = calibration_image.font()
        font.setPointSize(12)  # 设置字体大小
        font.setBold(True)     # 字体加粗
        calibration_image.setFont(font)


        calibration_layout = QHBoxLayout()
        self.select_calibration_bth = QPushButton("选择标定图像目录")
        self.select_calibration_bth.setFixedSize(120, 30)
        self.select_calibration_bth.clicked.connect(self.calibration_image_path_select)
        self.select_calibration_bth.setStyleSheet("""
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

        # 手动输入标定参数
        self.input_calibration_parameter = QPushButton("手动输入标定参数")
        
        self.input_calibration_parameter.setFixedSize(120, 30)
        self.input_calibration_parameter.clicked.connect(self.input_distortion_params)
        self.input_calibration_parameter.setStyleSheet("""
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

        self.calibration_func = QPushButton("相机参数标定")
        # self.calibration_func.clicked.connect(self.calibration_camera_params_algorithm)
        self.calibration_func.clicked.connect(lambda: self.start_task(self.calibration_camera_params_algorithm))
        self.calibration_func.setFixedSize(120, 30)
        self.calibration_func.setStyleSheet("""
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

        calibration_layout.addWidget(self.select_calibration_bth)
        calibration_layout.addWidget(self.calibration_func)
        calibration_layout.addWidget(self.input_calibration_parameter)
        calibration_image.setLayout(calibration_layout)
        button_layout.addWidget(calibration_image)




        ############################################################Image_feature_extraction##############################################


        # File selection button at top
        image_process_group = QGroupBox("图像特征提取")
        image_process_group.setStyleSheet("""
        QGroupBox {
            border: 2px solid gray;
            border-radius: 5px;
            margin-top: 10px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top center;                          
            color: black;
        }
        """)
        # 设置整个QGroupBox的字体
        font = image_process_group.font()
        font.setPointSize(12)  # 设置字体大小
        font.setBold(True)     # 字体加粗
        image_process_group.setFont(font)

        image_process_layout = QVBoxLayout()

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

        self.save_path_btn = QPushButton("结果保存路径")
        self.save_path_btn.clicked.connect(self.save_path_select)
        self.save_path_btn.setFixedSize(120, 30)
        file_select_layout.addWidget(self.select_file_btn)
        file_select_layout.addWidget(self.save_path_btn)

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
        
        control_layout.addWidget(self.preprocess_btn)
        control_layout.addWidget(self.segment_btn)



        
        image_process_layout.addLayout(file_select_layout)
        image_process_layout.addLayout(control_layout)
        image_process_group.setLayout(image_process_layout)
        button_layout.addWidget(image_process_group)

        #vbox.addLayout(save_layout)
        vbox.addLayout(button_layout)


        ##################################################################################################################
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

        ##############################################################################################################################################
     
        
        ##############################################################################################################################################
        result_display_layout = QHBoxLayout()
        # self.large_result_label = QLabel("自适应阈值分割示意图")
        # self.large_result_label.setAlignment(Qt.AlignCenter)
        # self.large_result_label.setStyleSheet("""
        #     QLabel {
        #         background: #333;
        #         color: #fff;
        #         font-size: 20px;
        #         min-width: 400px;
        #         min-height: 300px;                     
        #     }
        # """)
        # self.canvas = MplCanvas(width=8, height=4, dpi=100)
        self.canvas = MplCanvas(width=8, height=4, dpi=100)
        self.canvas.setFixedWidth(578)
        self.canvas.setFixedHeight(289)
        # 创建Matplotlib工具栏并添加到布局
        #self.toolbar = NavigationToolbar(self.canvas, self)
        

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
        self.results_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        #self.results_label.setFixedWidth(300)
        #self.results_label.setFixedHeight(300)
        # vbox.addWidget(self.results_label)
        
        # 输出文本
        # self.results_label = QTextEdit()
        # self.results_label.setFixedWidth(400)
        # self.results_label.setReadOnly(True)
        # self.results_label.setPlaceholderText("输出文本显示区域")
        # self.results_label.setStyleSheet("""
        #     font-size: 14px; 

        self.results_label.setWordWrap(True)  # Enable word wrap

        # """)
        result_display_layout.addWidget(self.canvas)
        result_display_layout.addWidget(self.results_label)
        # result_display_layout.addWidget(self.toolbar)
   

        vbox.addLayout(result_display_layout)

        # 右下角加载动画
        self.otherfunction = QHBoxLayout()
        self.statusLabel = QLabel(self)
        self.statusLabel.setFixedSize(50, 50)  # 增大控件尺寸
        self.statusLabel.setStyleSheet("background: transparent;")  # 添加边框便于调试
        self.statusLabel.raise_()  # 确保在最上层

        # 一键清除功能
        self.clearall = QPushButton("一键清除")
        self.clearall.clicked.connect(lambda: self.start_task(self.clear_all))
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
            self.results_label.setText(f"选定保存的路径为：{file_path}")

    def calibration_image_path_select(self):
        """
        选择标定板图像的路径
        """
                # 打开文件选择对话框，并获取用户选择的文件路径
        file_path  = QFileDialog.getExistingDirectory(
                    None,  # 父窗口
                    "Select Direatory",  # 对话框标题
                    "",  # 默认路径
                    )
        
        if file_path:
            self.cali_image_path = file_path
            self.results_label.setText(f"选定标定图像文件夹的路径为：{file_path}")

    
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
            if os.path.exists(ex_mask_directory) and self.intrinsic_parameter is not None:
                image = cv2.imread(self.image_path)
                ex_mask = cv2.imread(ex_mask_directory)
                spline_points, radial_vector, tangent_vectors, image_line_extract= preprocessing(image, ex_mask, mtx =self.intrinsic_parameter, dist = self.distortion_parameter)
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
                image_seg, adthresh_image, length, length_world =  liquidSegemntation(image, spline_points=spline_points, radial_vector= radial_vector, tangent_vectors=tangent_vectors, 
                                                                                      mtx = self.intrinsic_parameter, dist = self.distortion_parameter)
                pixmap_image_seg = self.cv2pixmap(image_seg)
                self.processed_image_label.setPixmap(
                    pixmap_image_seg.scaled(self.processed_image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                )

                self.canvas.update_with_figure(adthresh_image)
             
                result_text = f"图片中的液段像素长度为{round(length, 2)} pixel"
                self.results_label.setText(result_text)
            else:
                 self.results_label.setText("请进行一次图像的预处理")
        
        else:
            self.results_label.setText("请打开图片")
    

    def calibration_camera_params_algorithm(self):
        if self.cali_image_path:
            # Step1 生成世界坐标
            objp = generate_world_points(self.pattern_size, self.circle_spacing)
            # 获取所有图像文件路径
            image_files = glob.glob(os.path.join(self.cali_image_path, "*.jpg"))
            # Step2 图像增强
            if len(image_files) > 0:
                objpoints = []  # 3D世界坐标
                imgpoints = []  # 2D图像坐标
                # Step3 检测特征点
                for fname in image_files:
                    img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
                    enhanced_image = enhance_image(fname)
                    found, centers = cv2.findCirclesGrid(enhanced_image, self.pattern_size, None, cv2.CALIB_CB_SYMMETRIC_GRID)
                    if found:
                        objpoints.append(objp)
                        imgpoints.append(centers)
                        img_vis = cv2.drawChessboardCorners(img.copy(), self.pattern_size, centers, found)
                        img_p = self.cv2pixmap(img.copy())
                        img_vis_p = self.cv2pixmap(img_vis.copy())
                        self.original_image_label.clear()
                        self.original_image_label.setPixmap(
                            img_p.scaled(
                            self.processed_image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
                            )
                        )
                        self.processed_image_label.setPixmap(
                            img_vis_p.scaled(self.processed_image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                        )

  
                # Step4 标定相机
                ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[::-1], None, None)
                self.distortion_parameter = dist
                self.intrinsic_parameter = mtx
                # 计算重投影误差
                # 计算重投影误差
                mean_error = 0
                for i in range(len(objpoints)):
                    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
                    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
                    mean_error += error

                mean_error /= len(objpoints)
                self.results_label.setText(f"""输入的畸变: {self.distortion_parameter}, 相机内参参数:{self.intrinsic_parameter}, 重投影误差：{mean_error}""")
        else:
            pass


        
    def input_distortion_params(self):
        """弹出窗口输入畸变参数"""
        dialog = DistortionParamsDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            params = dialog.get_params()
            if params:
                distortion = params["distortion"]
                intrinsics = params["intrinsics"]
            self.distortion_parameter = [list(distortion.values())] # 1D list
            self.intrinsic_parameter = intrinsics # 别表 [3,3]
            self.results_label.setText(f"""输入的畸变: {self.distortion_parameter}, 相机内参参数:{self.intrinsic_parameter}""")
    

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
        self.canvas.clear()
        # self.large_result_label.setText("自适应阈值分割示意图")
        self.results_label.clear()
        self.results_label.setText("特征提取结果将显示在此处")
       


class DistortionParamsDialog(QDialog):
    """畸变参数和相机内参输入对话框"""

    # 定义默认畸变参数
    DEFAULT_DISTORTION_PARAMS = {
        "k1": -4.89724302e-01,
        "k2": 5.38995757e-02,
        "p1": -1.70527295e-03,
        "p2": 1.71884255e-04,
        "k3": 3.58879812e-01
    }

    # 定义默认相机内参矩阵
    DEFAULT_INTRINSICS = [
        [2.68796219e+03, 0.00000000e+00, 9.78010473e+02],
        [0.00000000e+00, 2.68879367e+03, 5.36347421e+02],
        [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("输入畸变参数和相机内参")
        self.setModal(True)
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()

        # 畸变参数分组
        distortion_group = QGroupBox("畸变参数")
        distortion_layout = QFormLayout()

        # 创建输入框并添加验证器，设置默认值
        self.k1_input = QLineEdit()
        self.k2_input = QLineEdit()
        self.p1_input = QLineEdit()
        self.p2_input = QLineEdit()
        self.k3_input = QLineEdit()

        validator = QDoubleValidator(-1e10, 1e10, 6, self)
        validator.setNotation(QDoubleValidator.StandardNotation)

        # 设置默认值并应用验证器
        self.k1_input.setValidator(validator)
        self.k1_input.setText(str(self.DEFAULT_DISTORTION_PARAMS["k1"]))

        self.k2_input.setValidator(validator)
        self.k2_input.setText(str(self.DEFAULT_DISTORTION_PARAMS["k2"]))

        self.p1_input.setValidator(validator)
        self.p1_input.setText(str(self.DEFAULT_DISTORTION_PARAMS["p1"]))

        self.p2_input.setValidator(validator)
        self.p2_input.setText(str(self.DEFAULT_DISTORTION_PARAMS["p2"]))

        self.k3_input.setValidator(validator)
        self.k3_input.setText(str(self.DEFAULT_DISTORTION_PARAMS["k3"]))

        # 可选：设置对齐方式
        for line_edit in [self.k1_input, self.k2_input, self.p1_input, self.p2_input, self.k3_input]:
            line_edit.setAlignment(Qt.AlignCenter)

        # 添加到布局
        distortion_layout.addRow("k1:", self.k1_input)
        distortion_layout.addRow("k2:", self.k2_input)
        distortion_layout.addRow("p1:", self.p1_input)
        distortion_layout.addRow("p2:", self.p2_input)
        distortion_layout.addRow("k3:", self.k3_input)

        distortion_group.setLayout(distortion_layout)
        main_layout.addWidget(distortion_group)

        # 相机内参分组
        intrinsics_group = QGroupBox("相机内参 (3x3 矩阵)")
        intrinsics_layout = QGridLayout()

        # 创建 3x3 的 QLineEdit 输入框，设置默认值
        self.intrinsics_inputs = [[QLineEdit() for _ in range(3)] for _ in range(3)]

        for i in range(3):
            for j in range(3):
                line_edit = self.intrinsics_inputs[i][j]
                line_edit.setValidator(validator)
                line_edit.setText(str(self.DEFAULT_INTRINSICS[i][j]))
                line_edit.setAlignment(Qt.AlignCenter)
                # 添加标签（例如 a11, a12, ...）
                label = QLabel(f"a{i+1}{j+1}:")
                intrinsics_layout.addWidget(label, i*2, j*2)
                intrinsics_layout.addWidget(line_edit, i*2, j*2 + 1)

        intrinsics_group.setLayout(intrinsics_layout)
        main_layout.addWidget(intrinsics_group)

        # 确认和取消按钮
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.validate_and_accept)
        buttons.rejected.connect(self.reject)
        main_layout.addWidget(buttons)

        self.setLayout(main_layout)

    def validate_and_accept(self):
        """验证输入并决定是否接受对话框"""
        errors = []

        # 验证畸变参数
        distortion_fields = {
            "k1": self.k1_input,
            "k2": self.k2_input,
            "p1": self.p1_input,
            "p2": self.p2_input,
            "k3": self.k3_input,
        }

        for name, field in distortion_fields.items():
            if not field.text().strip():
                errors.append(f"畸变参数 {name} 未填写。")
                field.setStyleSheet("border: 1px solid red;")
            else:
                field.setStyleSheet("")  # 清除错误样式

        # 验证内参矩阵
        for i in range(3):
            for j in range(3):
                field = self.intrinsics_inputs[i][j]
                if not field.text().strip():
                    errors.append(f"内参矩阵 a{i+1}{j+1} 未填写。")
                    field.setStyleSheet("border: 1px solid red;")
                else:
                    field.setStyleSheet("")  # 清除错误样式

        if errors:
            QMessageBox.warning(self, "输入错误", "\n".join(errors))
            return  # 不关闭对话框

        # 所有输入均有效，接受对话框
        self.accept()

    def get_params(self):
        """获取输入的畸变参数和相机内参"""
        # 获取畸变参数
        try:
            distortion_params = {
                "k1": float(self.k1_input.text()),
                "k2": float(self.k2_input.text()),
                "p1": float(self.p1_input.text()),
                "p2": float(self.p2_input.text()),
                "k3": float(self.k3_input.text()),
            }
        except ValueError as e:
            QMessageBox.critical(self, "输入错误", f"畸变参数输入有误: {e}")
            return None

        # 获取相机内参
        intrinsics = []
        try:
            for row in self.intrinsics_inputs:
                intrinsics_row = []
                for cell in row:
                    value = float(cell.text())
                    intrinsics_row.append(value)
                intrinsics.append(intrinsics_row)
        except ValueError as e:
            QMessageBox.critical(self, "输入错误", f"相机内参输入有误: {e}")
            return None

        return {
            "distortion": distortion_params,
            "intrinsics": intrinsics
        }