from PyQt5.QtCore import QThread, pyqtSignal
from utils.hardware_control import Pump_MultiValve_Control
import time
import cv2
import numpy as np


# data_collection_utils.py

from PyQt5.QtCore import QThread, pyqtSignal
from utils.hardware_control import Pump_MultiValve_Control
import time

class HardwareControlThread(QThread):
    control_finished = pyqtSignal(str)  # 发送操作类型
    status_update = pyqtSignal(str)     # 发送状态信息
    image_save = pyqtSignal(bool) # 发送保存此时图像的信号

    def __init__(self, operation: str, push_twice: bool = True):
        super().__init__()
        self.operation = operation
        self.push_twice = push_twice
        self._run_flag = True

    def run(self):
        try:
            self.status_update.emit(f"开始 {self.operation} 操作")
            print(f"开始 {self.operation} 操作")
            T1 = Pump_MultiValve_Control("COM6")  # 尝试打开泵1的串口
            print("泵1串口已打开")
            T2 = Pump_MultiValve_Control("COM7")  # 尝试打开泵2的串口
            print("泵2串口已打开")
            time.sleep(1)  # 等待1秒以确保摄像头已经打开

            if self.operation == "Stop":
                T1.close_pump()
                T2.close_pump()
                self.status_update.emit("泵已停止")
                print("泵已停止")
            elif self.operation == "Entering":
                T1.open_pump("Negative", 30)
                self.status_update.emit("泵1 开启，进液中...")
                print("泵1 开启，进液中...")
                while self._run_flag:
                    time.sleep(0.1)  # 等待停止信号
                T1.close_pump()
                self.status_update.emit("泵1 关闭，进液停止")
                print("泵1 关闭，进液停止")
                if self.push_twice:
                    T2.open_pump("Positive", 20)
                    self.status_update.emit("泵2 开启，推入一点空气")
                    time.sleep(0.5)
                    T2.close_pump()
                    time.sleep(1)
                    self.image_save.emit(True)
                    
            elif self.operation == "Exiting":
                T2.open_pump("Positive", 50)
                self.status_update.emit("泵2 开启，排液中")
                print("泵2 开启，排液中")
                time.sleep(8)
                T2.close_pump()
                self.status_update.emit("泵2 关闭，排液完成")
                print("泵2 关闭，排液完成")

            T1.close_serial()  # 关闭泵1串口
            T2.close_serial()  # 关闭泵2串口
            self.control_finished.emit(self.operation)
            self.status_update.emit(f"{self.operation} 操作完成")
            print(f"{self.operation} 操作完成")
        except Exception as e:
            self.status_update.emit(f"操作 {self.operation} 发生错误: {str(e)}")
            self.control_finished.emit(f"操作 {self.operation} 发生错误: {str(e)}")
            print(f"操作 {self.operation} 发生错误: {str(e)}")

    def stop(self):
        self._run_flag = False


# 定义追踪液体的类
class water_tracker():
    def __init__(self):
        self.background = cv2.createBackgroundSubtractorKNN(dist2Threshold= 1000, detectShadows=False)
        self.water_contour = []
        

    def moving_water_extraction(self, frame):
        self.frame = frame
        fgmask = self.background.apply(self.frame)
        h,w = fgmask.shape
        th = cv2.threshold(fgmask.copy(), 127, 255, cv2.THRESH_BINARY)[1]
        # 中值滤波去掉椒盐噪声
        img_median = cv2.medianBlur(th, 5)

        #膨胀
        dilated = cv2.dilate(img_median, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=2)
        # 轮廓提取：有一个疑问，先用canny算子再用findContour函数能够减少很多不必要的噪声以及轮廓检测
        #edges = cv2.Canny(dilated, threshold1=10, threshold2=100)
        contours, hier = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_mask, hier = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for contour in contours:
            Area  = cv2.contourArea(contour)
            if  Area > 200 and Area <100000:
                cv2.drawContours(self.frame,[contour],0, (255,0,255),2)
        self.contours = contours
        self.contours_mask = contours_mask
        if self.contours:
            self.water_contour  = max(contours, key=lambda x: cv2.arcLength(x, True))
            #cv2.drawContours(self.frame,[self.water_contour],0, (0,0,255),cv2.FILLED)
        
        return self.frame
    
        # 通过端点生成点集
    def generate_line_points(self, start_point, end_point, frame):
        frame_shape = frame.shape[:2]
        self.ROI_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        # 计算两点之间的最大距离，确保生成的点覆盖整条线
        num_points = int(max(abs(end_point[0] - start_point[0]), abs(end_point[1] - start_point[1])))

        # 生成线性间隔的x坐标和y坐标
        x_values = np.linspace(start_point[0], end_point[0], num_points, dtype=int)
        y_values = np.linspace(start_point[1], end_point[1], num_points, dtype=int)

        # 过滤掉超出视频帧边界的点
        x_values = x_values[(x_values >= 0) & (x_values < frame_shape[1])]
        y_values = y_values[(y_values >= 0) & (y_values < frame_shape[0])]

        # 将它们组合成(x, y)格式的点
        return set(zip(x_values, y_values))
    
    # 判断是否轮廓点集与标识线点集有交集：
    def is_intersecting(self, contour_points, label_points, intersection_threshold):
        # 计算交集
        intersection = contour_points.intersection(label_points)
        # 检查交集大小是否超过阈值
        return len(intersection) >= intersection_threshold
    

    # 用了这个方法虽然可以提升稳定性但是计算负担太大了
    def get_contour_points(self):
        # 创建一个和原始图像相同大小的零矩阵
        self.mask = np.zeros(self.frame.shape[:2], dtype=np.uint8)
        # 在零矩阵上填充轮廓
        for contour in self.contours:
            cv2.drawContours(self.mask,[contour],0, 255, thickness = 2)
        #cv2.drawContours(mask, [self.water_contour], -1, 255, thickness=2)
        for contour in self.contours_mask:
            cv2.drawContours(self.ROI_mask,[contour],0, 255, cv2.FILLED)

        # cv2.namedWindow("ROI_mask", cv2.WINDOW_NORMAL)
        # cv2.resizeWindow("ROI_mask", 640,320)
        # cv2.imshow("ROI_mask",self.ROI_mask)
        

        # 获取轮廓内的所有点?? 为什么要画上去再去点，不能直接将轮廓的点转换过来么
        points = np.transpose(np.nonzero(self.mask))

        # 将点转换为(x, y)格式
        points = set((x, y) for y, x in points)  # 注意坐标的顺序
        return points