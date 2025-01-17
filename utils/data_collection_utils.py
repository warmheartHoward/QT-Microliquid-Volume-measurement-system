# @CreateTime: Jan 16, 2025 5:12 PM 
# @Author: Howard 
# @Contact: wangh22@mails.tsinghua.edu.cn 
# @Last Modified By: Howard
# @Last Modified Time: Jan 16, 2025 7:32 PM
# @Description: Modify Here, Please 

from utils.hardware_control import Connecting_Serial, Pump_MultiValve_Control
from utils.Liquid_Entering_control import water_tracker
import numpy as np
import cv2
import time
import threading
import os
import argparse
import datetime
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal

# 命令行设置
parser = argparse.ArgumentParser(description='Test for different input')
parser.add_argument("--Operation",'-o', choices = ['Entering','Exiting','Stop'], default = "Exiting", help = "Deciding whether Enter the liquid or push out the liquid")
parser.add_argument("--data_number",'-d', type = int, default = 1, help = "The number of data to be recorded")
parser.add_argument("--Push_twice",'-p',type = bool, default=True, help = "apply the air to push the liquid a little bit" )
args = parser.parse_args()

# 进液控制 COM6
def Liquid_entering(RotatingSpeed:int = 30, ContinuousTime:int = 3):
    T1 = Pump_MultiValve_Control("COM6")# 打开泵1的串口
    time.sleep(5) # 等待5s再发送串口是为了等待摄像头打开
    T1.open_pump("Negative", 30)
    time.sleep(ContinuousTime)
    T1.close_pump()
    T1.close_serial() #关闭串口


# 定义VidThread类，用于调用摄像头，显示图像
class VidThread (threading.Thread):
    def __init__(self, cam_id:int, thread_name:str, event, event2, operation, push_twice = True):
        threading.Thread.__init__(self) # 开启线程
        self.cam_id = cam_id # 摄像头ID
        self.thread_name = thread_name # 线程名称
        self.event = event # 实现线程之间的同步与通信
        self.event2 = event2 # 触发event2时, 记录空气推进一点后的图像
        self.operation = operation # 提取operation信息
        self.Pump_stop_flag = False # 用于判断泵停止了没有
        self.P = push_twice

    def run(self):
        print("开启相机线程：" + self.thread_name)
        print("%s: %s" % (self.thread_name, time.ctime(time.time())))
        self.show_image()
        print("退出线程："+ self.thread_name)

    def show_image(self):
        size = (1920, 1080)  # the size used to save the video
        #size = (2592, 1944) 
        video = cv2.VideoCapture(self.cam_id, cv2.CAP_DSHOW)
        video.set(6, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G')) # 将视频编码格式改为MJPG
        video.set(cv2.CAP_PROP_FRAME_WIDTH,1900) # 将帧的高度设置为1900才能无延迟流畅运行？？？
        video.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        # video.set(cv2.CAP_PROP_FRAME_WIDTH, 2500)
        # video.set(cv2.CAP_PROP_FRAME_HEIGHT, 1944) # 设置帧的高度
        video.set(cv2.CAP_PROP_FPS, 120) # 设置帧率
        # the parameter used to save the video

        # Data_record
        Time = datetime.datetime.now().strftime("%Y-%m-%d")
        Saving_dir = rf"Data/Data_{Time}/{args.data_number}/"
        if  not os.path.exists(Saving_dir):#如果路径不存在
            os.makedirs(Saving_dir)
        video_output_name = f"{self.operation}-.avi"
        mask_video_output_name = f"{self.operation}-Mask.avi"
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(os.path.join(Saving_dir,video_output_name), fourcc, 25.0, size, True)
        out_mask = cv2.VideoWriter(os.path.join(Saving_dir,mask_video_output_name), fourcc, 25.0, size, True)

        # 初始化进液反馈控制的类
        ret, frame = video.read()
        WT = water_tracker()
        # 获取标识线点集
        label_begin = (1350,920)
        label_end = (1350,1050)
        Label_points = WT.generate_line_points(label_begin,label_end, frame) # 后续需要改成检测标识物
        frame_number = 0  # used to record the frame number
        label_color = (0, 0, 255)
        # 用于显示帧率

        while (video.isOpened()):
            ret, frame = video.read()
            # 更新当前帧的时间戳
            #curr_frame_time = time.time()
            if frame_number == 0:
                cv2.imwrite(os.path.join(Saving_dir,f"{self.operation}-Begining.jpg"), frame) 
            if ret:
                frame_Draw = frame.copy()
                T_frame= WT.moving_water_extraction(frame_Draw)
                # 绘制标识线位置
                #(1506，690),(1557,691)
                cv2.line(T_frame, label_begin,label_end, label_color, 3)
                if frame_number > 10:
                    contour_points = WT.get_contour_points()
                    intersecting_Flag = WT.is_intersecting(contour_points,Label_points,1)
                    if intersecting_Flag:
                        label_color = (0,255,0)
                        self.event.set()
                        self.Pump_stop_flag = True
                        #pass
                        # 还需要写一个停止进液的判定，然后保存图片和ROI_mask

                    # 展示视频
                    if self.operation == "Entering":
                        cv2.namedWindow("T_frame", cv2.WINDOW_NORMAL)
                        cv2.resizeWindow("T_frame", 640,360)
                        cv2.imshow('T_frame', T_frame)
                        out.write(T_frame)
                        ROI_mask = cv2.cvtColor(WT.ROI_mask.copy(),cv2.COLOR_GRAY2BGR)
                        out_mask.write(ROI_mask)

                    elif self.operation == "Exiting":
                        out.write(frame)
                        ROI_mask = cv2.cvtColor(WT.ROI_mask.copy(),cv2.COLOR_GRAY2BGR)
                        out_mask.write(ROI_mask)
                        if self.P == True:        
                            if self.event2.wait(timeout=0.01):
                                cv2.imwrite(os.path.join(Saving_dir, f"{self.operation}-push.jpg"), frame)
                                self.event2.clear()  # Reset the event to handle future cases
                        else:
                            pass


                    
                cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("Camera", 640, 360)
                cv2.imshow("Camera", frame)
                #cv2.imwrite("Demo_for_image.jpg",frame)
                #print(frame.shape)
                if self.Pump_stop_flag and cv2.countNonZero(WT.mask) == 0:
                    self.Pump_stop_flag = False
                frame_number += 1
                # 计算并显示帧率
                #fps = 1 / (curr_frame_time - prev_frame_time)
                #prev_frame_time = curr_frame_time

                # 将帧率转换为整数
                #fps = int(fps)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cv2.imwrite(os.path.join(Saving_dir,f"{self.operation}-Liquid_Entering_Mask.jpg"), WT.ROI_mask)
                    cv2.imwrite(os.path.join(Saving_dir,f"{self.operation}-End_frame.jpg"),frame)
                    self.event.set()
                    break
            else:
                break
        video.release()
        out.release()
        out_mask.release()
        cv2.destroyAllWindows()


class HardwareControlThread(QThread):
    control_finished = pyqtSignal()

    def __init__(self, operation: str, push_twice: bool = True):
        super().__init__()
        self.operation = operation
        self.push_twice = push_twice
        self._run_flag = True

    def run(self):
        print(f"开启控制线程: 操作 {self.operation}")
        T1 = Pump_MultiValve_Control("COM6")  # 打开泵1的串口
        T2 = Pump_MultiValve_Control("COM7")  # 打开泵2的串口
        time.sleep(5)
        if self.operation == "Stop":
            T1.close_pump()
            T2.close_pump()
        elif self.operation == "Entering":
            T1.open_pump("Negative", 30)
            while self._run_flag:
                time.sleep(0.01)
                # 可以添加更多的控制逻辑
            T1.close_pump()
        elif self.operation == "Exiting":
            if self.push_twice:
                T2.open_pump("Positive", 20)
                time.sleep(0.5)
                T2.close_pump()
                time.sleep(1)
                # Emit a signal or set a flag if needed
                time.sleep(3)
                T2.open_pump("Positive", 50)
                time.sleep(5)
                T2.close_pump()
            else:
                T2.open_pump("Positive", 50)
                time.sleep(8)
                T2.close_pump()

        T1.close_serial()  # 关闭串口
        T2.close_serial()
        print(f"退出控制线程: 操作 {self.operation}")
        self.control_finished.emit()

    def stop(self):
        self._run_flag = False
        self.wait()