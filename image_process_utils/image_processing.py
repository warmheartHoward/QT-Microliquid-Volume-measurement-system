# @CreateTime: Jan 10, 2025 3:38 PM 
# @Author: Howard 
# @Contact: wangh22@mails.tsinghua.edu.cn 
# @Last Modified By: Howard
# @Last Modified Time: Jan 11, 2025 2:52 PM
# @Description: Modify Here, Please 

import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import os
import datetime
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QMovie, QPixmap, QImage

Time = datetime.datetime.now().strftime("%Y-%m-%d")
import scienceplots
plt.style.use('science')
plt.savefig('figure.svg')  # 自动识别为 SVG 格式



class imageProcessing(QThread):
    result_ready = pyqtSignal(QPixmap)

    def __init__(self, image_path, ex_mask_path, save_path):
        super().__init__()
        self.save_path = save_path
        self.ex_mask_path = ex_mask_path
        self.image = cv2.imread(image_path)
        self.ex_mask = cv2.imread(ex_mask_path)


    def distort_image(self, image):
        """
        用于矫正畸变
        """
        mtx =  [[2.68796219e+03, 0.00000000e+00, 9.78010473e+02],
                [0.00000000e+00, 2.68879367e+03, 5.36347421e+02],
                [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
        dist =  [[-4.89724302e-01, 5.38995757e-02, -1.70527295e-03,  1.71884255e-04, 3.58879812e-01]]
        mtx = np.array(mtx)
        dist = np.array(dist)
        h, w = image.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        dst = cv2.undistort(image, mtx, dist, None, newcameramtx)
        # 裁剪图像以去除黑色区域
        x, y, w, h = roi
        dst = dst[y:y + h, x:x + w]
        return dst

    def mask_contour_find(self, mask) -> list:
        h, w = mask.shape[:2]
        dilted_kernel = np.ones((3,3),np.uint8) 
        # 先膨胀
        mask = cv2.dilate(mask,dilted_kernel)
        # 膨胀之后获取最大轮廓
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        longest_contour = max(contours, key=lambda x: cv2.arcLength(x, True))
        # 创建新的掩膜
        mask_new = np.zeros((h,w),dtype=np.uint8)
        mask = cv2.drawContours(mask_new,[longest_contour],0, (255,255,255),cv2.FILLED)
        # 中值滤波
        mask = cv2.medianBlur(mask, 45)
        return mask

    def find_nearest_point(self, points, current_point):
        # 计算距离当前点的距离（欧几里得距离）
        distances = np.linalg.norm(points - current_point, axis=1)
        nearest_point_index = np.argmin(distances)
        return nearest_point_index


    def points_sortting(self, x_list:list,y_list:list):
        # 找到左上角的点
        points = np.column_stack((x_list, y_list))
        start_points = np.argmin(np.sum(points, axis=1))
        current_index = start_points
        sorted_points = []
        sorted_points.append(points[current_index])
        while len(points) > 1:
            # 从列表中移除当前点
            points = np.delete(points, current_index, axis=0)
            current_index = self.find_nearest_point(points, sorted_points[-1])
            sorted_points.append(points[current_index])
        sorted_points = np.array(sorted_points)

        return sorted_points[:,0], sorted_points[:,1]


    def preprocessing(self):
        # 步骤一 图像畸变矫正
        image = self.distort_image(self.image)
        ex_mask = self.distort_image(self.ex_mask)
        # 掩膜灰度化
        frame = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
        ex_mask = cv2.cvtColor(ex_mask, cv2.COLOR_BGR2GRAY)
        #TODO：可以在读取的时候就直接导入二值图像
        _, result_ex = cv2.threshold(ex_mask, 127, 255, cv2.THRESH_BINARY)
        result_sim_ex= self.mask_contour_find(result_ex)
        # 步骤二 图像掩膜ROI提取
        ROI_image = cv2.bitwise_and(frame, frame, mask = result_sim_ex)
        line_image = cv2.cvtColor(ROI_image.copy(), cv2.COLOR_GRAY2BGR)
        # 步骤三 对掩膜进行聚类
        points = np.column_stack(np.where(result_sim_ex == 255))
        # 应用K-means聚类
        kmeans = KMeans(n_clusters=90).fit(points)
        # 获取聚类中心
        centers = kmeans.cluster_centers_
        x_ = centers[:,1]
        y_ = centers[:,0]
        x_sorting, y_sorting =  self.points_sortting(x_, y_)
        # 原始数据点
        x = x_sorting
        y = y_sorting
        # 参数t
        t = np.linspace(0, 1, len(x))
        # 创建样条曲线
        #cs_x = CubicSpline(t, x, bc_type='natural')
        cs_x = CubicSpline(t, x, bc_type='not-a-knot')
        #cs_y = CubicSpline(t, y, bc_type='natural')
        cs_y = CubicSpline(t, y, bc_type='not-a-knot')
        # 生成细分的参数值，用于绘图
        t_fine = np.linspace(-0.003, 1.005, 3000)
        # 计算样条曲线上的点
        spline_x = cs_x(t_fine)
        spline_y = cs_y(t_fine)
        for i in range(len(spline_x)-1):
            cv2.line(line_image, (int(spline_x[i]), int(spline_y[i])),(int(spline_x[i+1]), int(spline_y[i+1])), color = (0,0,255),  thickness = 5)
        return spline_x, spline_y, line_image
    
    def cv2pixmap(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # 获取图像的维度
        height, width, channel = image_rgb.shape
        bytes_per_line = 3 * width
        # 创建 QImage，然后转换为 QPixmap
        qimage = QImage(image_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)
        return pixmap

    def run(self):
        if os.path.exists(self.ex_mask_path):
            sp_x, sp_y, image_line_extract = self.preprocessing()
            img_result_save_path = os.path.join(self.save_path, "preprocessing_tube_line_image.jpg")
            cv2.imwrite(img_result_save_path, image_line_extract)
            pixmap = self.cv2pixmap(image_line_extract)
            self.result_ready.emit(pixmap)
        else:
            self.result_ready.emit(QPixmap())
    
            


# 设置液段自适应阈值分割的按钮功能            
class liquid_segmentation(QThread):
    result_ready = pyqtSignal(QPixmap)


