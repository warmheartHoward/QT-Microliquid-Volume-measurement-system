# @CreateTime: Dec 11, 2024 4:17 PM 
# @Author: Howard 
# @Contact: wangh22@mails.tsinghua.edu.cn 
# @Last Modified By: Howard
# @Last Modified Time: Dec 12, 2024 10:23 PMM
# @Description: Function of extracting the trajectory line of the transparent microtube

import cv2
import numpy as np
import os
import datetime
Time = datetime.datetime.now().strftime("%Y-%m-%d")
# 首先根据聚类简化点集
from sklearn.cluster import KMeans
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt


def distort_image(image):
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

# 先膨胀，将轮廓变大后提取其外部轮廓
def mask_contour_find(mask) -> list:
    h, w = mask.shape[:2]
    erode_kernel = np.ones((13,13),np.uint8) 
    dilted_kernel = np.ones((3,3),np.uint8) 
    # 先膨胀
    mask = cv2.dilate(mask,dilted_kernel)
    # 膨胀之后获取最大轮廓
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    longest_contour = max(contours, key=lambda x: cv2.arcLength(x, True))

    # 创建新的掩膜
    mask_new = np.zeros((h,w),dtype=np.uint8)
    mask = cv2.drawContours(mask_new,[longest_contour],0, (255,255,255),cv2.FILLED)
    # 最后再腐蚀
    # 使用中值滤波平滑看看效果
    mask = cv2.medianBlur(mask, 45)
    return mask

# 提取轮廓中的点集
def contour_points_find(ROI_image) -> list:
    ROI_image = cv2.cvtColor(ROI_image, cv2.COLOR_BGR2GRAY)
    _, ROI_image = cv2.threshold(ROI_image, 80, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(ROI_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    longest_contour = max(contours, key=lambda x: cv2.arcLength(x, True))
    contour_points = longest_contour[:, 0]
    x_coordinates = contour_points[:, 0]
    y_coordinates = contour_points[:, 1]
    return x_coordinates, y_coordinates

# 对微液管道轮廓进行聚类
def cluster_points(x_vals, y_vals, n_clusters=100):
    # 将点集格式化为(sklearn期望的)二维数组形式
    points = np.column_stack((x_vals, y_vals))
    
    # 应用K-means聚类
    kmeans = KMeans(n_clusters=n_clusters).fit(points)
    
    # 获取聚类中心
    centers = kmeans.cluster_centers_
    return centers[:, 0], centers[:, 1]  # x_vals, y_vals

def find_nearest_point(points, current_point):
    # 计算距离当前点的距离（欧几里得距离）
    distances = np.linalg.norm(points - current_point, axis=1)
    nearest_point_index = np.argmin(distances)
    return nearest_point_index


# TODO： 可以思考如何提高效率
def points_sortting(x_list:list,y_list:list):
    # 对点集按照轨迹进行排序
    # 找到左上角的点
    points = np.column_stack((x_list, y_list))
    start_points = np.argmin(np.sum(points, axis=1))
    current_index = start_points
    sorted_points = []
    sorted_points.append(points[current_index])

    while len(points) > 1:
        # 从列表中移除当前点
        points = np.delete(points, current_index, axis=0)
        current_index = find_nearest_point(points, sorted_points[-1])
        sorted_points.append(points[current_index])
    sorted_points = np.array(sorted_points)

    return sorted_points[:,0], sorted_points[:,1]

    
def TrajectoryLineExtraction(img, mask_in, mask_out):
    # 畸变矫正
    mask_in = distort_image(mask_in)
    frame = distort_image(img)
    mask_out = distort_image(mask_out)
    # 掩码灰度化+二值化
    mask_in_gray = cv2.cvtColor(mask_in, cv2.COLOR_BGR2GRAY)
    mask_out_gray = cv2.cvtColor(mask_out, cv2.COLOR_BGR2GRAY)
    _, mask_in_b = cv2.threshold(mask_in_gray, 127, 255, cv2.THRESH_BINARY)
    _, mask_out_b = cv2.threshold(mask_out_gray, 127, 255, cv2.THRESH_BINARY)
    # 找到各自的轮廓
    result_in = mask_contour_find(mask_in_b)
    result_out= mask_contour_find(mask_out_b)
    # 掩码区域合并
    mask_bit = cv2.bitwise_and(result_in, result_out)

    # 掩码区域提取
    ROI_image = cv2.bitwise_and(frame, frame, mask = result_out)
    # ROI区域轮廓提取
    x_vals_simplified_l, y_vals_simplified_l = contour_points_find(ROI_image)
    # 轮廓点集聚类
    x_vals_clustered_C, y_vals_clustered_C = cluster_points(x_vals_simplified_l, y_vals_simplified_l, n_clusters=68)
    # 轮廓聚类点集排序
    x_sorting, y_sorting =  points_sortting(x_vals_clustered_C, y_vals_clustered_C)

    # 对轮廓聚类点集进行插值，以获得最后的轨迹点集
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
    t_fine = np.linspace(-0.003, 1.01, 3000)

    # 计算样条曲线上的点
    spline_x = cs_x(t_fine)
    spline_y = cs_y(t_fine)

    return spline_x, spline_y, ROI_image



if __name__ == "__main__":
    imagefile_directory = "D:\Onedrive-University of Cincinnati\OneDrive - University of Cincinnati\Desktop\Yunjing\Vision_MicroFluid_Measurement\Data\Data_2024-06-23" 
    data_num = 25
    mask_pth = os.path.join(imagefile_directory,f"{data_num}\Entering-Liquid_Entering_Mask.jpg")
    Exiting_mask_pth = os.path.join(imagefile_directory,f"{data_num}\Exiting-Liquid_Entering_Mask.jpg")
    frame_pth = os.path.join(imagefile_directory,f"{data_num}\Exiting-push.jpg")
    mask = cv2.imread(mask_pth)
    frame = cv2.imread(frame_pth)
    ex_mask = cv2.imread(Exiting_mask_pth)
    traject_x, traject_y, ROI_Image = TrajectoryLineExtraction(frame, mask, ex_mask)
    public_line_pth = os.path.join(imagefile_directory, f"public_line-change-{Time}-{data_num}.npz")
    np.savez(public_line_pth, spline_x = traject_x, spline_y = traject_y)
    plt.imshow(ROI_Image)
    plt.plot(traject_x, traject_y, label='Spline Curve')
    plt.axis('off')
    plt.show()
    








    
    



    









