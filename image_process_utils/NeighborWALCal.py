# @CreateTime: Dec 13, 2024 11:18 AM 
# @Author: Howard 
# @Contact: wangh22@mails.tsinghua.edu.cn 
# @Last Modified By: Howard
# @Last Modified Time: Dec 13, 2024 2:34 PM
# @Description: Modify Here, Please 

import cv2
import numpy as np
import os
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import pandas as pd

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

# 获取点集中每个点的径向和切向方向
def calculate_tangents_and_radials(points):
    # 确保points为float64类型
    points = np.asarray(points, dtype=np.float64)  
    # 计算切线向量
    tangents = np.zeros_like(points)
    tangents[1:-1] = points[2:] - points[:-2]  # 中间点
    tangents[0] = points[1] - points[0]  # 第一个点
    tangents[-1] = points[-1] - points[-2]  # 最后一个点
    # 归一化切线向量
    norms = np.linalg.norm(tangents, axis=1, keepdims=True) # 计算每一个二维数组的范数
    norms[norms == 0] = 1  # 避免除以0
    tangents /= norms
    # 计算径向向量
    radials = np.zeros_like(tangents)
    radials[:, 0] = -tangents[:, 1]
    radials[:, 1] = tangents[:, 0]
    
    return tangents, radials

def mark_and_save_points_based_on_brightness(image, points, tangents, radials, radial_distance, tangent_distance, threshold):
    """
    对于每个点，计算其在图像上的切向和径向领域的加权平均亮度，如果加权平均亮度高于阈值，则在图像上标记该点，并保存这些点的坐标。
    
    :param image: 一个二维的numpy数组，表示图像。
    :param points: 点集，每个点是一个(x, y)坐标。
    :param tangents: 每个点的切线方向向量。
    :param radials: 每个点的径向方向向量。
    :param radial_distance: 径向领域的距离。
    :param tangent_distance: 切向领域的距离。
    :param threshold: 加权亮度阈值。
    :return: 修改后的图像和满足条件的点的列表；每个点对应领域的加权平均亮度；每个点对应的领域点
    """
    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("image", 640, 360)
        
    high_brightness_points = []
    marked_image = image.copy()
    weighted_brightness_list = []
    nb_for_point = []

    for point, tangent, radial in zip(points, tangents, radials):
        point = np.asarray(point, dtype=np.float64)
        tangent = np.asarray(tangent, dtype=np.float64)
        radial = np.asarray(radial, dtype=np.float64)
        neighborhood_brightness = []
        domain_point = []
        weights = []

        for d in np.linspace(-radial_distance, radial_distance, num=2*radial_distance+1):
            for t in np.linspace(-tangent_distance, tangent_distance, num=2*tangent_distance+1):
                neighborhood_point = point + d * radial + t * tangent
                x, y = int(round(neighborhood_point[0])), int(round(neighborhood_point[1]))
                if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                    cv2.circle(marked_image, (x, y), radius=1, color=(255, 0, 0), thickness=-1)
                    brightness = image[y, x]
                    if brightness != 0:
                        neighborhood_brightness.append(brightness)
                        domain_point.append((x, y))
                        weights.append(weight_function(brightness))

        nb_for_point.append(domain_point)

        if neighborhood_brightness:
            weighted_avg_brightness = np.average(neighborhood_brightness, weights=weights)
            weighted_brightness_list.append(weighted_avg_brightness)

            x, y = int(round(point[0])), int(round(point[1]))
            if weighted_avg_brightness > threshold:
                cv2.circle(marked_image, (x, y), radius=1, color=(0, 0, 0), thickness=-1)
                high_brightness_points.append((x, y))
            else:
                cv2.circle(marked_image, (x, y), radius=1, color=(255, 255, 255), thickness=-1)

        cv2.imshow("image", marked_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    return high_brightness_points, weighted_brightness_list, nb_for_point



def weight_function(brightness, threshold=110, base_weight=30, max_brightness=255):
    if brightness <= threshold:
        # 使用四次方函数进行权重的非线性下降，增加凸性
        ratio = (brightness / threshold)**4  # 将二次方改为四次方，增加凸性
        # 计算权重，并保证在亮度非常低时接近基础权重
        weight = base_weight - (base_weight - 1) * ratio
    else:
        # 亮度超出阈值后，权重固定为1
        weight = 1
    return weight


# # 输出或处理neighborhoods
def find_bright_points_TR(image, points, radial_num, tangent_num, brightness_threshold):
    """
    找出图像中切向与径向邻域平均亮度超过给定阈值的点。
    
    :param image: 二维numpy数组，表示灰度图像。
    :param points: 点的列表，每个点是(row, col)的形式。
    :param radial_num: 径向领域的像素数量
    :param tangent_num: 切向领域的像素数量
    :return: 平均亮度超过阈值的点的列表。
    :return: 每个点的平均亮度
    """
    average_brightness_list =[]
    # 计算每个点的径向方向与切向方向
    tangents, radials = calculate_tangents_and_radials(points)
    # 计算每个点的邻域加权平均亮度
    bright_points, average_brightness_list, nb_for_point = mark_and_save_points_based_on_brightness(image,points,tangents,radials,radial_num,tangent_num, brightness_threshold)
    return bright_points, average_brightness_list, nb_for_point


if __name__ == '__main__':
    imagefile_directory = "D:\Onedrive-University of Cincinnati\OneDrive - University of Cincinnati\Desktop\Yunjing\Vision_MicroFluid_Measurement\Data\Data_2024-06-23" 
    data_num = 12
    public_line_pth = os.path.join(imagefile_directory, f"public_line-change-2024-12-13-12.npz")
    excel_pth = os.path.join(imagefile_directory,f"{data_num}\data.xlsx")
    spline_x, spline_y = np.load(public_line_pth).values()
    frame_pth = os.path.join(imagefile_directory,f"{data_num}\Exiting-push.jpg")
    image = cv2.imread(frame_pth)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = distort_image(image)

    # 轨迹点集
    spline_points = np.column_stack((spline_x,spline_y))
    tangent_num = 3 # 切向领域的一半长度
    radial_num = 15 # 径向领域的一半长度
    brightness_threshold_tr = 90 # 阈值下限
    bright_points, AV_B_LIST, nb_for_point = find_bright_points_TR(image, spline_points, radial_num, tangent_num, brightness_threshold_tr)
    df = pd.DataFrame(AV_B_LIST, columns=['Numbers'])
    df.to_excel(excel_pth, index=False)
    X = list(range(len(AV_B_LIST))) 
    plt.figure(figsize =(10,5))
    plt.plot(X, AV_B_LIST)
    plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()



    
    

    
    



