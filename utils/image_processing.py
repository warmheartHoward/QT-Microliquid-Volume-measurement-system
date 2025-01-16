# @CreateTime: Jan 10, 2025 3:38 PM 
# @Author: Howard 
# @Contact: wangh22@mails.tsinghua.edu.cn 
# @Last Modified By: Howard
# @Last Modified Time: Jan 15, 2025 10:25 PM
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
from utils.Adapive_Threshold_Algorithm import adaptive_threshold_algorithm
Time = datetime.datetime.now().strftime("%Y-%m-%d")
import scienceplots
plt.style.use('science')
plt.savefig('figure.svg')  # 自动识别为 SVG 格式
from io import BytesIO
from matplotlib.figure import Figure
import matplotlib



def distort_image(image, mtx, dist):
    """
    用于矫正畸变
    """
    # mtx =  [[2.68796219e+03, 0.00000000e+00, 9.78010473e+02],
    #         [0.00000000e+00, 2.68879367e+03, 5.36347421e+02],
    #         [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
    mtx = mtx
    # dist =  [[-4.89724302e-01, 5.38995757e-02, -1.70527295e-03,  1.71884255e-04, 3.58879812e-01]]
    dist = dist
    mtx = np.array(mtx)
    dist = np.array(dist)
    h, w = image.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    dst = cv2.undistort(image, mtx, dist, None, newcameramtx)
    # 裁剪图像以去除黑色区域
    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]
    return dst

def mask_contour_find(mask) -> list:
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

def find_nearest_point(points, current_point):
    # 计算距离当前点的距离（欧几里得距离）
    distances = np.linalg.norm(points - current_point, axis=1)
    nearest_point_index = np.argmin(distances)
    return nearest_point_index


def points_sortting(x_list:list,y_list:list):
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


def preprocessing(image, ex_mask, mtx, dist):
    # 步骤一 图像畸变矫正
    image = distort_image(image, mtx, dist)
    ex_mask = distort_image(ex_mask, mtx, dist)
    # 掩膜灰度化
    frame = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
    ex_mask = cv2.cvtColor(ex_mask, cv2.COLOR_BGR2GRAY)
    #TODO：可以在读取的时候就直接导入二值图像
    _, result_ex = cv2.threshold(ex_mask, 127, 255, cv2.THRESH_BINARY)
    result_sim_ex= mask_contour_find(result_ex)
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
    x_sorting, y_sorting = points_sortting(x_, y_)
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
    # 计算一阶导数
    dx_dt = cs_x(t_fine, 1)
    dy_dt = cs_y(t_fine, 1)
    # 切向向量为 (dx_dt, dy_dt)
    # 切向向量的长度
    norm_T = np.sqrt(dx_dt**2 + dy_dt**2)

    # 切向单位向量 T(t) = (dx_dt / norm_T, dy_dt / norm_T)
    T_x = dx_dt / norm_T
    T_y = dy_dt / norm_T
    # 径向方向 N(t) 可通过将 T(t) 旋转90度来获得
    # 例如：N(t) = (-T_y, T_x)
    N_x = -T_y
    N_y = T_x
    spline_points = np.column_stack((spline_x,spline_y))
    radial_vector = np.column_stack((N_x, N_y))
    tangent_vectors = np.column_stack((T_x, T_y))

    for i in range(len(spline_x)-1):
        cv2.line(line_image, (int(spline_x[i]), int(spline_y[i])),(int(spline_x[i+1]), int(spline_y[i+1])), color = (0,0,255),  thickness = 5)
    return spline_points, radial_vector, tangent_vectors, line_image


def compute_average_brightness_vectorized_with_weight(
    image, track_points, radial_vectors, tangent_vectors, radial_radius, tangent_radius,
    threshold=110, base_weight=30):
    """
    利用向量化计算每个轨迹点邻域的加权平均亮度。
    权重函数:
    如果亮度 <= threshold:
        ratio = (brightness / threshold)**4
        weight = base_weight - (base_weight - 1) * ratio
    否则:
        weight = 1

    参数：
    image: 2D numpy数组，表示灰度图像
    track_points: shape=(N,2)，每行为(x,y)
    radial_vectors: shape=(N,2)，每行为轨迹点的径向单位向量(r_x,r_y)
    tangent_vectors: shape=(N,2)，每行为轨迹点的切向单位向量(t_x,t_y)
    radial_radius, tangent_radius: 邻域半径
    threshold, base_weight: 权重函数参数

    返回：
    brightness_values: shape=(N,) 的一维向量，每个元素为相应轨迹点邻域的加权平均亮度
    """
    h, w = image.shape[:2]
    N = len(track_points)
    brightness_values = np.zeros(N)

    # 构建邻域坐标网格 (alpha,beta)
    alphas = np.arange(-tangent_radius, tangent_radius+1)
    betas = np.arange(-radial_radius, radial_radius+1)
    alpha_grid, beta_grid = np.meshgrid(alphas, betas)
    
    # 展平以方便批量计算
    alpha_flat = alpha_grid.ravel()  # shape=(M,)
    beta_flat = beta_grid.ravel()    # shape=(M,)
    M = alpha_flat.size  # 邻域像素数目

    for i in range(N):
        x_c, y_c = track_points[i]
        t_x, t_y = tangent_vectors[i]
        r_x, r_y = radial_vectors[i]

        # 利用向量化计算所有邻域点的坐标
        x_p = x_c + alpha_flat * t_x + beta_flat * r_x
        y_p = y_c + alpha_flat * t_y + beta_flat * r_y

        # 将坐标转为整数
        x_int = np.round(x_p).astype(int)
        y_int = np.round(y_p).astype(int)

        # 利用掩码过滤在图像范围内的像素
        mask = (x_int >= 0) & (x_int < w) & (y_int >= 0) & (y_int < h)

        if np.any(mask):
            valid_x = x_int[mask]
            valid_y = y_int[mask]
            valid_values = image[valid_y, valid_x].astype(float)  # 有效像素值数组

            # 根据权重函数计算权重
            weights = np.ones_like(valid_values)
            low_mask = valid_values <= threshold
            # 当亮度较低时计算相应的非线性下降权重
            ratio = (valid_values[low_mask] / threshold)**4
            weights[low_mask] = base_weight - (base_weight - 1) * ratio

            # 计算加权平均亮度：sum(weighted_value)/sum(weight)
            weighted_sum = np.sum(valid_values * weights)
            total_weight = np.sum(weights)

            brightness_values[i] = weighted_sum / total_weight if total_weight > 0 else 0.0
        else:
            brightness_values[i] = 0.0
    
    return brightness_values


matplotlib.use('Agg')  # 使用非交互式后端
def liquidSegemntation(image, spline_points, radial_vector, tangent_vectors, mtx, dist):
    # 步骤一：针对tube line提取其径向与切向邻域加权亮度
    image = distort_image(image, mtx, dist)
    radial_num = 15
    tangent_num = 2
    AV_brightness_weight = compute_average_brightness_vectorized_with_weight(image, spline_points, radial_vector, tangent_vectors, radial_num, tangent_num)
    # 步骤二，自适应阈值分割
    brightness_threshold_tr = max(AV_brightness_weight)-70 # 阈值下限
    Ad_thresh, intervals = adaptive_threshold_algorithm(AV_brightness_weight, brightness_threshold_tr)
    if Ad_thresh < brightness_threshold_tr:
        threshold = brightness_threshold_tr
    else:
        threshold = Ad_thresh
    print(f"threshold: {threshold}")
    # 步骤三，特征提取与可视化
    #plt.figure(figsize = (8,3))
    sc1 = Figure(figsize=(8, 4), dpi=92)
    # sc1 = Figure()
    axes1 =sc1.add_subplot(111)
    X = list(range(len(AV_brightness_weight))) 

    axes1.plot(X, AV_brightness_weight, color = "g", label = "Weighted Neighborhood Average Brightness of Trajectory point")
    axes1.axhline(y=threshold, color='b', linestyle='--', label=f'Aadptive Threshold')
    axes1.set_xlabel("Parameterized Node of Trajectory Point t")
    axes1.set_ylabel("Brightness")
    axes1.set_title("Adptivae hreshold Extraction of Weighted Neighborhood Average Brightness of Trajectory Point")
    axes1.legend(loc = "lower left")

    # 将图像保存到内存缓冲区
    #buffer = BytesIO()
    #plt.savefig(buffer, format='png', dpi=100)
    #buffer.seek(0)

    # 读取缓冲区数据到 OpenCV 格式（BGR）
    # ad_thresh_image_array = np.frombuffer(buffer.getvalue(), dtype=np.uint8)
    # ad_thresh_image = cv2.imdecode(ad_thresh_image_array, cv2.IMREAD_COLOR)

    # 将图像上的点基于单应矩阵转换到世界坐标系下
    def image_to_world(image_point):
        # H =[[ 6.74796726e-03, -2.18825058e-04, -2.10950619e+00],
        # [ 1.19355280e-04,  5.55252687e-03, -1.33755865e+00],
        # [ 1.00979444e-05, -7.56087199e-06,  1.00000000e+00]]
        H = [[ 6.71436311e-03, -2.31093341e-04, -2.08467905e+00],
            [ 1.24171475e-04,  5.52040024e-03, -1.33115882e+00],
            [ 9.50420329e-06, -1.14737867e-05,  1.00000000e+00]]
        image_point_homogeneous = np.append(image_point, 1)
        world_point_homogeneous = np.dot(H, image_point_homogeneous)
        world_point = world_point_homogeneous[:2] / world_point_homogeneous[2]
        return world_point


    image_seg = image.copy()
    h, _ = image_seg.shape[:2]
    # 可视化+参数统计
    num_liquid = 0
    length = 0
    length_world = 0
    begin_interval_point = True
    touch_the_end_flag = False
    xp,yp = spline_points[0]
    xp_world, yp_world = image_to_world([xp,yp])
    for point, j in zip(spline_points, AV_brightness_weight):
        if j > threshold:
            num_liquid += 1
            x, y = int(round(point[0])), int(round(point[1]))
            x_world, y_world = image_to_world([x,y])
            if abs(y-h) < 20: 
                touch_the_end_flag = True
            cv2.circle(image_seg, (x, y), radius=5, color=(0, 255, 0), thickness=-1)  # 标记点
            if not begin_interval_point:
                distance = np.sqrt((x-xp)**2+(y-yp)**2)
                distance_world = np.sqrt((x_world-xp_world)**2+(y_world-yp_world)**2)
                length = length + distance
                length_world = length_world + distance_world
            begin_interval_point = False
            xp,yp = x,y
            xp_world, yp_world = x_world,y_world
        else:
            begin_interval_point = True

    
    return image_seg, sc1, length, length_world


def generate_world_points(pattern_size, circle_spacing):
    """生成3D世界坐标"""
    objp = np.zeros((np.prod(pattern_size), 3), np.float32) #np.prod(pattern_size)表示pattern_size的元素乘积，生成一个二维数组，每行有三列
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) # 将之前二维数组没一行的前两个元素变成对应x，y的坐标，reshape(-1, 2)表示将二维数组变成一维数组，-1表示不限制行数，2表示每行有两个元素
    objp[:, :2] *= circle_spacing
    return objp

def enhance_image(image_path):
    # 读取图像
    img = cv2.imread(image_path)
    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 应用高斯模糊
    blurred = cv2.GaussianBlur(gray, (5, 5), 0) 
    # 直方图均衡化
    equalized = cv2.equalizeHist(blurred)  
    # 边缘增强（使用Laplacian算子）
    laplacian = cv2.Laplacian(equalized, cv2.CV_64F)
    laplacian = cv2.convertScaleAbs(laplacian)  # 转换回合适的图像格式
    # 组合原图和边缘增强图像
    alpha = 1.5  # 控制原始图像的影响
    beta = 0.5   # 控制边缘增强的影响
    enhanced = cv2.addWeighted(equalized, alpha, laplacian, beta, 0)
    return enhanced

def distort_image(image, mtx, dist):
    """
    用于矫正畸变
    """
    mtx = mtx
    dist =  dist
    mtx = np.array(mtx)
    dist = np.array(dist)
    h, w = image.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    dst = cv2.undistort(image, mtx, dist, None, newcameramtx)
    # 裁剪图像以去除黑色区域
    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]
    return dst

def is_in_line(p1, p2, threshold=20):
    """判断两个点是否在同一行或同一列，使用阈值来处理浮点数误差"""
    return abs(p1[1] - p2[1]) < threshold  # 这里以y坐标判断是否在同一行，可根据需要调整

def extract_specific_distribution(circles, desire_config):
    """从给定的圆心列表中提取符合特定分布的圆心集合"""
    config = desire_config.copy()
    x_tolerance = 25  # 定义允许的x值偏差
    y_threshold = 15  # 定义允许的y值偏差
    # 先对所有点按y坐标进行排序
    rows = []
    
    # 首先对所有点按y坐标进行排序
    sorted_circles = sorted(circles, key=lambda x: x[1])
    # 现根据y分行，再根据x筛选行
    rows = []
    current_row = []
    current_y = sorted_circles[0][1]

    for center in sorted_circles:
        if abs(center[1] - current_y) <= y_threshold:
            current_row.append(center)
            current_y = center[1]
        else:
            if current_row:
                rows.append(current_row)
            current_row = [center]
            current_y = center[1]
    # 添加最后一行
    if current_row:
        rows.append(current_row)

    # 筛选符合规则的行，并按X坐标排序
    specific_rows = []
    for row in rows:
        if len(row) in config:
            sorted_row = sorted(row, key=lambda x: x[0])
            if not specific_rows or abs(sorted_row[0][0] - specific_rows[-1][0][0]) <= x_tolerance:
                specific_rows.append(sorted_row)
                config.remove(len(row))
    return specific_rows



def construct_coordinates(shape, distance):
    coordinates = []
    for y, count in enumerate(shape):
        for x in range(count):
            coordinates.append([x * distance, y * distance])
    return np.array(coordinates)


def homography_matrix_cal(image, mtx, dist, desired_config, circle_distance):
    """
    根据图像中已知位置的标记点与畸变系数进行单应矩阵的计算
    image：带有标记点的原图
    distortion_paramer:畸变系数
    point_distribution：图像中特征点的分布
    circle_distance: 世界坐标系下特征点的真实圆心距
    """
    # Step 1 畸变矫正
    dst = distort_image(image.copy(), mtx, dist)
    # 图像前处理
    gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    blurred_inv = cv2.bitwise_not(blurred)
    # 霍夫圆检测
    circles = cv2.HoughCircles(blurred_inv, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=30, minRadius=5, maxRadius=40) 
    # 确保至少检测到一些圆
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
    # 绘制检测到的圆
    # for (x, y, r) in circles:
    #     cv2.circle(image, (x, y), r, (0, 255, 0), 4) 
    # 提取特定pattern的圆
    # pattern rule:
    # desired_config = [8, 1, 1, 8]
    desired_config = desired_config 
    # Step3: 提取特定分布的圆：
    # 提取符合分布要求的圆心集合
    specific_rows = extract_specific_distribution(circles, desired_config)
    dst_ = dst.copy()
    for row in specific_rows:
        for (x, y, r) in row:
            cv2.circle(dst_, (x, y), r, (0, 255, 0), 2)
    # 求解单应矩阵
    # 构造世界坐标系下的圆心坐标
    image_points = np.concatenate(specific_rows, axis = 0)
    image_points = np.array([[x,y] for (x,y,r) in image_points])
    world_points = construct_coordinates(desired_config, circle_distance)
    H, status = cv2.findHomography(image_points, world_points)
    return H, dst_