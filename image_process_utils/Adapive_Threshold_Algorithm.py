# @CreateTime: Dec 13, 2024 1:14 PM 
# @Author: Howard 
# @Contact: wangh22@mails.tsinghua.edu.cn 
# @Last Modified By: Howard
# @Last Modified Time: Dec 13, 2024 2:43 PM
# @Description: Modify Here, Please 

import pandas as pd
from matplotlib import font_manager
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os

# 设置中文字体
plt.rcParams['font.family'] = ['SimHei']  # 例如使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号 '-' 显示为方块的问题

def moving_average_filter(data, window_size):
    """
    对输入的数据进行移动平均滤波。
    
    参数:
    data (list of float): 输入数据列表。
    window_size (int): 滤波窗口的大小。
    
    返回:
    list of float: 滤波后的数据列表。
    """
    filtered_data = []
    for i in range(len(data)):
        # 确定滑动窗口的边界
        start_index = max(i - window_size // 2, 0)
        end_index = min(i + window_size // 2 + 1, len(data))
        # 计算窗口内数据的平均值
        window_average = sum(data[start_index:end_index]) / (end_index - start_index)
        filtered_data.append(window_average)
    return filtered_data

def find_peaks_with_threshold(lst, threshold):
    """
    寻找输入数据中大于阈值的峰值。
    参数:
    lst (list of float): 输入数据列表。
    threshold (float): 阈值。
    返回:
    list of int: 峰值索引列表。
    list of float: 峰值列表。
    """
    # 存储结果的列表，元素格式为(index, peak_value)
    index = []
    result = []

    # 遍历列表，寻找峰值
    for i in range(1, len(lst) - 1):
        # 判断正峰值
        if lst[i] > lst[i - 1] and lst[i] > lst[i + 1] and lst[i] > threshold:
            index.append(i)
            result.append(lst[i])
        # 判断负峰值
        elif lst[i] < lst[i - 1] and lst[i] < lst[i + 1] and abs(lst[i]) > threshold:
            index.append(i)
            result.append(lst[i])

    return index, result

def find_intervals(index,data):
    """
    找到符合要求的液段区间，这种区间的特征是其梯度先是负峰值，再是正峰值
    """
    intervals = []
    i = 0
    while i < len(data):
        start, end = None, None
        # 寻找区间开始（正值）
        while i < len(data) and data[i] <= 0:
            i += 1
        if i < len(data):
            start = i
        
        # 寻找区间结束（负值）
        while i < len(data) and data[i] >= 0:
            i += 1
        if i < len(data):
            end = i
        
        # 如果找到有效的区间，则添加到列表中
        if start is not None and end is not None and index[end] > index[start]:
            print(f"start:{index[start]}; end:{index[end]}")
            intervals.append((index[start]+7, index[end]-3)) #缩小区间，以避免边界会产生的一些问题 会由于一些很短的气段产生问题
            # intervals.append((index[start], index[end])) #缩小区间，以避免边界会产生的一些问题 会由于一些很短的气段产生问题
            i += 1  # 从下一个点开始继续寻找新的区间
        else:
            # 如果未找到完整区间，则结束循环
            continue
    return intervals

def find_peaks_value(data, intervals):
    """
    用于寻找峰值并以一个阈值进行过滤
    """
    peaks = []
    for start_idx, end_idx in intervals:
        # 区间是基于索引的，所以我们使用索引来获取实际的数据点
        interval_data = data[start_idx:end_idx+1]  # 包含结束索引的数据 # 可以改进为子串的最小值
        peak_value = min(interval_data)  # 根据值找到峰值
        peaks.append(peak_value)
    data_bottom = min(peaks)
    data_up = min(data[0:30])
    #threshold = (data_up+data_bottom)/2
    threshold = data_bottom-2
    return threshold

def adaptive_threshold_algorithm(AV_B_list:list, default_threshold)->float:
    """
    Used to find out the most suitable threshold for segment the air district and the liquid district
    Input: the weighted average brightness calculated by each point's tangent and radiant domain
    output: the threshold
    """
    try:
        av_data = moving_average_filter(AV_B_list,10) # 以10个点为一组做一次移动滤波
        # plt.figure(figsize =(10,5))
        # plt.plot(x,data_list_from_excel)
        # plt.plot(x,av_data)
        av_data_np = np.array(av_data)
        gradient = np.zeros_like(av_data)
        gradient[:-1] = av_data_np[1:] - av_data_np[:-1]
        index,peaks = find_peaks_with_threshold(gradient,5) # 对斜率进行一次寻峰与滤波
        # plt.figure(figsize =(10,5))
        # plt.plot(x,gradient)
        # plt.plot(index,peaks,'.')
        intervals = find_intervals(index, peaks) #找出空气区域的区间
        #intervals = adjust_intervals(intervals, 8, AV_B_list)
        threshold = find_peaks_value(AV_B_list, intervals) #以空气区间的最大值与起始区间的最小值的平均值作为阈值
        print(f"the threshold of the adaptive algorithm is {threshold}")
        
    except:
        threshold = default_threshold
    return threshold, intervals


if __name__ == "__main__":
    imagefile_directory = "D:\Onedrive-University of Cincinnati\OneDrive - University of Cincinnati\Desktop\Yunjing\Vision_MicroFluid_Measurement\Data\Data_2024-06-23" 
    data_num = 12
    excel_pth = os.path.join(imagefile_directory,f"{data_num}\data.xlsx")
    df = pd.read_excel(excel_pth)
    # 将DataFrame中的数据导出为列表
    data_list_from_excel = df['Numbers'].tolist()
    brightness_threshold_tr = max(data_list_from_excel)-70 # 阈值下限
    x = [i for i in range(len(data_list_from_excel))]
    Ad_thresh, intervals = adaptive_threshold_algorithm(data_list_from_excel, brightness_threshold_tr)
    print(Ad_thresh)
    print(intervals)
    

    plt.figure(figsize =(10,5))
    plt.plot(x,data_list_from_excel)
    for intervel in intervals:
        plt.plot(intervel[0],data_list_from_excel[intervel[0]],'.', color = "green")
        plt.plot(intervel[1],data_list_from_excel[intervel[1]],'.', color = "red")
    plt.plot(x, Ad_thresh*np.ones(len(data_list_from_excel)), label = "自适应阈值")
    plt.title("轨迹点径向与切向领域加权平均亮度自适应阈值示意图")
    plt.xlabel("轨迹点序号")
    plt.ylabel("径向与切向加权平均亮度")
    plt.legend()
    plt.show()


