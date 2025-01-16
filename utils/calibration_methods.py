# @CreateTime: Jan 13, 2025 8:02 PM 
# @Author: Howard 
# @Contact: wangh22@mails.tsinghua.edu.cn 
# @Last Modified By: Howard
# @Last Modified Time: Jan 13, 2025 8:19 PM
# @Description: Modify Here, Please 

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
 
def calibration_process(excel_path, liqid_length_col = "F", balance_col = "J", touch_flag = "H"):
    real_weight_read = pd.read_excel(excel_path, usecols=balance_col, skiprows=1).to_numpy().flatten()
    x2_read = pd.read_excel(excel_path, usecols=liqid_length_col, skiprows=1).to_numpy().flatten()
    touch_flag = pd.read_excel(excel_path, usecols=touch_flag, skiprows=1).to_numpy().flatten()
    # print(x2_read)
    # print(touch_flag)
    # print(touch_flag[:-1])
    mask = [ele == 0 for ele in touch_flag]
    
    filtered_real_weight = real_weight_read[:-1][mask[:-1]]
    filtered_x2 = x2_read[:-1][mask[:-1]]
    # 数据赋值
    x2 = filtered_x2
    real_weight = filtered_real_weight

    # 线性拟合
    p = np.polyfit(x2, real_weight, 1)
    y_fit = np.polyval(p, x2)

    #print(f"Linear prediction model: y = {p[0]:.5f}x + {p[1]:.5f}")
    # 可决系数 R²
    y_mean = np.mean(real_weight)
    ss_tot = np.sum((real_weight - y_mean) ** 2)
    ss_res = np.sum((real_weight - y_fit) ** 2)
    R2 = 1 - (ss_res / ss_tot)
    #print(f"Calibration measurement R²: {R2:.5f}")
    # 绝对误差
    abs_errors = np.abs(y_fit - real_weight)
    #print(f"Maximum absolute error: {np.max(abs_errors):.5f}")
    #print(f"Mean absolute error (MAE): {np.mean(abs_errors):.5f}")
    # 相对误差
    relative_errors = np.abs((y_fit - real_weight) / real_weight)
    var_cali = np.var(relative_errors * 100)
    #print(f"Variance of absolute percentage errors: {var_cali:.5f}")
    max_relative_error = np.max(relative_errors)
    #print(f"Maximum relative error (MRE): {max_relative_error * 100:.5f}%")
    mean_relative_error = np.mean(relative_errors * 100)
    #print(f"Mean absolute percentage error (MAPE): {mean_relative_error:.5f}%")

    # 均方根误差 (RMSE)
    rmse = np.sqrt(np.mean((real_weight - y_fit) ** 2))
    #print(f"Root mean square error (RMSE): {rmse:.5f}")
    model_str = f"""
        Linear prediction model: y = {p[0]:.5f}x + {p[1]:.5f};
        Calibration measurement R²: {R2:.5f};
        Maximum absolute error: {np.max(abs_errors):.5f};
        Mean absolute error (MAE): {np.mean(abs_errors):.5f};
        Variance of absolute percentage errors: {var_cali:.5f};
        Maximum relative error (MRE): {max_relative_error * 100:.5f}%;
        Mean absolute percentage error (MAPE): {mean_relative_error:.5f}%
        """ 

    # 分段计算均值和标准误差
    num_bins = 20
    bin_edges = np.linspace(np.min(real_weight), np.max(real_weight), num_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_means = []
    bin_errors = []

    for i in range(num_bins):
        bin_data = y_fit[(real_weight >= bin_edges[i]) & (real_weight < bin_edges[i + 1])]
        if len(bin_data) > 0:
            bin_means.append(np.mean(bin_data))
            bin_errors.append(np.std(bin_data) / np.sqrt(len(bin_data)))
        else:
            bin_means.append(np.nan)
            bin_errors.append(np.nan)

    bin_means = np.array(bin_means)
    bin_errors = np.array(bin_errors)
    """
    返回值： 
    * model_str 模型的字符描述
    * x2: liquid pixel length
    * real weight：天平称重的结果
    * y_fit: 拟合的线性体积
    * bin_center: 误差带图x坐标
    * bin_means: 误差带图y坐标
    * bin_errors: 误差带图的误差分布
    * R2：线性拟合的可决系数
    * 
    
    """
    return model_str, x2, real_weight, y_fit, bin_centers, bin_means, bin_errors, R2
 

