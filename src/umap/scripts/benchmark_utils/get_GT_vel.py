"""
python src/umap/scripts/benchmark_utils/get_GT_vel.py -i input.csv -o output.csv --dt 0.05
"""

import pandas as pd
import numpy as np
import argparse
from scipy.interpolate import interp1d

def calculate_velocity_from_data(data, dt=0.1):
    """
    从轨迹数据计算每个时刻的速度
    使用固定时间间隔dt来计算速度，并对中间点进行插值
    
    Args:
        data: 包含time, pos_x, pos_y, pos_z列的DataFrame
        dt: 计算速度的时间间隔（秒），默认0.1秒
    """
    # 确保数据按时间排序
    data = data.sort_values('time')
    
    # 转换时间戳为相对秒数（从第一个时间点开始）
    time_original = data['time'].values - data['time'].iloc[0]
    
    # 创建均匀的时间网格，间隔为dt
    time_start = time_original[0]
    time_end = time_original[-1]
    time_uniform = np.arange(time_start, time_end + dt, dt)
    
    # 如果最后一个点没有被包含，添加原始数据的最后时间点
    if time_uniform[-1] < time_end:
        time_uniform = np.append(time_uniform, time_end)
    
    # 对位置数据进行插值到均匀时间网格
    interp_x = interp1d(time_original, data['pos_x'].values, kind='linear', 
                        bounds_error=False, fill_value='extrapolate')
    interp_y = interp1d(time_original, data['pos_y'].values, kind='linear',
                        bounds_error=False, fill_value='extrapolate')
    interp_z = interp1d(time_original, data['pos_z'].values, kind='linear',
                        bounds_error=False, fill_value='extrapolate')
    
    pos_x_uniform = interp_x(time_uniform)
    pos_y_uniform = interp_y(time_uniform)
    pos_z_uniform = interp_z(time_uniform)
    
    # 在均匀时间网格上计算速度
    vx_uniform = np.zeros(len(time_uniform))
    vy_uniform = np.zeros(len(time_uniform))
    vz_uniform = np.zeros(len(time_uniform))
    
    # 使用中心差分法计算速度（除了端点）
    for i in range(1, len(time_uniform) - 1):
        dt_actual = time_uniform[i+1] - time_uniform[i-1]  # 2*dt for central difference
        vx_uniform[i] = (pos_x_uniform[i+1] - pos_x_uniform[i-1]) / dt_actual
        vy_uniform[i] = (pos_y_uniform[i+1] - pos_y_uniform[i-1]) / dt_actual
        vz_uniform[i] = (pos_z_uniform[i+1] - pos_z_uniform[i-1]) / dt_actual
    
    # 端点使用前向/后向差分
    if len(time_uniform) > 1:
        # 第一个点：前向差分
        dt_forward = time_uniform[1] - time_uniform[0]
        vx_uniform[0] = (pos_x_uniform[1] - pos_x_uniform[0]) / dt_forward
        vy_uniform[0] = (pos_y_uniform[1] - pos_y_uniform[0]) / dt_forward
        vz_uniform[0] = (pos_z_uniform[1] - pos_z_uniform[0]) / dt_forward
        
        # 最后一个点：后向差分
        dt_backward = time_uniform[-1] - time_uniform[-2]
        vx_uniform[-1] = (pos_x_uniform[-1] - pos_x_uniform[-2]) / dt_backward
        vy_uniform[-1] = (pos_y_uniform[-1] - pos_y_uniform[-2]) / dt_backward
        vz_uniform[-1] = (pos_z_uniform[-1] - pos_z_uniform[-2]) / dt_backward
    
    # 将速度插值回原始时间点
    if len(time_uniform) > 1:
        interp_vx = interp1d(time_uniform, vx_uniform, kind='linear',
                            bounds_error=False, fill_value='extrapolate')
        interp_vy = interp1d(time_uniform, vy_uniform, kind='linear',
                            bounds_error=False, fill_value='extrapolate')
        interp_vz = interp1d(time_uniform, vz_uniform, kind='linear',
                            bounds_error=False, fill_value='extrapolate')
        
        vx = interp_vx(time_original)
        vy = interp_vy(time_original)
        vz = interp_vz(time_original)
    else:
        vx = np.zeros(len(data))
        vy = np.zeros(len(data))
        vz = np.zeros(len(data))
    
    # 计算速度大小（标量速度）
    speed = np.sqrt(vx**2 + vy**2 + vz**2)
    
    return vx, vy, vz, speed

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate velocity from trajectory data')
    parser.add_argument('--input', '-i', type=str, required=True, help='Input CSV file path')
    parser.add_argument('--output', '-o', type=str, help='Output CSV file path (optional)')
    parser.add_argument('--dt', type=float, default=0.1, 
                       help='Time interval for velocity calculation (seconds), default=0.1')

    args = parser.parse_args()

    # 读取数据
    data = pd.read_csv(args.input)

    # 计算速度
    vx, vy, vz, speed = calculate_velocity_from_data(data, dt=args.dt)

    # 创建输出数据框
    result_data = data.copy()
    result_data['vx'] = vx
    result_data['vy'] = vy
    result_data['vz'] = vz
    result_data['speed'] = speed

    print("Velocity calculation completed!")
    print(f"Data shape: {result_data.shape}")
    print(f"Time interval (dt): {args.dt} seconds")
    print("\nFirst 5 rows:")
    print(result_data.head())
    print(f"\nVelocity statistics:")
    print(f"Speed - Mean: {speed.mean():.4f}, Std: {speed.std():.4f}, Max: {speed.max():.4f}")
    
    # 保存结果
    if args.output:
        result_data.to_csv(args.output, index=False)
        print(f"\nResults saved to: {args.output}")
    else:
        # 如果没有指定输出文件，创建一个默认名称
        import os
        input_dir = os.path.dirname(args.input)
        input_name = os.path.splitext(os.path.basename(args.input))[0]
        output_file = os.path.join(input_dir, f"{input_name}_with_velocity.csv")
        result_data.to_csv(output_file, index=False)
        print(f"\nResults saved to: {output_file}")