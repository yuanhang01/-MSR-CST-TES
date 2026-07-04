"""
电力/热力需求曲线与 DNI 数据加载器
基于 Bayomy & Moore (2020) Section 6, Figure 2, 6, 7
可加载外部CSV文件，也可生成合成数据用于测试
"""

import numpy as np
import os

# ============================================================================
# 合成数据生成器（当没有实际数据文件时使用）
# ============================================================================

def generate_ottawa_dni(n_hours: int = 8760, seed: int = 42) -> np.ndarray:
    """
    生成渥太华地区的合成 DNI 数据 [W/m²]
    近似于文献中 Figure 2 的年度变化规律

    特点：
    - 夏季 (6-8月) DNI 较高，冬季较低
    - 每天白天有日照（约10-16h时段），夜间为0
    - 包含云量随机性
    """
    np.random.seed(seed)
    dni = np.zeros(n_hours)

    for t in range(n_hours):
        day_of_year = (t // 24)  # 0-364
        hour_of_day = t % 24

        # 季节因子 (北半球，6月21日=夏至, day ~172)
        seasonal_factor = 0.5 + 0.5 * np.cos(2 * np.pi * (day_of_year - 172) / 365.0)

        # 日间日照 (10:00-18:00 峰值)
        if 6 <= hour_of_day <= 20:
            hour_factor = np.sin(np.pi * (hour_of_day - 6) / 14.0)
        else:
            hour_factor = 0.0

        # 基础辐照
        base_dni = 800.0 * seasonal_factor * hour_factor

        # 云量随机扰动
        cloud_factor = np.random.beta(2, 1.5)  # 偏向晴天的分布
        dni[t] = base_dni * cloud_factor

        # 完全阴天概率
        if np.random.random() < 0.15 * (1 - seasonal_factor):
            dni[t] *= np.random.uniform(0.05, 0.3)

    return dni


def generate_ontario_electric_demand(n_hours: int = 8760, seed: int = 42,
                                      base_load_MW: float = 90.0) -> np.ndarray:
    """
    生成安大略省的合成电力需求曲线 [MW]
    近似于文献中 Figure 6 的变化规律

    特点：
    - 冬季较高（供暖），夏季中等（空调），春秋较低
    - 日内双峰：上午10-12点和下午5-7点
    - 夜间低谷
    """
    np.random.seed(seed)
    demand = np.zeros(n_hours)

    for t in range(n_hours):
        day_of_year = t // 24
        hour_of_day = t % 24

        # 季节因子：冬季最高，夏季次之
        winter_factor = 0.5 + 0.5 * np.cos(2 * np.pi * (day_of_year - 15) / 365.0)
        summer_factor = 0.3 + 0.3 * np.cos(2 * np.pi * (day_of_year - 200) / 365.0)
        seasonal_factor = max(winter_factor, summer_factor * 0.85)

        # 日内双峰模式
        morning_peak = 0.7 * np.exp(-((hour_of_day - 10) / 2.5) ** 2)
        evening_peak = 0.9 * np.exp(-((hour_of_day - 18) / 3.0) ** 2)
        midday = 0.6 * np.exp(-((hour_of_day - 14) / 4.0) ** 2)
        night_base = 0.35 * (1 - np.exp(-(hour_of_day / 6.0)**2)) + 0.25 * np.exp(-((hour_of_day-24)/3.0)**2)

        daily_factor = morning_peak + evening_peak + midday + night_base
        daily_factor = daily_factor / np.max([morning_peak, evening_peak, midday, night_base]) * 0.7 + 0.3

        # 周末因子
        is_weekend = (day_of_year % 7) >= 5
        weekend_factor = 0.85 if is_weekend else 1.0

        # 合成需求
        demand[t] = base_load_MW * (0.6 + 0.4 * seasonal_factor) * daily_factor * weekend_factor

        # 随机噪声
        demand[t] *= np.random.normal(1.0, 0.03)

    return demand


def generate_residential_heat_demand(n_hours: int = 8760, seed: int = 42,
                                      base_load_MW: float = 15.0) -> np.ndarray:
    """
    生成住宅供暖热需求曲线 [MW]
    近似于文献中 Figure 7 的变化规律

    特点：
    - 冬季需求高，夏季几乎为0
    - 早晨和晚上高峰
    - 取决于室外温度
    """
    np.random.seed(seed)
    demand = np.zeros(n_hours)

    for t in range(n_hours):
        day_of_year = t // 24
        hour_of_day = t % 24

        # 室外温度近似 (°C)
        T_outside = -10.0 + 20.0 * np.cos(2 * np.pi * (day_of_year - 15) / 365.0)

        # 供暖需求与温差成正比
        T_indoor = 20.0
        heating_need = max(0, T_indoor - T_outside)

        # 日内模式：早晚高峰
        daily_factor = 0.6
        morning_factor = 0.4 * np.exp(-((hour_of_day - 7) / 2.0) ** 2)
        evening_factor = 0.5 * np.exp(-((hour_of_day - 19) / 2.5) ** 2)
        daily_factor += morning_factor + evening_factor

        # 工作日/周末差异
        is_weekend = (day_of_year % 7) >= 5
        weekend_factor = 1.1 if is_weekend else 1.0

        demand[t] = base_load_MW * (heating_need / 25.0) * daily_factor * weekend_factor

        # 随机噪声
        demand[t] *= np.random.normal(1.0, 0.05)
        demand[t] = max(0.0, demand[t])

    return demand


def generate_ambient_temperature(n_hours: int = 8760, seed: int = 42) -> np.ndarray:
    """
    生成环境温度曲线 [°C]
    用于 CST 计算的 T_amb 输入
    """
    np.random.seed(seed)
    T = np.zeros(n_hours)

    for t in range(n_hours):
        day_of_year = t // 24
        hour_of_day = t % 24

        # 季节性：正弦波，最冷1月15日(day~15)，最热7月15日(day~196)
        T_seasonal = 5.0 + 15.0 * np.sin(2 * np.pi * (day_of_year - 105) / 365.0)

        # 日内变化：下午最热(14-15h)，凌晨最冷(4-5h)
        T_daily = 5.0 * np.sin(2 * np.pi * (hour_of_day - 14) / 24.0)

        T[t] = T_seasonal + T_daily + np.random.normal(0, 2.0)

    return T


# ============================================================================
# 数据加载器
# ============================================================================

def load_or_generate_dni(filepath: str = None, n_hours: int = 8760,
                          use_synthetic: bool = False) -> np.ndarray:
    """
    加载 DNI 数据，若文件不存在则生成合成数据
    """
    if filepath and os.path.exists(filepath) and not use_synthetic:
        try:
            data = np.loadtxt(filepath, delimiter=',', skiprows=1)
            if data.ndim == 1:
                return data
            # 假设第二列为 DNI
            return data[:, 1] if data.shape[1] > 1 else data[:, 0]
        except Exception as e:
            print(f"警告: 无法加载 {filepath}: {e}，使用合成数据。")
    return generate_ottawa_dni(n_hours)


def load_or_generate_electric_demand(filepath: str = None, n_hours: int = 8760,
                                      use_synthetic: bool = False,
                                      base_load_MW: float = 90.0) -> np.ndarray:
    """
    加载电力需求数据
    """
    if filepath and os.path.exists(filepath) and not use_synthetic:
        try:
            data = np.loadtxt(filepath, delimiter=',', skiprows=1)
            if data.ndim == 1:
                return data
            return data[:, 1] if data.shape[1] > 1 else data[:, 0]
        except Exception as e:
            print(f"警告: 无法加载 {filepath}: {e}，使用合成数据。")
    return generate_ontario_electric_demand(n_hours, base_load_MW=base_load_MW)


def load_or_generate_heat_demand(filepath: str = None, n_hours: int = 8760,
                                  use_synthetic: bool = False,
                                  base_load_MW: float = 15.0) -> np.ndarray:
    """
    加载热需求数据
    """
    if filepath and os.path.exists(filepath) and not use_synthetic:
        try:
            data = np.loadtxt(filepath, delimiter=',', skiprows=1)
            if data.ndim == 1:
                return data
            return data[:, 1] if data.shape[1] > 1 else data[:, 0]
        except Exception as e:
            print(f"警告: 无法加载 {filepath}: {e}，使用合成数据。")
    return generate_residential_heat_demand(n_hours, base_load_MW=base_load_MW)


def load_or_generate_ambient_temperature(filepath: str = None, n_hours: int = 8760,
                                          use_synthetic: bool = False) -> np.ndarray:
    """
    加载环境温度数据
    """
    if filepath and os.path.exists(filepath) and not use_synthetic:
        try:
            data = np.loadtxt(filepath, delimiter=',', skiprows=1)
            if data.ndim == 1:
                return data
            return data[:, 1] if data.shape[1] > 1 else data[:, 0]
        except Exception as e:
            print(f"警告: 无法加载 {filepath}: {e}，使用合成数据。")
    return generate_ambient_temperature(n_hours)


def print_data_statistics(dni: np.ndarray, P_demand: np.ndarray,
                           H_demand: np.ndarray, T_amb: np.ndarray):
    """打印数据统计信息"""
    print("\n" + "=" * 60)
    print("  输入数据统计")
    print("=" * 60)
    data_sets = [
        ("DNI [W/m²]", dni),
        ("电力需求 [MW]", P_demand),
        ("热需求 [MW]", H_demand),
        ("环境温度 [°C]", T_amb),
    ]
    for name, data in data_sets:
        print(f"  {name:<20}  "
              f"均值={np.mean(data):.2f}  "
              f"最大={np.max(data):.2f}  "
              f"最小={np.min(data):.2f}  "
              f"总和={np.sum(data):.2f}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    # 测试合成数据生成
    n = 8760
    dni = generate_ottawa_dni(n)
    P = generate_ontario_electric_demand(n)
    H = generate_residential_heat_demand(n)
    T = generate_ambient_temperature(n)
    print_data_statistics(dni, P, H, T)