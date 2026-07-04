"""
聚光太阳能塔 (CST) 模型
基于 Bayomy & Moore (2020) Section 4, 方程 (5)-(19)
参考 Gemasolar 设计参数 (Table 4, Table 5)
FT (固定出口温度) 控制策略
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fluid_library import StorageFluid


class ConcentratedSolarTower:
    """
    聚光太阳能塔模型
    包含定日镜场、腔式接收器的能量平衡计算
    实现方程 (5)-(19)
    """

    # 常量
    SIGMA = 5.67e-8  # Stefan-Boltzmann 常数 [W/m²·K⁴]

    def __init__(self, config: dict, storage_fluid: StorageFluid = None):
        """
        Parameters
        ----------
        config : dict
            config.yaml 中 'cst' 部分的配置
        storage_fluid : StorageFluid
            储热流体实例
        """
        # 定日镜场参数
        self.A_field_m2 = config.get('field_area_m2', 150000.0)
        self.C_ratio = config.get('concentration_ratio', 900.0)
        self.eta_field = config.get('field_efficiency', 0.60)

        # 接收器参数
        self.epsilon = config.get('receiver_emissivity', 0.88)
        self.F_view = config.get('view_factor', 0.80)
        self.rho_reflect = config.get('reflectivity', 0.06)
        self.D_tube_m = config.get('tube_outer_diameter_m', 0.04)
        self.k_tube_WpmK = config.get('tube_thermal_conductivity_WpmK', 23.9)
        self.v_wind_mps = config.get('wind_velocity_mps', 7.0)
        self.T_amb_design_C = config.get('ambient_temperature_C', 25.0)
        self.v_max_fluid_mps = config.get('max_fluid_velocity_mps', 4.0)

        # 控制策略
        self.control_strategy = config.get('control_strategy', 'FT')
        self.T_receiver_in_C = config.get('receiver_inlet_temperature_C', 290.0)
        self.T_receiver_out_set_C = config.get('receiver_outlet_temperature_C', 565.0)
        self.DNI_threshold_Wpm2 = config.get('dni_threshold_Wpm2', 350.0)

        # 存储流体
        self.fluid = storage_fluid

        # 接收器几何参数计算
        self._calculate_geometry()

        # 空气物性（25°C, 1 atm 参考值）
        self._init_air_properties()

    def _calculate_geometry(self):
        """
        计算接收器几何参数
        方程 (12): A_rec,ap = C × A_field
        方程 (13): A_rec,surface = A_rec,ap / F
        """
        # 方程 (12): 接收器孔径面积
        self.A_rec_ap_m2 = self.A_field_m2 / self.C_ratio

        # 方程 (13): 接收器表面积
        self.A_rec_surface_m2 = self.A_rec_ap_m2 / self.F_view

        # 接收器特征高度（假设方形孔径）
        self.H_rec_m = np.sqrt(self.A_rec_ap_m2)

        # 等效平均发射率 (方程 11)
        self.epsilon_avg = self.epsilon / (self.epsilon + (1 - self.epsilon) * self.F_view)

        # 接收管数量估算
        tube_circumference = np.pi * self.D_tube_m
        self.N_tubes = max(1, int(self.A_rec_surface_m2 / (tube_circumference * self.H_rec_m)))
        self.N_tubes = config_correction_tubes(self.N_tubes)  # 使用默认值调整

    def _init_air_properties(self):
        """初始化空气物性"""
        # 25°C, 1 atm
        self.rho_air_kgpm3 = 1.184  # 空气密度 [kg/m³]
        self.mu_air_Pa_s = 1.849e-5  # 空气动力粘度 [Pa·s]
        self.k_air_WpmK = 0.02551   # 空气导热系数 [W/m·K]
        self.Pr_air = 0.7296         # 空气 Prandtl 数
        self.cp_air_JpkgK = 1005.0   # 空气比热 [J/kg·K]

    def _update_air_properties(self, T_amb_C: float):
        """更新空气物性（温度相关）"""
        T_K = T_amb_C + 273.15
        self.rho_air_kgpm3 = 1.184 * (298.15 / T_K)
        self.mu_air_Pa_s = 1.849e-5 * (T_K / 298.15) ** 0.7
        self.k_air_WpmK = 0.02551 * (T_K / 298.15) ** 0.8

    def compute(self, DNI_Wpm2: float, T_amb_C: float,
                T_fluid_in_C: float = None) -> dict:
        """
        计算 CST 在一个时间步长的输出

        Parameters
        ----------
        DNI_Wpm2 : float
            直接法向辐照度 [W/m²]
        T_amb_C : float
            环境温度 [°C]
        T_fluid_in_C : float, optional
            工质入口温度，默认 T_cold_tank

        Returns
        -------
        dict: 包含 Q_abs_MW, m_fluid_kgps, T_fluid_out_C, eta_receiver, Q_losses 等
        """
        if T_fluid_in_C is None:
            if self.fluid is not None:
                T_fluid_in_C = self.fluid.T_cold_tank
            else:
                T_fluid_in_C = self.T_receiver_in_C

        # 更新空气物性
        self._update_air_properties(T_amb_C)

        # 方程 (5): 定日镜场总入射
        Q_field_W = DNI_Wpm2 * self.A_field_m2

        # 方程 (6): 接收器接收热功率
        Q_rec_in_W = Q_field_W * self.eta_field

        # 设定接收器表面温度
        # 在 FT 策略下，表面温度逼近出口温度
        T_rec_surface_K = self.T_receiver_out_set_C + 273.15
        T_amb_K = T_amb_C + 273.15

        # ---- 热损失计算 ----

        # 方程 (10): 辐射损失
        Q_rad_loss_W = (
            self.epsilon_avg * self.SIGMA *
            (T_rec_surface_K**4 - T_amb_K**4) *
            self.A_rec_surface_m2 * self.F_view
        )

        # 方程 (14): 反射损失
        Q_reflection_loss_W = self.rho_reflect * Q_rec_in_W * self.F_view

        # 方程 (15)-(16): 强制对流损失
        Re_air = (self.rho_air_kgpm3 * self.v_wind_mps * self.H_rec_m) / self.mu_air_Pa_s
        h_FC_Wpm2K = 0.0287 * (Re_air**0.8) * (self.Pr_air**(1/3)) * (self.k_air_WpmK / self.H_rec_m)

        # 方程 (17)
        Q_FC_loss_W = h_FC_Wpm2K * (T_rec_surface_K - T_amb_K) * self.A_rec_surface_m2

        # 方程 (18)-(19): 自然对流损失
        delta_T = max(0.1, T_rec_surface_K - T_amb_K)
        h_NC_Wpm2K = 0.81 * (delta_T ** 0.426)
        Q_NC_loss_W = h_NC_Wpm2K * delta_T * self.A_rec_surface_m2

        # 方程 (8): 总热损失
        Q_loss_total_W = Q_rad_loss_W + Q_reflection_loss_W + Q_FC_loss_W + Q_NC_loss_W

        # 方程 (7) 和 (9): 吸收热功率
        Q_abs_W = max(0.0, Q_rec_in_W - Q_loss_total_W)

        # 方程 (9): 接收器效率
        eta_receiver = Q_abs_W / Q_rec_in_W if Q_rec_in_W > 0 else 0.0

        # ---- 工质流量计算 ----

        if self.fluid is not None:
            # 计算将流体从 T_in 加热到 T_out_set 所需的质量流量
            cp_fluid = self.fluid.specific_heat(
                (T_fluid_in_C + self.T_receiver_out_set_C) / 2.0
            )  # kJ/kg·K
            delta_T = self.T_receiver_out_set_C - T_fluid_in_C
            if delta_T > 0 and Q_abs_W > 0:
                m_dot_kgps = (Q_abs_W / 1000.0) / (cp_fluid * delta_T)
            else:
                m_dot_kgps = 0.0

            # 检查最大流速约束
            if m_dot_kgps > 0 and self.fluid is not None:
                rho_fluid = self.fluid.density(
                    (T_fluid_in_C + self.T_receiver_out_set_C) / 2.0
                )
                # 单管截面积
                A_tube_m2 = np.pi * (self.D_tube_m / 2.0) ** 2
                A_total_m2 = A_tube_m2 * self.N_tubes
                v_fluid_mps = m_dot_kgps / (rho_fluid * A_total_m2) if A_total_m2 > 0 else 999
                if v_fluid_mps > self.v_max_fluid_mps:
                    # 限制流量
                    m_dot_kgps = rho_fluid * A_total_m2 * self.v_max_fluid_mps

            T_fluid_out_C = self.T_receiver_out_set_C
        else:
            # 无流体时的简化计算
            m_dot_kgps = 0.0
            T_fluid_out_C = self.T_receiver_out_set_C

        # 实际约束：DNI 低于阈值时流量置零
        if DNI_Wpm2 < self.DNI_threshold_Wpm2:
            m_dot_kgps = 0.0
            Q_abs_W = 0.0
            eta_receiver = 0.0

        # 转换为 MW
        Q_field_MW = Q_field_W / 1e6
        Q_rec_in_MW = Q_rec_in_W / 1e6
        Q_abs_MW = Q_abs_W / 1e6
        Q_loss_MW = Q_loss_total_W / 1e6

        return {
            'DNI_Wpm2': DNI_Wpm2,
            'T_amb_C': T_amb_C,
            'Q_field_MW': Q_field_MW,
            'Q_rec_in_MW': Q_rec_in_MW,
            'Q_abs_MW': Q_abs_MW,
            'Q_loss_total_MW': Q_loss_MW,
            'Q_rad_loss_MW': Q_rad_loss_W / 1e6,
            'Q_reflection_loss_MW': Q_reflection_loss_W / 1e6,
            'Q_FC_loss_MW': Q_FC_loss_W / 1e6,
            'Q_NC_loss_MW': Q_NC_loss_W / 1e6,
            'm_fluid_kgps': m_dot_kgps,
            'T_fluid_in_C': T_fluid_in_C,
            'T_fluid_out_C': T_fluid_out_C,
            'eta_receiver': eta_receiver,
            'eta_field': self.eta_field,
            'N_tubes': self.N_tubes,
        }

    def compute_design_point(self) -> dict:
        """
        计算设计点工况（与 Table 6 对比验证）
        条件：DNI=900 W/m², T_amb=25°C, T_in=290°C
        """
        return self.compute(
            DNI_Wpm2=900.0,
            T_amb_C=self.T_amb_design_C,
            T_fluid_in_C=self.T_receiver_in_C
        )

    def compute_annual_efficiency(self, dni_array: np.ndarray,
                                   T_amb_array: np.ndarray,
                                   T_fluid_in_array: np.ndarray = None) -> float:
        """
        计算年平均接收器效率

        Parameters
        ----------
        dni_array : np.ndarray
            全年 DNI 时间序列 [W/m²]
        T_amb_array : np.ndarray
            全年环境温度时间序列 [°C]
        T_fluid_in_array : np.ndarray, optional
            全年流体入口温度时间序列

        Returns
        -------
        float: 年平均接收器效率
        """
        eta_sum = 0.0
        count = 0
        for t in range(len(dni_array)):
            if dni_array[t] > 0:
                T_in = T_fluid_in_array[t] if T_fluid_in_array is not None else None
                result = self.compute(dni_array[t], T_amb_array[t], T_in)
                eta_sum += result['eta_receiver']
                count += 1
        return eta_sum / count if count > 0 else 0.0

    def get_design_summary(self) -> dict:
        """返回设计参数摘要"""
        return {
            'field_area_m2': self.A_field_m2,
            'concentration_ratio': self.C_ratio,
            'field_efficiency': self.eta_field,
            'receiver_emissivity': self.epsilon,
            'view_factor': self.F_view,
            'A_rec_ap_m2': self.A_rec_ap_m2,
            'A_rec_surface_m2': self.A_rec_surface_m2,
            'N_tubes': self.N_tubes,
            'epsilon_avg': self.epsilon_avg,
            'T_rec_out_set_C': self.T_receiver_out_set_C,
        }

    def print_summary(self):
        """打印设计摘要"""
        s = self.get_design_summary()
        print("\n" + "=" * 60)
        print("  聚光太阳能塔 (CST) — 设计参数")
        print("=" * 60)
        print(f"  定日镜场面积:      {s['field_area_m2']:.0f} m²")
        print(f"  聚光比:            {s['concentration_ratio']:.0f}")
        print(f"  场综合效率:        {s['field_efficiency']:.2f}")
        print(f"  接收器发射率:      {s['receiver_emissivity']:.2f}")
        print(f"  视因子:            {s['view_factor']:.2f}")
        print(f"  接收器孔径面积:    {s['A_rec_ap_m2']:.2f} m²")
        print(f"  接收器表面积:      {s['A_rec_surface_m2']:.2f} m²")
        print(f"  等效发射率:        {s['epsilon_avg']:.4f}")
        print(f"  接收管数量:        {s['N_tubes']}")
        print(f"  设计出口温度:      {s['T_rec_out_set_C']:.0f} °C")
        print("=" * 60 + "\n")


# ============================================================================
# 辅助函数
# ============================================================================

def config_correction_tubes(N: int) -> int:
    """
    根据 Gemasolar 实际数据调整管数
    文献 Table 6 显示 39 根管 (实际) vs 模型预测 33 根
    """
    return N  # 保持模型原始计算

def create_gemasolar_reference(config_override: dict = None) -> ConcentratedSolarTower:
    """
    创建 Gemasolar 参考 CST
    """
    default_config = {
        'field_area_m2': 306658.0,
        'concentration_ratio': 900.0,
        'field_efficiency': 0.60,
        'receiver_emissivity': 0.88,
        'view_factor': 0.80,
        'reflectivity': 0.06,
        'tube_outer_diameter_m': 0.04,
        'tube_thermal_conductivity_WpmK': 23.9,
        'wind_velocity_mps': 7.0,
        'ambient_temperature_C': 25.0,
        'max_fluid_velocity_mps': 4.0,
        'receiver_inlet_temperature_C': 290.0,
        'receiver_outlet_temperature_C': 565.0,
        'control_strategy': 'FT',
        'dni_threshold_Wpm2': 350.0,
    }
    if config_override:
        default_config.update(config_override)
    return ConcentratedSolarTower(default_config)


if __name__ == "__main__":
    # 自检：设计点计算
    cst = create_gemasolar_reference()
    cst.print_summary()

    # 设计点输出
    dp = cst.compute_design_point()
    print("\n--- 设计点 (DNI=900 W/m², T_amb=25°C) ---")
    print(f"  Q_field:          {dp['Q_field_MW']:.1f} MW")
    print(f"  Q_rec_in:         {dp['Q_rec_in_MW']:.1f} MW")
    print(f"  Q_abs:            {dp['Q_abs_MW']:.1f} MW")
    print(f"  Q_loss_total:     {dp['Q_loss_total_MW']:.1f} MW")
    print(f"  Q_rad_loss:       {dp['Q_rad_loss_MW']:.2f} MW")
    print(f"  Q_reflection_loss:{dp['Q_reflection_loss_MW']:.2f} MW")
    print(f"  Q_FC_loss:        {dp['Q_FC_loss_MW']:.3f} MW")
    print(f"  Q_NC_loss:        {dp['Q_NC_loss_MW']:.3f} MW")
    print(f"  eta_receiver:     {dp['eta_receiver']:.4f} ({dp['eta_receiver']*100:.2f}%)")
    print(f"  m_fluid:          {dp['m_fluid_kgps']:.1f} kg/s")
    print(f"  T_fluid_out:      {dp['T_fluid_out_C']:.1f} °C")

    # 变工况测试 (对应 Fig 4)
    print("\n--- 变工况测试 ---")
    T_amb_values = [284, 286, 288, 290]
    DNI_values = [485, 709, 810, 750]
    print(f"  {'T_amb [K]':<12} {'DNI [W/m²]':<14} {'m_dot [kg/s]':<14} {'eta_rec':<10} {'Q_rec [MW]':<12}")
    for T_amb_K, DNI in zip(T_amb_values, DNI_values):
        T_amb_C = T_amb_K - 273.15
        result = cst.compute(DNI, T_amb_C)
        print(f"  {T_amb_K:<12.0f} {DNI:<14.0f} {result['m_fluid_kgps']:<14.1f} {result['eta_receiver']:<10.4f} {result['Q_abs_MW']:<12.1f}")