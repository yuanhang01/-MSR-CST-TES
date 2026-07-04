"""
双罐显热储热系统 (TES) 模型
基于 Bayomy & Moore (2020) Section 5, Figure 9
包含充/放能逻辑和热损失模型
"""

import numpy as np
import sys
import os
import logging
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fluid_library import StorageFluid

logger = logging.getLogger(__name__)


class ThermalEnergyStorage:
    """
    双罐显热储热系统
    Figure 9 所示：冷罐 (state 20) ↔ HEX #1 ↔ 热罐 (state 21)
    """

    def __init__(self, config: dict, storage_fluid: StorageFluid = None):
        """
        Parameters
        ----------
        config : dict
            config.yaml 中 'tes' 部分的配置
        storage_fluid : StorageFluid
            储热流体实例
        """
        self.H_m = config.get('tank_height_m', 14.0)
        self.D_m = config.get('tank_diameter_m', 10.0)
        self.U_Wpm2K = config.get('tank_insulation_U_Wpm2K', 0.5)
        self.aux_heater_eff = config.get('auxiliary_heater_efficiency', 0.95)

        # 储热流体
        self.fluid = storage_fluid
        if self.fluid is not None:
            self.T_cold = config.get('cold_tank_temperature_C', self.fluid.T_cold_tank)
        else:
            self.T_cold = config.get('cold_tank_temperature_C', 25.0)

        self.T_hot_init = config.get('hot_tank_initial_temperature_C', 272.0)

        # 几何参数
        self._calculate_geometry()

        # 状态变量
        self.T_hot = self.T_hot_init  # 热罐温度 [°C]
        self.T_cold_current = self.T_cold  # 冷罐当前温度 [°C]
        self.SOC = 0.5  # 储能量状态 [0-1]
        self.m_fluid_total_kg = self.volume_m3 * self._get_density_avg()

        # 累积统计
        self.total_charge_MWh = 0.0
        self.total_discharge_MWh = 0.0
        self.wasted_energy_MWh = 0.0  # 储罐满时无法存储的能量

        # 辅助加热功率（高熔点盐类）
        if self.fluid is not None and self.fluid.needs_auxiliary_heating():
            self.aux_heater_power_MW = self._calculate_aux_heater_power()
        else:
            self.aux_heater_power_MW = 0.0

        # 边界触发计数器（Section 三 防御性编程）
        self.boundary_events = {
            't_hot_max_hit': 0,       # 热罐触及 T_max
            't_hot_min_hit': 0,       # 热罐触及 T_cold (完全放空)
            'tank_full_overflow': 0,   # 储罐满溢
            'tank_empty': 0,           # 储罐耗尽
            'charge_clipped': 0,       # 充能受限
            'discharge_clipped': 0,    # 放能受限
        }

    # ========================================================================
    # P0: 独立温度递推方法（一阶显式欧拉法，Section 二核心公式）
    # ========================================================================

    @staticmethod
    def update_temperature(T_current_C: float, Q_net_MW: float, dt_hours: float,
                           m_total_kg: float, cp_kJpkgK: float,
                           T_min_C: float, T_max_C: float) -> float:
        """
        一阶显式欧拉法温度递推（Section 二 核心公式）
        T[t+1] = T[t] + (dt / (m * cp)) * Q_net

        Parameters
        ----------
        T_current_C : float    当前温度 [°C]
        Q_net_MW : float       净热输入功率 [MW]（充能 > 0, 放能 < 0, 散热 < 0）
        dt_hours : float       时间步长 [h]
        m_total_kg : float     储热介质总质量 [kg]
        cp_kJpkgK : float      储热介质比热 [kJ/kg·K]
        T_min_C : float        允许最低温度 [°C]
        T_max_C : float        允许最高温度 [°C]

        Returns
        -------
        float: 更新后的温度 [°C]，自动夹紧至 [T_min, T_max]
        """
        if m_total_kg <= 0 or cp_kJpkgK <= 0:
            logger.warning("update_temperature: m_total or cp is zero, returning current T")
            return T_current_C

        # 核心递推公式：ΔT = (Q * dt * 3600) / (m * cp)
        delta_T_C = (Q_net_MW * dt_hours * 3600.0) / (m_total_kg * cp_kJpkgK)

        T_new = T_current_C + delta_T_C

        # Section 三 边界约束：强制夹紧
        if T_new > T_max_C:
            logger.warning(f"T_hot ({T_new:.1f}C) exceeds T_max ({T_max_C:.1f}C), "
                          f"clipped to T_max. (Q_net={Q_net_MW:.2f}MW, dt={dt_hours}h)")
            return T_max_C
        elif T_new < T_min_C:
            logger.warning(f"T_hot ({T_new:.1f}C) below T_min ({T_min_C:.1f}C), "
                          f"clipped to T_min. (Q_net={Q_net_MW:.2f}MW, dt={dt_hours}h)")
            return T_min_C

        return T_new

    # ========================================================================
    # 几何计算
    # ========================================================================

    def _calculate_geometry(self):
        """计算储罐几何参数"""
        self.A_cross_m2 = np.pi * (self.D_m / 2.0) ** 2
        self.volume_m3 = self.A_cross_m2 * self.H_m
        self.A_surface_m2 = 2.0 * self.A_cross_m2 + np.pi * self.D_m * self.H_m

    def _get_density_avg(self) -> float:
        """获取平均密度 [kg/m³]"""
        if self.fluid is not None:
            T_avg = (self.T_cold + self.T_hot) / 2.0
            return self.fluid.density(T_avg)
        return 1000.0

    def _calculate_aux_heater_power(self) -> float:
        """
        计算维持冷罐温度所需的辅助加热功率 [MW]
        （防止熔盐凝固）
        """
        if self.fluid is None:
            return 0.0
        T_amb = 20.0  # 假设环境温度
        Q_loss_W = self.U_Wpm2K * self.A_surface_m2 * (self.T_cold - T_amb)
        return Q_loss_W / (1e6 * self.aux_heater_eff) if Q_loss_W > 0 else 0.0

    def _heat_loss_rate(self) -> float:
        """
        热罐散热损失速率 [MW]
        """
        T_amb = 20.0  # 假设环境温度
        Q_loss_W = self.U_Wpm2K * self.A_surface_m2 * max(0, self.T_hot - T_amb)
        return Q_loss_W / 1e6

    @property
    def max_storage_energy_MWh(self) -> float:
        """
        最大储能量 [MWh]
        E_max = ρ × V × cp × (T_hot_max - T_cold) / 3600
        """
        if self.fluid is None:
            return 0.0
        T_avg = (self.T_cold + self.fluid.T_out_solar) / 2.0
        rho = self.fluid.density(T_avg)
        cp = self.fluid.specific_heat(T_avg)  # kJ/kg·K
        delta_T = self.fluid.T_out_solar - self.T_cold
        E_MJ = rho * self.volume_m3 * cp * delta_T
        return E_MJ / 3600.0  # MJ → MWh

    @property
    def current_stored_energy_MWh(self) -> float:
        """
        当前储能量 [MWh]
        """
        if self.fluid is None:
            return 0.0
        T_avg = (self.T_cold_current + self.T_hot) / 2.0
        rho = self.fluid.density(T_avg)
        cp = self.fluid.specific_heat(T_avg)
        delta_T = max(0, self.T_hot - self.T_cold_current)
        E_MJ = rho * self.volume_m3 * cp * delta_T
        return E_MJ / 3600.0

    def charge(self, Q_excess_MW: float, dt_hours: float,
               T_fluid_from_source_C: float = None) -> float:
        """
        充能操作

        Parameters
        ----------
        Q_excess_MW : float
            可用的多余热功率 [MW]
        dt_hours : float
            时间步长 [h]
        T_fluid_from_source_C : float, optional
            来自热源的流体温度

        Returns
        -------
        float: 实际存储的能量 [MWh]
        """
        if self.fluid is None:
            self.wasted_energy_MWh += Q_excess_MW * dt_hours
            return 0.0

        if T_fluid_from_source_C is None:
            T_fluid_from_source_C = self.fluid.T_out_solar

        # 最大可充能量
        E_max_charge = self.max_storage_energy_MWh - self.current_stored_energy_MWh

        # 实际充能量 [MWh]
        E_charge_MWh = min(Q_excess_MW * dt_hours, E_max_charge)

        if E_charge_MWh <= 0:
            self.wasted_energy_MWh += Q_excess_MW * dt_hours
            return 0.0

        # 更新热罐温度
        T_avg = (self.T_hot + T_fluid_from_source_C) / 2.0
        rho = self.fluid.density(T_avg)
        cp = self.fluid.specific_heat(T_avg)  # kJ/kg·K
        m_total = rho * self.volume_m3

        if m_total > 0 and cp > 0:
            delta_T = (E_charge_MWh * 3600.0) / (m_total * cp)  # [K]
            self.T_hot = min(self.T_hot + delta_T, T_fluid_from_source_C)

        # 充能热损失
        Q_loss_charge_MWh = self._heat_loss_rate() * dt_hours
        self.T_hot = max(self.T_cold_current, self.T_hot - Q_loss_charge_MWh / (m_total * cp * 3600.0) if m_total > 0 else 0)

        # 更新统计
        self.total_charge_MWh += E_charge_MWh

        # 剩余不能存储的能量
        wasted = Q_excess_MW * dt_hours - E_charge_MWh
        if wasted > 0:
            self.wasted_energy_MWh += wasted

        # 更新 SOC
        self._update_soc()

        return E_charge_MWh

    def discharge(self, Q_demand_MW: float, dt_hours: float) -> float:
        """
        放能操作

        Parameters
        ----------
        Q_demand_MW : float
            需求的热功率 [MW]
        dt_hours : float
            时间步长 [h]

        Returns
        -------
        float: 实际释放的能量 [MWh]
        """
        if self.fluid is None or self.current_stored_energy_MWh <= 0:
            return 0.0

        E_demand_MWh = Q_demand_MW * dt_hours
        E_available_MWh = self.current_stored_energy_MWh

        # 实际放能量
        E_discharge_MWh = min(E_demand_MWh, E_available_MWh)

        if E_discharge_MWh <= 0:
            return 0.0

        # 更新热罐温度
        T_avg = (self.T_hot + self.T_cold_current) / 2.0
        rho = self.fluid.density(T_avg)
        cp = self.fluid.specific_heat(T_avg)
        m_total = rho * self.volume_m3

        if m_total > 0 and cp > 0:
            delta_T = (E_discharge_MWh * 3600.0) / (m_total * cp)
            self.T_hot = max(self.T_cold_current, self.T_hot - delta_T)

        # 放能热损失
        Q_loss_discharge_MWh = self._heat_loss_rate() * dt_hours
        if m_total > 0 and cp > 0:
            self.T_hot = max(self.T_cold_current, self.T_hot - Q_loss_discharge_MWh / (m_total * cp * 3600.0))

        # 更新统计
        self.total_discharge_MWh += E_discharge_MWh

        # 更新 SOC
        self._update_soc()

        return E_discharge_MWh

    def idle(self, dt_hours: float):
        """
        空闲时间步的散热损失
        """
        if self.fluid is None:
            return

        T_avg = (self.T_hot + self.T_cold_current) / 2.0
        rho = self.fluid.density(T_avg)
        cp = self.fluid.specific_heat(T_avg)
        m_total = rho * self.volume_m3

        Q_loss_MWh = self._heat_loss_rate() * dt_hours
        if m_total > 0 and cp > 0:
            self.T_hot = max(self.T_cold_current, self.T_hot - Q_loss_MWh / (m_total * cp * 3600.0))

        self._update_soc()

    def _update_soc(self):
        """更新储能状态"""
        E_max = self.max_storage_energy_MWh
        E_current = self.current_stored_energy_MWh
        self.SOC = E_current / E_max if E_max > 0 else 0.0

    @property
    def round_trip_efficiency(self) -> float:
        """
        往返效率
        η_storage = E_discharge / E_charge
        """
        if self.total_charge_MWh > 0:
            return self.total_discharge_MWh / self.total_charge_MWh
        return 0.0

    @property
    def average_hot_tank_temperature(self) -> float:
        """年均热罐温度近似（由调用者在统计中更新）"""
        return self.T_hot

    def get_summary(self) -> dict:
        """返回 TES 摘要"""
        return {
            'tank_height_m': self.H_m,
            'tank_diameter_m': self.D_m,
            'tank_volume_m3': self.volume_m3,
            'total_charge_MWh': self.total_charge_MWh,
            'total_discharge_MWh': self.total_discharge_MWh,
            'storage_efficiency_percent': self.round_trip_efficiency * 100.0,
            'wasted_energy_MWh': self.wasted_energy_MWh,
            'hot_tank_temperature_C': self.T_hot,
            'cold_tank_temperature_C': self.T_cold_current,
            'SOC': self.SOC,
            'aux_heater_power_MW': self.aux_heater_power_MW,
            'max_storage_MWh': self.max_storage_energy_MWh,
        }

    def print_summary(self):
        """打印 TES 摘要"""
        s = self.get_summary()
        print("\n" + "=" * 60)
        print("  双罐显热储热系统 (TES)")
        print("=" * 60)
        print(f"  储罐尺寸:          H={s['tank_height_m']:.1f} m, D={s['tank_diameter_m']:.1f} m")
        print(f"  储罐容积:          {s['tank_volume_m3']:.0f} m³")
        print(f"  最大储能量:        {s['max_storage_MWh']:.1f} MWh")
        print(f"  总充能量:          {s['total_charge_MWh']:.1f} MWh")
        print(f"  总放能量:          {s['total_discharge_MWh']:.1f} MWh")
        print(f"  往返效率:          {s['storage_efficiency_percent']:.1f}%")
        print(f"  当前SOC:           {s['SOC']:.4f}")
        print(f"  热罐温度:          {s['hot_tank_temperature_C']:.1f} °C")
        print(f"  冷罐温度:          {s['cold_tank_temperature_C']:.1f} °C")
        if s['wasted_energy_MWh'] > 0:
            print(f"  浪费能量:          {s['wasted_energy_MWh']:.1f} MWh")
        if s['aux_heater_power_MW'] > 0:
            print(f"  辅助加热功率:      {s['aux_heater_power_MW']:.4f} MW")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    from fluid_library import get_fluid

    fluid = get_fluid("Therminol")
    config = {
        'tank_height_m': 14.0,
        'tank_diameter_m': 10.0,
        'tank_insulation_U_Wpm2K': 0.5,
        'cold_tank_temperature_C': 25.0,
        'hot_tank_initial_temperature_C': 272.0,
        'auxiliary_heater_efficiency': 0.95,
    }

    tes = ThermalEnergyStorage(config, fluid)
    print(f"储罐容积: {tes.volume_m3:.0f} m³")
    print(f"最大储能量: {tes.max_storage_energy_MWh:.1f} MWh")
    print(f"初始储能: {tes.current_stored_energy_MWh:.1f} MWh")

    # 测试充放能
    charged = tes.charge(50.0, 1.0)  # 50 MW充能1小时
    print(f"充能 50 MW × 1h = {charged:.1f} MWh")
    print(f"储能后温度: {tes.T_hot:.1f} °C")
    print(f"SOC: {tes.SOC:.3f}")

    discharged = tes.discharge(30.0, 1.0)
    print(f"放能 30 MW × 1h = {discharged:.1f} MWh")
    print(f"放能后温度: {tes.T_hot:.1f} °C")
    print(f"往返效率: {tes.round_trip_efficiency:.4f}")