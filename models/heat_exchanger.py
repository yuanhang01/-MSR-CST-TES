"""
中间换热器 (IHEX) 模型
基于 Bayomy & Moore (2020) Figure 9
用于蒸汽-储热流体之间的换热
"""

import numpy as np


class IntermediateHeatExchanger:
    """
    中间换热器模型
    Figure 9: HEX #1 和 HEX #2
    充能模式：热蒸汽 → 冷储热流体（冷→热）
    放能模式：热储热流体 → 冷给水 → 蒸汽
    """

    def __init__(self, effectiveness: float = 0.85, heat_loss_frac: float = 0.05,
                 TTD_C: float = 5.0, P_steam_bar: float = 31.0):
        """
        Parameters
        ----------
        effectiveness : float
            换热器效能 (ε-NTU 方法)
        heat_loss_frac : float
            换热器热损失比例
        TTD_C : float
            换热端差 (Terminal Temperature Difference) [°C]
        P_steam_bar : float
            蒸汽侧压力 [bar]
        """
        self.eff = effectiveness
        self.loss_frac = heat_loss_frac
        self.TTD = TTD_C
        self.P_steam_bar = P_steam_bar

    def charge_mode(self, Q_excess_MW: float, T_steam_in_C: float,
                    T_fluid_cold_C: float, cp_fluid_kJpkgK: float,
                    m_fluid_kgps: float = None) -> dict:
        """
        充能模式：蒸汽 → 储热流体

        Parameters
        ----------
        Q_excess_MW : float
            多余热功率 [MW]
        T_steam_in_C : float
            蒸汽入口温度 [°C]
        T_fluid_cold_C : float
            冷罐流体温度 [°C]
        cp_fluid_kJpkgK : float
            储热流体比热 [kJ/kg·K]
        m_fluid_kgps : float, optional
            储热流体质量流量（可迭代求解）

        Returns
        -------
        dict: 包含热流体出口温度、实际传热量等
        """
        # 最大可能的流体出口温度
        T_fluid_out_max = T_steam_in_C - self.TTD

        # 根据效能计算实际传热量
        # ε-NTU: Q = ε × C_min × (T_hot_in - T_cold_in)
        # 简化为直接换热模型
        if m_fluid_kgps is not None and m_fluid_kgps > 0:
            C_fluid = m_fluid_kgps * cp_fluid_kJpkgK  # kW/K
            # 传热上限
            Q_max = C_fluid * (T_steam_in_C - T_fluid_cold_C) / 1000.0  # MW
            Q_actual = min(Q_excess_MW, Q_max) * self.eff * (1 - self.loss_frac)
            T_fluid_out = T_fluid_cold_C + Q_actual * 1000.0 / C_fluid if C_fluid > 0 else T_fluid_cold_C
        else:
            # 无流量时
            Q_actual = 0.0
            T_fluid_out = T_fluid_cold_C

        # 蒸汽侧出口（冷凝/冷却）
        T_steam_out = T_steam_in_C - self.TTD - 10.0  # 简化估算
        T_steam_out = max(100.0, T_steam_out)  # 不低于100°C

        return {
            'Q_transferred_MW': Q_actual,
            'T_fluid_out_C': min(T_fluid_out, T_fluid_out_max),
            'T_steam_out_C': T_steam_out,
            'effectiveness': self.eff,
        }

    def discharge_mode(self, Q_demand_MW: float, T_fluid_hot_C: float,
                       T_water_feed_C: float, m_water_kgps: float,
                       cp_water_kJpkgK: float = 4.186) -> dict:
        """
        放能模式：热储热流体 → 给水 → 蒸汽

        Parameters
        ----------
        Q_demand_MW : float
            需求热功率 [MW]
        T_fluid_hot_C : float
            热罐流体温度 [°C]
        T_water_feed_C : float
            补水温度 [°C]
        m_water_kgps : float
            给水质量流量 [kg/s]
        cp_water_kJpkgK : float
            水的比热 [kJ/kg·K]

        Returns
        -------
        dict: 包含蒸汽出口温度、实际传热量等
        """
        # 蒸汽可能达到的最高温度
        T_steam_out_max = T_fluid_hot_C - self.TTD

        # 给水加热+蒸发
        C_water = m_water_kgps * cp_water_kJpkgK  # kW/K
        Q_heating = C_water * (T_steam_out_max - T_water_feed_C) / 1000.0  # MW

        Q_actual = min(Q_demand_MW, Q_heating) * self.eff * (1 - self.loss_frac)
        Q_actual = max(0.0, Q_actual)

        # 蒸汽出口温度
        if C_water > 0:
            T_steam_out = T_water_feed_C + Q_actual * 1000.0 / C_water
        else:
            T_steam_out = T_water_feed_C
        T_steam_out = min(T_steam_out, T_steam_out_max)

        # 储热流体出口温度
        T_fluid_out = T_fluid_hot_C - Q_actual * 1000.0 / C_water if C_water > 0 else T_fluid_hot_C
        T_fluid_out = max(T_fluid_out, T_water_feed_C + self.TTD)

        return {
            'Q_transferred_MW': Q_actual,
            'T_steam_out_C': T_steam_out,
            'T_fluid_out_C': T_fluid_out,
            'm_steam_kgps': m_water_kgps,
            'effectiveness': self.eff,
        }

    def superheater_duty(self, m_steam_kgps: float, T_steam_in_C: float,
                         T_target_C: float, cp_steam_kJpkgK: float = 2.0) -> float:
        """
        过热器热负荷计算

        过热器用于提升蒸汽温度从 T_steam_in 到 T_target
        可使用太阳能或化石燃料

        Returns
        -------
        float: 所需热功率 [MW]
        """
        delta_T = max(0, T_target_C - T_steam_in_C)
        Q_MW = m_steam_kgps * cp_steam_kJpkgK * delta_T / 1000.0
        return Q_MW

    def reheater_duty(self, m_steam_kgps: float, T_steam_in_C: float,
                      T_target_C: float, cp_steam_kJpkgK: float = 2.0) -> float:
        """
        再热器热负荷计算

        再热器用于提升汽轮机中间抽汽温度
        """
        return self.superheater_duty(m_steam_kgps, T_steam_in_C, T_target_C, cp_steam_kJpkgK)


if __name__ == "__main__":
    ihex = IntermediateHeatExchanger(effectiveness=0.85, heat_loss_frac=0.05, TTD_C=5.0)

    # 测试充能模式
    charge_result = ihex.charge_mode(
        Q_excess_MW=20.0,
        T_steam_in_C=255.0,
        T_fluid_cold_C=25.0,
        cp_fluid_kJpkgK=1.91,
        m_fluid_kgps=50.0
    )
    print("充能模式:")
    for k, v in charge_result.items():
        print(f"  {k}: {v:.2f}")

    # 测试放能模式
    discharge_result = ihex.discharge_mode(
        Q_demand_MW=15.0,
        T_fluid_hot_C=272.0,
        T_water_feed_C=149.0,
        m_water_kgps=30.0
    )
    print("\n放能模式:")
    for k, v in discharge_result.items():
        print(f"  {k}: {v:.2f}")

    # 过热器
    Q_sh = ihex.superheater_duty(71.3, 255.0, 295.0)
    print(f"\n过热器需求 (255→295°C): {Q_sh:.2f} MW")