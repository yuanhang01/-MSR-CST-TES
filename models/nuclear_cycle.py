"""
熔盐堆 (MSR) 发电循环模型
基于 Bayomy & Moore (2020) Section 3 — 适配 MSR 高温 Brayton/超临界蒸汽循环
效率直接指定（44%），无需展开 Rankine 循环计算
"""

import numpy as np


class NuclearPowerCycle:
    """
    熔盐堆 (MSR) 发电循环
    高温布雷顿/超临界蒸汽循环，效率直接由配置指定
    """

    def __init__(self, config: dict):
        """
        Parameters
        ----------
        config : dict
            config.yaml 中 'smr' 部分配置
        """
        self.Q_th_MW = config.get('reactor_thermal_power_MWth', 160.0)
        self.T_turbine_in_C = config.get('turbine_inlet_temperature_C', 650.0)
        self.T_reactor_outlet_C = config.get('reactor_outlet_temperature_C', 700.0)
        self.eta_design = config.get('cycle_efficiency', 0.44)

        # 净电功率直接计算
        self.P_net_MW = self.Q_th_MW * self.eta_design
        
        # 虚拟参数（不再展开 Rankine 循环，保留接口兼容性）
        self.P_turbine_in_bar = 250.0
        self.m_steam_design_kgps = 0.0
        self.T_feedwater_C = 300.0
        self.P_condenser_bar = 0.08
        self.h1 = 3200.0
        self.h6 = 1800.0
        self.h2 = 800.0
        self.h7 = 810.0
        self.h_feedwater = 1300.0
        self.W_turbine_MW = self.P_net_MW * 1.02
        self.W_pump_MW = self.W_turbine_MW - self.P_net_MW
        self.T6 = 60.0
        self.T7 = 305.0
        self.x6 = 0.95
        self.s1 = 6.5
        self.T2 = 45.0

        self.eta_base_load = self.eta_design
        self.reactor_type = 'MSR'

    def compute_steam_power(self, m_steam_kgps: float, T_superheat_C: float = None) -> dict:
        """
        MSR 模式：直接按效率计算，不依赖蒸汽流量
        """
        if T_superheat_C is None:
            eta = self.eta_design
        else:
            # 高温带来微小效率提升
            eta = self.eta_design + max(0, (T_superheat_C - self.T_turbine_in_C) / 1000.0 * 0.05)

        P_net = self.Q_th_MW * eta

        return {
            'P_net_MW': P_net,
            'W_turbine_MW': P_net * 1.02,
            'W_pump_MW': P_net * 0.02,
            'eta': eta,
            'Q_th_MW': self.Q_th_MW,
            'h_in_kJpkg': self.h1,
            'h_out_kJpkg': self.h6,
            'T_in_C': T_superheat_C or self.T_turbine_in_C,
            'm_steam_kgps': m_steam_kgps,
        }

    def steam_flow_for_power(self, P_target_MW: float, T_superheat_C: float = None) -> float:
        """MSR 不依赖蒸汽流量，返回虚拟值"""
        return self.Q_th_MW * 0.002

    def get_design_summary(self) -> dict:
        return {
            'reactor_type': 'MSR (Molten Salt Reactor)',
            'reactor_thermal_power_MWth': self.Q_th_MW,
            'reactor_outlet_temperature_C': self.T_reactor_outlet_C,
            'turbine_inlet_temperature_C': self.T_turbine_in_C,
            'net_electric_power_MWe': self.P_net_MW,
            'net_efficiency_percent': self.eta_design * 100.0,
            'cycle_efficiency': self.eta_design,
            'W_turbine_MW': self.W_turbine_MW,
            'W_pump_MW': self.W_pump_MW,
        }

    def validate_against_nuscale(self) -> dict:
        """MSR 不与 NuScale 对比，返回自身参数"""
        s = self.get_design_summary()
        return {
            'reactor_thermal_power_MWth': {
                'nuscale': 160.0, 'model': s['reactor_thermal_power_MWth'], 'rel_error_%': 0.0
            },
            'net_electric_power_MWe': {
                'nuscale': 45.0, 'model': s['net_electric_power_MWe'],
                'rel_error_%': abs(s['net_electric_power_MWe'] - 45.0) / 45.0 * 100
            },
            'net_efficiency_percent': {
                'nuscale': 28.0, 'model': s['net_efficiency_percent'],
                'rel_error_%': abs(s['net_efficiency_percent'] - 28.0) / 28.0 * 100
            },
            'live_steam_flow_kgps': {
                'nuscale': 71.3, 'model': 'N/A (MSR)', 'rel_error_%': 'N/A'
            },
            'condenser_pressure_bar': {
                'nuscale': 0.085, 'model': 'N/A (MSR)', 'rel_error_%': 'N/A'
            },
        }

    def print_summary(self):
        s = self.get_design_summary()
        print("\n" + "=" * 60)
        print("  MSR (Molten Salt Reactor) — Design Point")
        print("=" * 60)
        print(f"  Reactor thermal power:      {s['reactor_thermal_power_MWth']:.1f} MWth")
        print(f"  Reactor outlet temp:        {s['reactor_outlet_temperature_C']:.0f} C")
        print(f"  Turbine inlet temp:         {s['turbine_inlet_temperature_C']:.0f} C")
        print(f"  Net electric power:         {s['net_electric_power_MWe']:.1f} MWe")
        print(f"  Cycle efficiency:           {s['net_efficiency_percent']:.1f} %")
        print(f"  Turbine power:              {s['W_turbine_MW']:.1f} MW")
        print(f"  Pump power:                 {s['W_pump_MW']:.3f} MW")
        print("=" * 60 + "\n")