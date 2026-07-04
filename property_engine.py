"""
水/蒸汽物性引擎 — 基于 IAPWS-IF97 标准
主要使用 CoolProp 库；若不可用则回退至近似公式
"""

import numpy as np

# 尝试导入 CoolProp
try:
    import CoolProp.CoolProp as CP
    HAS_COOLPROP = True
except ImportError:
    HAS_COOLPROP = False
    print("警告: CoolProp 未安装。使用近似蒸汽表公式。安装: pip install CoolProp")


class WaterProperty:
    """水/蒸汽热物性封装"""

    @staticmethod
    def enthalpy_from_PT(P_bar: float, T_C: float) -> float:
        """
        由压力 [bar] 和温度 [°C] 计算比焓 [kJ/kg]
        """
        if HAS_COOLPROP:
            P_Pa = P_bar * 1e5
            try:
                return CP.PropsSI('H', 'P', P_Pa, 'T', T_C + 273.15, 'Water') / 1000.0
            except ValueError:
                # P,T 恰好在饱和线上时回退，判断干度
                T_sat = CP.PropsSI('T', 'P', P_Pa, 'Q', 0.0, 'Water') - 273.15
                if T_C >= T_sat:
                    return CP.PropsSI('H', 'P', P_Pa, 'Q', 1.0, 'Water') / 1000.0
                else:
                    return CP.PropsSI('H', 'P', P_Pa, 'Q', 0.0, 'Water') / 1000.0
        else:
            # 近似公式（适用于过冷水和过热蒸汽区域）
            return WaterProperty._approx_h_from_PT(P_bar, T_C)

    @staticmethod
    def entropy_from_PT(P_bar: float, T_C: float) -> float:
        """
        由压力 [bar] 和温度 [°C] 计算比熵 [kJ/kg·K]
        """
        if HAS_COOLPROP:
            P_Pa = P_bar * 1e5
            try:
                return CP.PropsSI('S', 'P', P_Pa, 'T', T_C + 273.15, 'Water') / 1000.0
            except ValueError:
                T_sat = CP.PropsSI('T', 'P', P_Pa, 'Q', 0.0, 'Water') - 273.15
                if T_C >= T_sat:
                    return CP.PropsSI('S', 'P', P_Pa, 'Q', 1.0, 'Water') / 1000.0
                else:
                    return CP.PropsSI('S', 'P', P_Pa, 'Q', 0.0, 'Water') / 1000.0
        else:
            return WaterProperty._approx_s_from_PT(P_bar, T_C)

    @staticmethod
    def temperature_from_Ph(P_bar: float, h_kJpkg: float) -> float:
        """
        由压力 [bar] 和比焓 [kJ/kg] 反算温度 [°C]
        """
        if HAS_COOLPROP:
            P_Pa = P_bar * 1e5
            return CP.PropsSI('T', 'P', P_Pa, 'H', h_kJpkg * 1000.0, 'Water') - 273.15
        else:
            return WaterProperty._approx_T_from_Ph(P_bar, h_kJpkg)

    @staticmethod
    def enthalpy_saturated_vapor(P_bar: float) -> float:
        """
        饱和蒸汽比焓 [kJ/kg] at P_bar
        """
        if HAS_COOLPROP:
            P_Pa = P_bar * 1e5
            return CP.PropsSI('H', 'P', P_Pa, 'Q', 1.0, 'Water') / 1000.0
        else:
            return WaterProperty._approx_h_sat_vapor(P_bar)

    @staticmethod
    def enthalpy_saturated_liquid(P_bar: float) -> float:
        """
        饱和水比焓 [kJ/kg] at P_bar
        """
        if HAS_COOLPROP:
            P_Pa = P_bar * 1e5
            return CP.PropsSI('H', 'P', P_Pa, 'Q', 0.0, 'Water') / 1000.0
        else:
            return WaterProperty._approx_h_sat_liquid(P_bar)

    @staticmethod
    def saturation_temperature(P_bar: float) -> float:
        """
        饱和温度 [°C] at P_bar
        """
        if HAS_COOLPROP:
            P_Pa = P_bar * 1e5
            return CP.PropsSI('T', 'P', P_Pa, 'Q', 0.0, 'Water') - 273.15
        else:
            return WaterProperty._approx_T_sat(P_bar)

    @staticmethod
    def saturation_pressure(T_C: float) -> float:
        """
        饱和压力 [bar] at T_C
        """
        if HAS_COOLPROP:
            return CP.PropsSI('P', 'T', T_C + 273.15, 'Q', 0.0, 'Water') / 1e5
        else:
            return WaterProperty._approx_P_sat(T_C)

    @staticmethod
    def quality(P_bar: float, h_kJpkg: float) -> float:
        """
        干度 [-] at 给定 P 和 h
        """
        hl = WaterProperty.enthalpy_saturated_liquid(P_bar)
        hv = WaterProperty.enthalpy_saturated_vapor(P_bar)
        if hv == hl:
            return 1.0
        x = (h_kJpkg - hl) / (hv - hl)
        return max(0.0, min(1.0, x))

    # ========================================================================
    #  近似公式（无 CoolProp 时的回退方案）
    #  参考: IAPWS-IF97 简化拟合
    # ========================================================================

    @staticmethod
    def _approx_h_from_PT(P_bar: float, T_C: float) -> float:
        """
        过热蒸汽近似比焓 [kJ/kg]
        适用于 1-100 bar, 100-600°C
        """
        P_MPa = P_bar / 10.0
        T_K = T_C + 273.15
        # 简化公式：h ≈ 4.18*(T_C - 0.5) + 2500 用于近似
        # 更准确的蒸汽近似
        T = T_C
        if T_C >= 100 and P_bar < 50:
            # 过热蒸汽区域近似
            h = 2500.0 + 1.82 * T + 0.0003 * T**2
            # 压力修正
            h += (P_bar - 1.0) * (-0.5 + 0.001 * T)
            return h
        else:
            # 过冷水近似
            return 4.186 * T_C

    @staticmethod
    def _approx_s_from_PT(P_bar: float, T_C: float) -> float:
        """近似比熵"""
        T_K = T_C + 273.15
        # 蒸汽近似
        s0 = 6.5 + 1.8 * np.log(T_K / 373.15)
        s0 -= 0.4615 * np.log(P_bar / 1.013)
        return s0

    @staticmethod
    def _approx_T_from_Ph(P_bar: float, h_kJpkg: float) -> float:
        """由 P 和 h 反算 T"""
        # 迭代法
        T_guess = 200.0
        for _ in range(50):
            h_calc = WaterProperty._approx_h_from_PT(P_bar, T_guess)
            dh = h_calc - h_kJpkg
            if abs(dh) < 0.1:
                return T_guess
            # 导数近似
            cp_approx = 2.0  # kJ/kg·K
            T_guess -= dh / cp_approx
        return T_guess

    @staticmethod
    def _approx_h_sat_vapor(P_bar: float) -> float:
        """饱和蒸汽比焓近似"""
        T_sat = WaterProperty._approx_T_sat(P_bar)
        # 汽化潜热近似
        L = 2500.0 - 2.4 * T_sat  # kJ/kg
        hl = WaterProperty._approx_h_sat_liquid(P_bar)
        return hl + L

    @staticmethod
    def _approx_h_sat_liquid(P_bar: float) -> float:
        """饱和水比焓近似"""
        T_sat = WaterProperty._approx_T_sat(P_bar)
        return 4.186 * T_sat

    @staticmethod
    def _approx_T_sat(P_bar: float) -> float:
        """饱和温度近似 [°C] at P [bar] (Antoine 型公式)"""
        if P_bar <= 0:
            return 0.0
        P_MPa = P_bar / 10.0
        # 简化 Antoine 拟合
        import math
        return 173.0 * (P_MPa ** 0.25) + 75.0 * math.log(P_MPa + 0.001)

    @staticmethod
    def _approx_P_sat(T_C: float) -> float:
        """饱和压力近似 [bar] at T [°C]"""
        T_K = T_C + 273.15
        # 简化 Antoine 公式
        import math
        ln_P_kPa = 16.0 - 3880.0 / (T_K - 42.0)
        return math.exp(ln_P_kPa) / 100.0  # kPa → bar


# ============================================================================
# Rankine 循环便捷计算函数
# ============================================================================

class RankineCycleCalculator:
    """
    Rankine 循环热力学计算器
    实现了文献中方程 (1)-(4)
    """

    def __init__(self, eta_isentropic_turbine=0.90, eta_isentropic_pump=0.75,
                 eta_mechanical=0.90, hex_loss_frac=0.05, hex_TTD=5.0):
        self.eta_turbine = eta_isentropic_turbine
        self.eta_pump = eta_isentropic_pump
        self.eta_mechanical = eta_mechanical
        self.hex_loss = hex_loss_frac
        self.hex_TTD = hex_TTD

    def turbine_exit(self, P_in_bar: float, T_in_C: float, P_out_bar: float) -> dict:
        """
        汽轮机出口状态计算
        文献方程 (1): η_isentropic = (h_in - h_out) / (h_in - h_out,s)

        Returns:
            dict: {'h_out': kJ/kg, 'T_out': °C, 'x_out': quality, 's_out': kJ/kg·K}
        """
        h_in = WaterProperty.enthalpy_from_PT(P_in_bar, T_in_C)
        s_in = WaterProperty.entropy_from_PT(P_in_bar, T_in_C)

        # 等熵出口状态
        # 在 P_out 下找熵等于 s_in 的焓值
        if HAS_COOLPROP:
            P_out_Pa = P_out_bar * 1e5
            h_out_s = CP.PropsSI('H', 'P', P_out_Pa, 'S', s_in * 1000.0, 'Water') / 1000.0
        else:
            # 近似等熵出口
            T_sat_out = WaterProperty.saturation_temperature(P_out_bar)
            h_v_sat = WaterProperty.enthalpy_saturated_vapor(P_out_bar)
            h_l_sat = WaterProperty.enthalpy_saturated_liquid(P_out_bar)
            s_v_sat = WaterProperty.entropy_from_PT(P_out_bar, T_sat_out)
            s_l_sat = WaterProperty.entropy_from_PT(P_out_bar, T_sat_out) - (h_v_sat - h_l_sat) / (T_sat_out + 273.15)
            if s_in > s_v_sat:
                # 过热区
                T_guess = T_sat_out + 50
                h_out_s = WaterProperty.enthalpy_from_PT(P_out_bar, T_guess)
            else:
                x_s = (s_in - s_l_sat) / (s_v_sat - s_l_sat) if s_v_sat != s_l_sat else 1.0
                x_s = max(0.0, min(1.0, x_s))
                h_out_s = h_l_sat + x_s * (h_v_sat - h_l_sat)

        # 实际出口焓 (方程 1)
        h_out = h_in - self.eta_turbine * (h_in - h_out_s)

        # 出口温度与干度
        T_out = WaterProperty.temperature_from_Ph(P_out_bar, h_out)
        x_out = WaterProperty.quality(P_out_bar, h_out)

        # 使用干度计算出口熵（避免两相区 PT 输入导致的 CoolProp 饱和线误差）
        if HAS_COOLPROP:
            P_out_Pa = P_out_bar * 1e5
            if 0.0 < x_out < 1.0:
                s_out = CP.PropsSI('S', 'P', P_out_Pa, 'Q', x_out, 'Water') / 1000.0
            elif x_out <= 0.0:
                s_out = CP.PropsSI('S', 'P', P_out_Pa, 'Q', 0.0, 'Water') / 1000.0
            else:
                s_out = CP.PropsSI('S', 'P', P_out_Pa, 'Q', 1.0, 'Water') / 1000.0
        else:
            s_out = WaterProperty._approx_s_from_PT(P_out_bar, T_out)

        return {'h_out': h_out, 'T_out': T_out, 'x_out': x_out, 's_out': s_out}

    def pump_exit(self, P_in_bar: float, T_in_C: float, P_out_bar: float) -> dict:
        """
        泵出口状态计算（等熵效率法）
        """
        h_in = WaterProperty.enthalpy_from_PT(P_in_bar, T_in_C)
        s_in = WaterProperty.entropy_from_PT(P_in_bar, T_in_C)

        # 等熵压缩
        if HAS_COOLPROP:
            P_out_Pa = P_out_bar * 1e5
            h_out_s = CP.PropsSI('H', 'P', P_out_Pa, 'S', s_in * 1000.0, 'Water') / 1000.0
        else:
            # 不可压缩流体近似
            v = 0.001  # m³/kg
            h_out_s = h_in + v * (P_out_bar - P_in_bar) * 100.0  # bar→kPa

        h_out = h_in + (h_out_s - h_in) / self.eta_pump
        T_out = WaterProperty.temperature_from_Ph(P_out_bar, h_out)

        return {'h_out': h_out, 'T_out': T_out}

    def turbine_power(self, m_steam_kgps: float, h_in_kJpkg: float, h_out_kJpkg: float) -> float:
        """
        汽轮机功率 [MW]
        文献方程 (2): W_turbine = m_steam × (h_in - h_out) × η_mechanical
        """
        return m_steam_kgps * (h_in_kJpkg - h_out_kJpkg) * self.eta_mechanical / 1000.0

    def pump_power(self, m_kgps: float, h_out_kJpkg: float, h_in_kJpkg: float) -> float:
        """
        泵功耗 [MW]
        """
        return m_kgps * (h_out_kJpkg - h_in_kJpkg) / 1000.0

    def net_electric_power(self, W_turbine_MW: float, W_pump_MW: float) -> float:
        """
        净电功率 [MW]
        文献方程 (3): P_net = W_turbine - W_pump
        """
        return W_turbine_MW - W_pump_MW

    def cycle_efficiency(self, P_net_MW: float, Q_th_MW: float) -> float:
        """
        热-电转换效率
        文献方程 (4): η = P_net / Q_th
        """
        return P_net_MW / Q_th_MW if Q_th_MW > 0 else 0.0


# ============================================================================
# 自检
# ============================================================================

if __name__ == "__main__":
    print(f"CoolProp available: {HAS_COOLPROP}")

    # 检验 NuScale 设计点
    wp = WaterProperty()
    print("\n--- NuScale 设计点检验 ---")
    P_turbine_in = 31.0  # bar
    T_turbine_in = 255.0  # °C
    P_condenser = 0.085  # bar
    T_feedwater = 149.0  # °C

    h_turbine_in = wp.enthalpy_from_PT(P_turbine_in, T_turbine_in)
    print(f"汽轮机入口比焓: {h_turbine_in:.1f} kJ/kg")

    rc = RankineCycleCalculator()
    exit_state = rc.turbine_exit(P_turbine_in, T_turbine_in, P_condenser)
    print(f"汽轮机出口: h={exit_state['h_out']:.1f} kJ/kg, T={exit_state['T_out']:.1f}°C, x={exit_state['x_out']:.3f}")

    W_turb = rc.turbine_power(71.3, h_turbine_in, exit_state['h_out'])
    print(f"汽轮机功率: {W_turb:.1f} MW")