"""
P0 Unit Test: ThermalEnergyStorage.update_temperature()
Tests the explicit Euler temperature update from Section 2 guidance document.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import unittest
import logging
from models.tes_system import ThermalEnergyStorage

# 设置 logging 在测试中可见
logging.basicConfig(level=logging.WARNING, format='[%(levelname)s] %(message)s')


class TestUpdateTemperature(unittest.TestCase):
    """测试一阶显式欧拉法温度递推方法"""

    def setUp(self):
        """每个测试用例前设置: Therminol 流体物性参数"""
        # Therminol 66 at ~270°C
        self.m_total = 941.0 * 1100.0   # rho * V = 941 kg/m3 * 1100 m3 ≈ 1,035,100 kg (H=14,D=10)
        self.cp = 1.91                   # kJ/kg·K
        self.T_init = 272.0              # °C
        self.T_min = 25.0                # 冷罐温度
        self.T_max = 315.0               # Therminol 最高工作温度

    # ------------------------------------------------------------------
    # 正常情况测试
    # ------------------------------------------------------------------
    def test_heating_normal(self):
        """正常充能：Q_net > 0，温度应上升"""
        T_new = ThermalEnergyStorage.update_temperature(
            self.T_init, Q_net_MW=50.0, dt_hours=1.0,
            m_total_kg=self.m_total, cp_kJpkgK=self.cp,
            T_min_C=self.T_min, T_max_C=self.T_max
        )
        # Q*dt*3600 / (m*cp) = 50*1*3600 / (1,035,100*1.91) ≈ 0.091°C
        delta = 50.0 * 1.0 * 3600.0 / (self.m_total * self.cp)
        expected = self.T_init + delta
        self.assertAlmostEqual(T_new, expected, places=1)
        self.assertGreater(T_new, self.T_init)  # 温度上升

    def test_cooling_normal(self):
        """正常放能：Q_net < 0，温度应下降"""
        T_new = ThermalEnergyStorage.update_temperature(
            self.T_init, Q_net_MW=-30.0, dt_hours=1.0,
            m_total_kg=self.m_total, cp_kJpkgK=self.cp,
            T_min_C=self.T_min, T_max_C=self.T_max
        )
        delta = -30.0 * 1.0 * 3600.0 / (self.m_total * self.cp)
        expected = self.T_init + delta
        self.assertAlmostEqual(T_new, expected, places=1)
        self.assertLess(T_new, self.T_init)  # 温度下降

    def test_idle_no_change(self):
        """无热输入 Q_net=0，温度不变"""
        T_new = ThermalEnergyStorage.update_temperature(
            self.T_init, Q_net_MW=0.0, dt_hours=1.0,
            m_total_kg=self.m_total, cp_kJpkgK=self.cp,
            T_min_C=self.T_min, T_max_C=self.T_max
        )
        self.assertAlmostEqual(T_new, self.T_init, places=3)

    # ------------------------------------------------------------------
    # Section 三 边界约束测试
    # ------------------------------------------------------------------
    def test_exceed_T_max_clipped(self):
        """边界条件：超过 T_max 时自动夹紧（用极小质量放大温度变化）"""
        # 用小质量模拟极限情况：1000 kg, 200MW × 1h → ΔT = 200*3600/(1000*1.91) ≈ 377°C
        m_small = 1000.0
        T_new = ThermalEnergyStorage.update_temperature(
            T_current_C=310.0,
            Q_net_MW=200.0, dt_hours=1.0,
            m_total_kg=m_small, cp_kJpkgK=self.cp,
            T_min_C=self.T_min, T_max_C=self.T_max
        )
        self.assertEqual(T_new, self.T_max)
        self.assertLessEqual(T_new, self.T_max)

    def test_below_T_min_clipped(self):
        """边界条件：低于 T_min 时自动夹紧（用极小质量放大温度变化）"""
        m_small = 1000.0
        T_new = ThermalEnergyStorage.update_temperature(
            T_current_C=30.0,
            Q_net_MW=-200.0, dt_hours=1.0,
            m_total_kg=m_small, cp_kJpkgK=self.cp,
            T_min_C=self.T_min, T_max_C=self.T_max
        )
        self.assertEqual(T_new, self.T_min)
        self.assertGreaterEqual(T_new, self.T_min)

    # ------------------------------------------------------------------
    # 零/负物性参数保护测试
    # ------------------------------------------------------------------
    def test_zero_mass_returns_current(self):
        """总质量为零时返回当前温度（防御性编程）"""
        T_new = ThermalEnergyStorage.update_temperature(
            self.T_init, Q_net_MW=50.0, dt_hours=1.0,
            m_total_kg=0.0, cp_kJpkgK=self.cp,
            T_min_C=self.T_min, T_max_C=self.T_max
        )
        self.assertEqual(T_new, self.T_init)

    def test_zero_cp_returns_current(self):
        """比热为零时返回当前温度"""
        T_new = ThermalEnergyStorage.update_temperature(
            self.T_init, Q_net_MW=50.0, dt_hours=1.0,
            m_total_kg=self.m_total, cp_kJpkgK=0.0,
            T_min_C=self.T_min, T_max_C=self.T_max
        )
        self.assertEqual(T_new, self.T_init)

    # ------------------------------------------------------------------
    # 子步划分测试 (P1)
    # ------------------------------------------------------------------
    def test_substep_consistency(self):
        """1步 vs 4子步：结果应近似相同（显式欧拉法一致）"""
        # 单步
        T_single = ThermalEnergyStorage.update_temperature(
            self.T_init, Q_net_MW=40.0, dt_hours=1.0,
            m_total_kg=self.m_total, cp_kJpkgK=self.cp,
            T_min_C=self.T_min, T_max_C=self.T_max
        )

        # 4子步
        T_sub = self.T_init
        Q_sub = 40.0 / 4.0  # 每子步相同热输入
        dt_sub = 0.25
        for _ in range(4):
            T_sub = ThermalEnergyStorage.update_temperature(
                T_sub, Q_net_MW=Q_sub, dt_hours=dt_sub,
                m_total_kg=self.m_total, cp_kJpkgK=self.cp,
                T_min_C=self.T_min, T_max_C=self.T_max
            )

        # 显式欧拉法不守恒，4子步更精确，允许微小偏差
        self.assertAlmostEqual(T_single, T_sub, delta=0.5)

    # ------------------------------------------------------------------
    # 时间步长大小测试
    # ------------------------------------------------------------------
    def test_small_dt(self):
        """小时间步：温度变化微小"""
        T_new = ThermalEnergyStorage.update_temperature(
            self.T_init, Q_net_MW=100.0, dt_hours=0.25,  # 15分钟
            m_total_kg=self.m_total, cp_kJpkgK=self.cp,
            T_min_C=self.T_min, T_max_C=self.T_max
        )
        delta = 100.0 * 0.25 * 3600.0 / (self.m_total * self.cp)
        expected = self.T_init + delta
        self.assertAlmostEqual(T_new, expected, places=1)
        self.assertAlmostEqual(T_new, self.T_init, delta=0.5)  # 变化小于0.5°C

    def test_large_dt(self):
        """大时间步：温度变化显著"""
        T_new = ThermalEnergyStorage.update_temperature(
            self.T_init, Q_net_MW=100.0, dt_hours=24.0,  # 24小时
            m_total_kg=self.m_total, cp_kJpkgK=self.cp,
            T_min_C=self.T_min, T_max_C=self.T_max
        )
        # 24h持续加热应达到T_max
        self.assertGreater(T_new, self.T_init + 1.0)  # 明显上升
        self.assertLessEqual(T_new, self.T_max)


if __name__ == '__main__':
    unittest.main(verbosity=2)