"""
Economic analysis model
Based on Bayomy & Moore (2020) Section 8, Figure 9, Table 9 & 13
All monetary values in CNY (RMB), converted from CAD at ~5.2 CNY/CAD
"""

import numpy as np


class EconomicAnalyzer:
    """Economic analyzer for NR-HES"""

    def __init__(self, config: dict):
        self.lifetime_years = config.get('lifetime_years', 20)
        self.interest_rate = config.get('interest_rate', 0.05)

        # Unit costs (Table 9, converted to CNY)
        self.smr_cost_per_kWe = config.get('smr_cost_per_kWe_CNY', 36909.6)
        self.ng_cost_per_kWe = config.get('natural_gas_cost_per_kWe_CNY', 8054.8)
        self.cst_cost_per_kWe = config.get('cst_cost_per_kWe_CNY', 48880.0)

        # Electricity rates
        self.peak_rate_per_kWh = config.get('electricity_peak_rate_CNY_per_kWh', 0.78)
        self.offpeak_rate_per_kWh = config.get('electricity_offpeak_rate_CNY_per_kWh', 0.416)

        # NG fuel cost
        self.ng_fuel_cost_per_MWh = config.get('natural_gas_fuel_cost_CNY_per_MWh', 156.0)

        # CRF
        self.CRF = self._calculate_crf()

    def _calculate_crf(self) -> float:
        """CRF = i*(1+i)^n / ((1+i)^n - 1)"""
        i = self.interest_rate
        n = self.lifetime_years
        if i == 0:
            return 1.0 / n
        return i * (1 + i) ** n / ((1 + i) ** n - 1)

    def smr_capital_cost(self, P_net_MWe: float) -> float:
        return P_net_MWe * 1000.0 * self.smr_cost_per_kWe

    def cst_capital_cost(self, P_net_MWe: float) -> float:
        return P_net_MWe * 1000.0 * self.cst_cost_per_kWe

    def fossil_capital_cost(self, P_capacity_MWe: float) -> float:
        return P_capacity_MWe * 1000.0 * self.ng_cost_per_kWe

    def tes_capital_cost(self, fluid_cost: float, tank_height_m: float,
                         tank_diameter_m: float) -> dict:
        """TES capital cost breakdown (Table 13) in CNY"""
        volume_m3 = np.pi * (tank_diameter_m / 2) ** 2 * tank_height_m
        tank_cost = 2.0 * 150000.0 * (volume_m3 / 1000.0) ** 0.7

        fluid_total_cost = fluid_cost
        pumps_hex_cost = tank_cost * 0.688
        piping_insulation = tank_cost * 0.033
        i_c_cost = tank_cost * 0.133
        foundation = tank_cost * 0.011

        total_capital = fluid_total_cost + tank_cost + pumps_hex_cost + piping_insulation + i_c_cost + foundation

        return {
            'fluid_cost_CNY': fluid_total_cost,
            'tanks_cost_CNY': tank_cost,
            'pumps_hex_cost_CNY': pumps_hex_cost,
            'piping_insulation_CNY': piping_insulation,
            'i_c_cost_CNY': i_c_cost,
            'foundation_cost_CNY': foundation,
            'total_capital_CNY': total_capital,
        }

    def annualize_capital(self, capital_cost: float) -> float:
        return capital_cost * self.CRF

    def tes_om_cost(self, capital_cost: float) -> float:
        return capital_cost * 0.416

    def tes_decommissioning_cost(self, capital_cost: float) -> float:
        return capital_cost * 0.0076

    def tes_total_annual_cost(self, capital_cost: float) -> float:
        annual_capital = self.annualize_capital(capital_cost)
        om = self.tes_om_cost(capital_cost)
        decom = self.tes_decommissioning_cost(capital_cost)
        return annual_capital + om + decom

    def tes_revenue(self, total_discharge_MWh: float, storage_efficiency: float) -> float:
        charge_MWh = total_discharge_MWh / storage_efficiency if storage_efficiency > 0 else 0
        revenue = total_discharge_MWh * 1000.0 * self.peak_rate_per_kWh
        cost = charge_MWh * 1000.0 * self.offpeak_rate_per_kWh
        return revenue - cost

    def tes_profit(self, total_discharge_MWh: float, storage_efficiency: float,
                   total_annual_cost: float) -> float:
        revenue = self.tes_revenue(total_discharge_MWh, storage_efficiency)
        return revenue - total_annual_cost

    def performance_index(self, profit_per_year: float, power_cycle_efficiency: float,
                          storage_efficiency: float, tes_annual_capital: float) -> float:
        """Equation (20): I = Profit * eff_power * eff_storage / annual_capital_TES"""
        if tes_annual_capital > 0:
            return (profit_per_year * power_cycle_efficiency * storage_efficiency / tes_annual_capital) * 100.0
        return 0.0

    def system_total_capital(self, smr_P_MWe: float, cst_P_MWe: float,
                              fossil_P_MWe: float, tes_capital: float) -> float:
        smr = self.smr_capital_cost(smr_P_MWe)
        cst = self.cst_capital_cost(cst_P_MWe)
        fossil = self.fossil_capital_cost(fossil_P_MWe)
        return smr + cst + fossil + tes_capital

    def system_annualized_cost(self, total_capital: float) -> float:
        return total_capital * self.CRF

    def clean_energy_shaving_factor(self, total_demand_MWh: float,
                                     fossil_energy_MWh: float) -> float:
        if total_demand_MWh > 0:
            return 1.0 - fossil_energy_MWh / total_demand_MWh
        return 1.0

    def annual_revenue(self, total_electric_MWh: float, total_heat_MWh: float,
                       avg_electric_price_CNY_per_kWh: float = 0.78,
                       avg_heat_price_CNY_per_kWh: float = 0.30) -> float:
        """年收益 = 售电收入 + 售热收入 [CNY]"""
        electric_revenue = total_electric_MWh * 1000.0 * avg_electric_price_CNY_per_kWh
        heat_revenue = total_heat_MWh * 1000.0 * avg_heat_price_CNY_per_kWh
        return electric_revenue + heat_revenue

    def npv(self, initial_investment: float, annual_cashflow: float) -> float:
        """净现值 NPV = Σ(CF/(1+r)^t) - I0, t=1..n"""
        total = 0.0
        for t in range(1, self.lifetime_years + 1):
            total += annual_cashflow / ((1.0 + self.interest_rate) ** t)
        return total - initial_investment

    def irr(self, initial_investment: float, annual_cashflow: float,
            tolerance: float = 1e-6) -> float:
        """内部收益率 IRR: 二分法求解 NPV=0 的折现率"""
        lo, hi = 0.0, 1.0
        for _ in range(100):
            mid = (lo + hi) / 2.0
            npv_val = -initial_investment
            for t in range(1, self.lifetime_years + 1):
                npv_val += annual_cashflow / ((1.0 + mid) ** t)
            if abs(npv_val) < tolerance:
                return mid * 100.0  # 转为百分比
            if npv_val > 0:
                lo = mid
            else:
                hi = mid
        return mid * 100.0

    def lcoe(self, annualized_cost: float, annual_generation_MWh: float,
             annual_fuel_cost: float = 0.0) -> float:
        """平准化度电成本 LCOE = (年均化投资 + 运维 + 燃料) / 年净发电量 [CNY/MWh]"""
        if annual_generation_MWh <= 0:
            return 0.0
        return (annualized_cost + annual_fuel_cost) / annual_generation_MWh

    def print_economic_summary(self, smr_P_MWe: float, cst_P_MWe: float,
                                fossil_P_MWe: float, tes_capital: float,
                                tes_annual_cost: float, tes_profit: float):
        """Print economic summary in CNY"""
        total_capital = self.system_total_capital(smr_P_MWe, cst_P_MWe, fossil_P_MWe, tes_capital)
        annualized = self.system_annualized_cost(total_capital)

        print("\n" + "=" * 60)
        print("  NR-HES Economic Analysis (CNY)")
        print("=" * 60)
        print(f"  Lifetime/Rate:      {self.lifetime_years}y / {self.interest_rate*100:.1f}%")
        print(f"  CRF:                {self.CRF:.6f}")
        print(f"-" * 60)
        print(f"  SMR Cost:           {self.smr_capital_cost(smr_P_MWe):,.0f} CNY")
        print(f"  CST Cost:           {self.cst_capital_cost(cst_P_MWe):,.0f} CNY")
        print(f"  Fossil Cost:        {self.fossil_capital_cost(fossil_P_MWe):,.0f} CNY")
        print(f"  TES Capital Cost:   {tes_capital:,.0f} CNY")
        print(f"-" * 60)
        print(f"  System Total:       {total_capital:,.0f} CNY")
        print(f"  System Annualized:  {annualized:,.0f} CNY/year")
        print(f"-" * 60)
        print(f"  TES Annual Cost:    {tes_annual_cost:,.0f} CNY/year")
        print(f"  TES Annual Profit:  {tes_profit:,.0f} CNY/year")
        print("=" * 60 + "\n")