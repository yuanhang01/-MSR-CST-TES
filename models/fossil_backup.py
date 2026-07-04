"""
Fossil fuel backup system model (NGCC)
Based on Bayomy & Moore (2020) Section 7, 9
All monetary values in CNY
"""

import numpy as np


class FossilFuelBackup:
    """Natural gas combined cycle backup"""

    def __init__(self, efficiency: float = 0.55, fuel_cost_per_MWh: float = 156.0,
                 capital_cost_per_kWe: float = 8054.8):
        self.eta = efficiency
        self.fuel_cost_per_MWh = fuel_cost_per_MWh
        self.capital_cost_per_kWe = capital_cost_per_kWe

        self.total_thermal_MWh = 0.0
        self.total_electric_MWh = 0.0
        self.total_fuel_cost = 0.0
        self.total_fuel_consumed_MWh = 0.0
        self.capacity_MWe = 0.0

    def produce_heat(self, Q_required_MW: float, dt_hours: float) -> dict:
        Q_actual_MWh = Q_required_MW * dt_hours
        fuel_MWh = Q_actual_MWh / self.eta
        cost = fuel_MWh * self.fuel_cost_per_MWh
        self.total_thermal_MWh += Q_actual_MWh
        self.total_fuel_consumed_MWh += fuel_MWh
        self.total_fuel_cost += cost
        return {'Q_produced_MWh': Q_actual_MWh, 'Q_rate_MW': Q_required_MW,
                'fuel_consumed_MWh': fuel_MWh, 'fuel_cost': cost, 'efficiency': self.eta}

    def produce_electricity(self, P_required_MW: float, dt_hours: float) -> dict:
        P_actual_MWh = P_required_MW * dt_hours
        fuel_MWh = P_actual_MWh / self.eta
        cost = fuel_MWh * self.fuel_cost_per_MWh
        self.total_electric_MWh += P_actual_MWh
        self.total_fuel_consumed_MWh += fuel_MWh
        self.total_fuel_cost += cost
        return {'P_produced_MWh': P_actual_MWh, 'P_rate_MW': P_required_MW,
                'fuel_consumed_MWh': fuel_MWh, 'fuel_cost': cost, 'efficiency': self.eta}

    def get_capital_cost(self) -> float:
        return self.capacity_MWe * self.capital_cost_per_kWe * 1000.0

    def get_summary(self) -> dict:
        return {'total_thermal_output_MWh': self.total_thermal_MWh,
                'total_electric_output_MWh': self.total_electric_MWh,
                'total_fuel_consumed_MWh': self.total_fuel_consumed_MWh,
                'total_fuel_cost': self.total_fuel_cost,
                'efficiency': self.eta, 'capacity_MWe': self.capacity_MWe,
                'capital_cost': self.get_capital_cost()}

    def print_summary(self):
        s = self.get_summary()
        print("\n" + "=" * 60)
        print("  Fossil Fuel Backup System")
        print("=" * 60)
        print(f"  Thermal output:      {s['total_thermal_output_MWh']:.1f} MWh")
        print(f"  Electric output:     {s['total_electric_output_MWh']:.1f} MWh")
        print(f"  Fuel consumed:       {s['total_fuel_consumed_MWh']:.1f} MWh")
        print(f"  Fuel cost:           {s['total_fuel_cost']:,.0f} CNY")
        print(f"  Efficiency:          {s['efficiency']*100:.1f}%")
        print(f"  Capacity:            {s['capacity_MWe']:.1f} MWe")
        print("=" * 60 + "\n")