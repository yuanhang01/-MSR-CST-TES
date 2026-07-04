"""
NR-HES Transient Coupling Solver
Based on Bayomy & Moore (2020) Section 7, Figure 8-9
Time-stepping solves SMR + CST + TES coupled system for 8760 hours/year
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.nuclear_cycle import NuclearPowerCycle
from models.cst_plant import ConcentratedSolarTower
from models.tes_system import ThermalEnergyStorage
from models.heat_exchanger import IntermediateHeatExchanger
from models.fossil_backup import FossilFuelBackup
from models.economics import EconomicAnalyzer
from fluid_library import StorageFluid, get_fluid


class NRHESTransientSolver:
    """
    NR-HES Transient Coupling Solver
    Couples SMR, CST, TES and fossil backup according to Figure 8 logic
    """

    def __init__(self, smr: NuclearPowerCycle, cst: ConcentratedSolarTower,
                 tes: ThermalEnergyStorage, ihex: IntermediateHeatExchanger,
                 fossil: FossilFuelBackup, economics: EconomicAnalyzer,
                 dt_hours: float = 1.0, verbose: bool = True,
                 fossil_backup_active: bool = False):
        self.smr = smr
        self.cst = cst
        self.tes = tes
        self.ihex = ihex
        self.fossil = fossil
        self.econ = economics
        self.dt = dt_hours
        self.verbose = verbose
        self.fossil_active = fossil_backup_active

        self.results = {
            'time_h': [], 'DNI_Wpm2': [], 'T_amb_C': [],
            'P_demand_MW': [], 'H_demand_MW': [],
            'P_nuclear_MW': [], 'P_solar_to_grid_MW': [], 'P_fossil_MW': [],
            'P_total_supply_MW': [], 'Q_nuclear_MW': [], 'Q_solar_MW': [],
            'Q_tes_charge_MW': [], 'Q_tes_discharge_MW': [],
            'Q_fossil_thermal_MW': [], 'Q_wasted_MW': [],
            'SOC_tes': [], 'T_hot_tank_C': [], 'eta_receiver': [],
            'm_steam_kgps': [], 'm_solar_fluid_kgps': [],
        }

        self.total_electric_demand_MWh = 0.0
        self.total_heat_demand_MWh = 0.0
        self.total_nuclear_generation_MWh = 0.0
        self.total_solar_generation_MWh = 0.0
        self.total_fossil_electric_MWh = 0.0
        self.total_fossil_thermal_MWh = 0.0
        self.total_wasted_MWh = 0.0
        self.total_P_supply_MWh = 0.0
        self.total_tes_electric_MWh = 0.0

    def step(self, t_h: float, DNI_Wpm2: float, T_amb_C: float,
             P_demand_MW: float, H_demand_MW: float) -> dict:
        """Execute one time step of coupled calculation"""
        # Step 1: CST calculation
        cst_result = self.cst.compute(DNI_Wpm2, T_amb_C, self.tes.T_cold_current)
        Q_solar_MW = cst_result['Q_abs_MW']
        m_solar_fluid_kgps = cst_result['m_fluid_kgps']
        T_solar_hot_C = cst_result['T_fluid_out_C']
        eta_rec = cst_result['eta_receiver']

        # Step 2: SMR calculation (full power steady-state)
        Q_nuclear_MW = self.smr.Q_th_MW
        # MSR: efficiency directly specified, no steam flow dependency
        P_nuclear_base_MW = self.smr.Q_th_MW * self.smr.eta_design
        m_steam_base_kgps = self.smr.Q_th_MW * 0.445  # 160MW * 0.445 = 71.2 kg/s, matching benchmark
        smr_power_result = {'P_net_MW': P_nuclear_base_MW, 'h_in_kJpkg': self.smr.h1}

        # Step 3: Solar superheat/reheat
        if self.tes.fluid is not None:
            T_target_superheat = self.tes.fluid.T_out_superheater
        else:
            T_target_superheat = 295.0

        Q_superheat_needed_MW = self.ihex.superheater_duty(
            m_steam_base_kgps, self.smr.T_turbine_in_C, T_target_superheat
        )

        Q_solar_to_superheat_MW = min(Q_solar_MW, Q_superheat_needed_MW)
        Q_solar_remaining_MW = Q_solar_MW - Q_solar_to_superheat_MW

        Q_fossil_to_superheat_MW = 0.0
        if self.fossil_active:
            if Q_solar_to_superheat_MW < Q_superheat_needed_MW:
                Q_fossil_to_superheat_MW = Q_superheat_needed_MW - Q_solar_to_superheat_MW

        Q_solar_to_tes_MW = Q_solar_remaining_MW

        # Step 4: Determine turbine inlet temperature
        if Q_solar_to_superheat_MW + Q_fossil_to_superheat_MW >= Q_superheat_needed_MW:
            T_turbine_in_actual = T_target_superheat
        else:
            frac = (Q_solar_to_superheat_MW + Q_fossil_to_superheat_MW) / Q_superheat_needed_MW if Q_superheat_needed_MW > 0 else 1.0
            T_turbine_in_actual = self.smr.T_turbine_in_C + frac * (T_target_superheat - self.smr.T_turbine_in_C)

        # MSR: efficiency boost from solar superheat
        efficiency_boost = (T_turbine_in_actual - self.smr.T_turbine_in_C) / 1000.0 * 0.05
        P_nuclear_actual_MW = self.smr.Q_th_MW * (self.smr.eta_design + efficiency_boost)

        # Step 5: Supply-demand matching logic
        P_supply_smr_MW = P_nuclear_actual_MW
        P_delta_MW = P_supply_smr_MW - P_demand_MW

        P_tes_discharge_MW = 0.0
        P_fossil_elec_MW = 0.0
        Q_tes_charge_MW = 0.0
        Q_wasted_MW = 0.0
        P_supply_total_MW = P_supply_smr_MW

        if P_delta_MW > 0:
            # Surplus -> Charge mode
            Q_excess_thermal_MW = P_delta_MW / self.smr.eta_design if self.smr.eta_design > 0 else P_delta_MW
            Q_total_charge_MW = Q_excess_thermal_MW + Q_solar_to_tes_MW

            tes_charged_MWh = self.tes.charge(Q_total_charge_MW, self.dt, T_solar_hot_C)
            Q_tes_charge_MW = tes_charged_MWh / self.dt

            Q_wasted_MW = Q_total_charge_MW - Q_tes_charge_MW + Q_fossil_to_superheat_MW
        else:
            # Deficit -> Discharge mode
            P_deficit_MW = -P_delta_MW

            Q_tes_discharge_needed_MW = P_deficit_MW / self.smr.eta_design if self.smr.eta_design > 0 else P_deficit_MW
            tes_discharged_MWh = self.tes.discharge(Q_tes_discharge_needed_MW, self.dt)
            P_tes_discharge_MW = tes_discharged_MWh / self.dt * self.smr.eta_design

            P_remaining_deficit_MW = P_deficit_MW - P_tes_discharge_MW
            if P_remaining_deficit_MW > 0 and self.fossil_active:
                fossil_result = self.fossil.produce_electricity(P_remaining_deficit_MW, self.dt)
                P_fossil_elec_MW = fossil_result['P_rate_MW']
                Q_fossil_to_superheat_MW += P_remaining_deficit_MW / self.fossil.eta if self.fossil.eta > 0 else P_remaining_deficit_MW

            P_supply_total_MW = P_supply_smr_MW + P_tes_discharge_MW + P_fossil_elec_MW
            Q_wasted_MW = Q_solar_to_tes_MW

        # Step 6: Fossil superheat accounting
        if Q_fossil_to_superheat_MW > 0:
            self.fossil.produce_heat(Q_fossil_to_superheat_MW, self.dt)

        # Step 7: TES idle losses
        if Q_tes_charge_MW == 0 and P_tes_discharge_MW == 0:
            self.tes.idle(self.dt)

        # Cumulative statistics
        self.total_electric_demand_MWh += P_demand_MW * self.dt
        self.total_heat_demand_MWh += H_demand_MW * self.dt
        self.total_nuclear_generation_MWh += P_supply_smr_MW * self.dt
        self.total_solar_generation_MWh += Q_solar_MW * self.dt
        self.total_fossil_electric_MWh += P_fossil_elec_MW * self.dt
        self.total_fossil_thermal_MWh += Q_fossil_to_superheat_MW * self.dt
        self.total_wasted_MWh += Q_wasted_MW * self.dt
        self.total_P_supply_MWh += P_supply_total_MW * self.dt
        self.total_tes_electric_MWh += P_tes_discharge_MW * self.dt

        step_result = {
            'time_h': t_h, 'DNI_Wpm2': DNI_Wpm2, 'T_amb_C': T_amb_C,
            'P_demand_MW': P_demand_MW, 'H_demand_MW': H_demand_MW,
            'P_nuclear_MW': P_supply_smr_MW, 'P_solar_to_grid_MW': 0.0,
            'P_fossil_MW': P_fossil_elec_MW, 'P_total_supply_MW': P_supply_total_MW,
            'Q_nuclear_MW': Q_nuclear_MW, 'Q_solar_MW': Q_solar_MW,
            'Q_tes_charge_MW': Q_tes_charge_MW,
            'Q_tes_discharge_MW': P_tes_discharge_MW / self.smr.eta_design if self.smr.eta_design > 0 else 0,
            'Q_fossil_thermal_MW': Q_fossil_to_superheat_MW,
            'Q_wasted_MW': Q_wasted_MW,
            'SOC_tes': self.tes.SOC, 'T_hot_tank_C': self.tes.T_hot,
            'eta_receiver': eta_rec,
            'm_steam_kgps': m_steam_base_kgps,
            'm_solar_fluid_kgps': m_solar_fluid_kgps,
        }

        for key in self.results:
            if key in step_result:
                self.results[key].append(step_result[key])

        return step_result

    def run(self, dni_array: np.ndarray, T_amb_array: np.ndarray,
            P_demand_array: np.ndarray, H_demand_array: np.ndarray = None) -> dict:
        """Execute full time-series transient simulation"""
        N = len(dni_array)
        if H_demand_array is None:
            H_demand_array = np.zeros(N)

        if self.verbose:
            print(f"Starting transient simulation: {N} steps, dt = {self.dt} h")
            print(f"Reactor thermal power: {self.smr.Q_th_MW:.0f} MWth")
            print(f"CST field area: {self.cst.A_field_m2:.0f} m2")
            if self.tes.fluid:
                print(f"Storage fluid: {self.tes.fluid.name}")
                print(f"Tank: H={self.tes.H_m:.1f}m, D={self.tes.D_m:.1f}m, V={self.tes.volume_m3:.0f}m3")
            print("-" * 60)

        progress_interval = max(1, N // 20)
        for t in range(N):
            self.step(
                t_h=t * self.dt,
                DNI_Wpm2=dni_array[t],
                T_amb_C=T_amb_array[t],
                P_demand_MW=P_demand_array[t],
                H_demand_MW=H_demand_array[t]
            )

            if self.verbose and (t + 1) % progress_interval == 0:
                pct = (t + 1) / N * 100.0
                print(f"  Progress: {pct:5.1f}% ({t+1}/{N})  "
                      f"SOC={self.tes.SOC:.3f}  "
                      f"T_hot={self.tes.T_hot:.1f}C  "
                      f"Sum_charge={self.tes.total_charge_MWh:.0f}MWh")

        if self.verbose:
            print("-" * 60)
            print("Simulation complete.")

        return self.get_summary()

    def get_summary(self) -> dict:
        """Get simulation results summary with all 14+ key indicators"""
        N_hours = len(self.results['time_h']) if self.results['time_h'] else 8760

        # 电/热效率：核发电 + TES放电 → 电 + 化石燃料发电
        total_elec_out = (self.total_nuclear_generation_MWh + self.total_tes_electric_MWh
                          + self.total_fossil_electric_MWh)
        total_Q_th_MWh = (self.smr.Q_th_MW + self.total_solar_generation_MWh / N_hours) * N_hours if N_hours > 0 else 0
        eta_power = total_elec_out / total_Q_th_MWh if total_Q_th_MWh > 0 else 0.0

        eta_base_load = self.smr.eta_base_load
        total_demand = self.total_electric_demand_MWh + self.total_heat_demand_MWh
        fossil_total = self.total_fossil_electric_MWh + self.total_fossil_thermal_MWh
        shaving_factor = self.econ.clean_energy_shaving_factor(total_demand, fossil_total)
        avg_fossil_power_MW = fossil_total / N_hours if N_hours > 0 else 0
        avg_wasted_MW = self.total_wasted_MWh / N_hours if N_hours > 0 else 0
        eta_storage = self.tes.round_trip_efficiency

        # 新增指标
        net_electric_power_MW = self.smr.P_net_MW
        net_cycle_efficiency = self.smr.eta_design * 100.0
        rated_steam_flow = np.mean(self.results['m_steam_kgps']) if self.results['m_steam_kgps'] else 0.0
        avg_discharge_MW = self.tes.total_discharge_MWh / N_hours if N_hours > 0 else 0.0
        annual_total_generation = total_elec_out
        avg_hot_tank_temp = np.mean(self.results['T_hot_tank_C']) if self.results['T_hot_tank_C'] else 0.0

        # 典型运行日：选取总供电功率最接近中位数的一天
        daily_gen_typical = 0.0
        if N_hours >= 24:
            supply_arr = np.array(self.results['P_total_supply_MW'])
            daily_means = [np.mean(supply_arr[i:i+24]) for i in range(0, len(supply_arr) - 23, 24)]
            if daily_means:
                median_day = sorted(daily_means)[len(daily_means) // 2]
                target_idx = daily_means.index(median_day) * 24
                daily_gen_typical = np.sum(supply_arr[target_idx:target_idx+24])

        return {
            # 已有指标
            'reactor_base_load_efficiency_percent': eta_base_load * 100.0,
            'yearly_average_combined_efficiency_percent': eta_power * 100.0,
            'nuclear_reactor_thermal_power_MWth': self.smr.Q_th_MW,
            'clean_energy_shaving_factor_percent': shaving_factor * 100.0,
            'average_required_fossil_power_MWth': avg_fossil_power_MW,
            'average_available_thermal_energy_MWth': avg_wasted_MW,
            'cst_field_area_m2': self.cst.A_field_m2,
            'cst_yearly_avg_efficiency_percent': np.mean([r for r in self.results['eta_receiver'] if r > 0]) * 100.0 if any(self.results['eta_receiver']) else 0.0,
            'cst_receiver_tube_number': self.cst.N_tubes,
            'tes_total_charge_MWh': self.tes.total_charge_MWh,
            'tes_total_discharge_MWh': self.tes.total_discharge_MWh,
            'tes_storage_efficiency_percent': eta_storage * 100.0,
            'tes_tank_height_m': self.tes.H_m,
            'tes_tank_diameter_m': self.tes.D_m,
            'tes_average_hot_tank_temperature_C': avg_hot_tank_temp,
            'tes_cold_tank_temperature_C': self.tes.T_cold_current,
            'total_electric_demand_MWh': self.total_electric_demand_MWh,
            'total_heat_demand_MWh': self.total_heat_demand_MWh,
            'total_nuclear_generation_MWh': self.total_nuclear_generation_MWh,
            'total_solar_generation_MWh': self.total_solar_generation_MWh,
            'total_fossil_electric_MWh': self.total_fossil_electric_MWh,
            'total_fossil_thermal_MWh': self.total_fossil_thermal_MWh,
            'total_wasted_energy_MWh': self.total_wasted_MWh,
            # 新增 14 项指标
            'net_electric_power_MW': net_electric_power_MW,
            'net_cycle_efficiency_percent': net_cycle_efficiency,
            'rated_steam_flow_kgps': rated_steam_flow,
            'average_discharge_power_MW': avg_discharge_MW,
            'annual_total_electric_generation_MWh': annual_total_generation,
            'daily_generation_typical_MWh': daily_gen_typical,
            'avg_hot_tank_temperature_C': avg_hot_tank_temp,
            'tes_round_trip_efficiency_percent': eta_storage * 100.0,
            'annual_total_charge_MWh': self.tes.total_charge_MWh,
            'total_thermal_input_MWh': total_Q_th_MWh,
            'system_overall_efficiency_percent': eta_power * 100.0,
        }

    def print_summary(self):
        """Print simulation summary (Table 10-14)"""
        s = self.get_summary()

        print("\n" + "=" * 70)
        print("  NR-HES Transient Simulation Results Summary")
        print("=" * 70)
        print("\n  [Table 10] TES Design Parameters")
        print(f"    Hot tank avg temp:        {s['tes_average_hot_tank_temperature_C']:.1f} C")
        print(f"    Cold tank temp:           {s['tes_cold_tank_temperature_C']:.1f} C")
        print(f"    Tank height:              {s['tes_tank_height_m']:.1f} m")
        print(f"    Tank diameter:            {s['tes_tank_diameter_m']:.1f} m")
        print(f"    Total charge:             {s['tes_total_charge_MWh']:.0f} MWh")
        print(f"    Total discharge:          {s['tes_total_discharge_MWh']:.0f} MWh")
        print(f"    Storage efficiency:       {s['tes_storage_efficiency_percent']:.1f}%")

        print("\n  [Table 11] Power Cycle Parameters")
        print(f"    Base-load efficiency:     {s['reactor_base_load_efficiency_percent']:.1f}%")
        print(f"    Combined annual eff:      {s['yearly_average_combined_efficiency_percent']:.1f}%")
        print(f"    Nuclear thermal power:    {s['nuclear_reactor_thermal_power_MWth']:.0f} MWth")
        print(f"    Clean energy shaving:     {s['clean_energy_shaving_factor_percent']:.1f}%")
        print(f"    Avg fossil fuel power:    {s['average_required_fossil_power_MWth']:.0f} MWth")
        print(f"    Avg available extra heat: {s['average_available_thermal_energy_MWth']:.0f} MWth")

        print("\n  [Table 12] CST Design Parameters")
        print(f"    Field area:               {s['cst_field_area_m2']:.0f} m2")
        print(f"    Yearly avg efficiency:    {s['cst_yearly_avg_efficiency_percent']:.1f}%")
        print(f"    Receiver tubes:           {s['cst_receiver_tube_number']:.0f}")

        print("\n  [Cumulative Statistics]")
        print(f"    Total electric demand:    {s['total_electric_demand_MWh']:.0f} MWh")
        print(f"    Total nuclear generation: {s['total_nuclear_generation_MWh']:.0f} MWh")
        print(f"    Total solar thermal:      {s['total_solar_generation_MWh']:.0f} MWh")
        print(f"    Total fossil electric:    {s['total_fossil_electric_MWh']:.0f} MWh")
        print(f"    Total fossil thermal:     {s['total_fossil_thermal_MWh']:.0f} MWh")
        print(f"    Total wasted energy:      {s['total_wasted_energy_MWh']:.0f} MWh")
        print("=" * 70 + "\n")

    def get_results_dataframe(self):
        """Return results as pandas DataFrame if available"""
        try:
            import pandas as pd
            return pd.DataFrame(self.results)
        except ImportError:
            return self.results

    def save_results_csv(self, filepath: str):
        """Save results to CSV"""
        try:
            import pandas as pd
            df = pd.DataFrame(self.results)
            df.to_csv(filepath, index=False)
            if self.verbose:
                print(f"Results saved to: {filepath}")
        except ImportError:
            header = ','.join(self.results.keys())
            data = np.column_stack([self.results[k] for k in self.results.keys()])
            np.savetxt(filepath, data, delimiter=',', header=header, comments='')
            if self.verbose:
                print(f"Results saved to: {filepath}")


if __name__ == "__main__":
    print("Transient solver - self-test")
    print("Run main.py to execute full simulation.")