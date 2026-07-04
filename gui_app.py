"""
SMR + CST + TES Transient Simulation — GUI (MSR + SolarSalt, CNY)
Double-click to run, fill in parameters, click Start, auto-generate results and charts
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
import threading
import numpy as np
import sys
import os
import time
import yaml
from datetime import datetime

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

from models.nuclear_cycle import NuclearPowerCycle
from models.cst_plant import ConcentratedSolarTower
from models.tes_system import ThermalEnergyStorage
from models.heat_exchanger import IntermediateHeatExchanger
from models.fossil_backup import FossilFuelBackup
from models.economics import EconomicAnalyzer
from solver.transient_solver import NRHESTransientSolver
from fluid_library import get_fluid
from io_utils.demand_loader import (
    generate_ottawa_dni, generate_ontario_electric_demand,
    generate_residential_heat_demand, generate_ambient_temperature
)
from visualization.plots import NRHESVisualizer


# ===================== Color Palette =====================
COLOR_BG = '#F5F6FA'
COLOR_PRIMARY = '#1A237E'
COLOR_ACCENT = '#00C853'
COLOR_WARNING = '#FF6D00'
COLOR_TEXT = '#212121'
COLOR_SECONDARY = '#546E7A'
FONT_TITLE = ('Microsoft YaHei', 14, 'bold')
FONT_SECTION = ('Microsoft YaHei', 11, 'bold')
FONT_BODY = ('Microsoft YaHei', 9)
FONT_MONO = ('Consolas', 9)


class SimulationGUI:
    """NR-HES Simulation GUI"""

    def __init__(self, root):
        self.root = root
        self.root.title("SMR + CST + TES Transient Simulation — MSR Edition")
        self.root.geometry("1000x720")
        self.root.configure(bg=COLOR_BG)
        self.sim_running = False
        self.results = None
        self.data_files = {'DNI': None, 'P_elec': None, 'Q_heat': None, 'T_amb': None}
        self._create_widgets()
        self.log("Software ready. Set parameters and click [Start Simulation].")

    def _create_widgets(self):
        # === Top: Title ===
        title_frame = tk.Frame(self.root, bg=COLOR_PRIMARY, pady=12)
        title_frame.pack(fill='x')
        tk.Label(title_frame, text="SMR + CST + TES Coupled Transient Simulation",
                 font=FONT_TITLE, fg='white', bg=COLOR_PRIMARY).pack()
        tk.Label(title_frame, text="MSR Molten Salt Reactor | Gemasolar CST | SolarSalt Two-Tank TES",
                 font=FONT_BODY, fg='#B0BEC5', bg=COLOR_PRIMARY).pack()

        # === Main: Left params + Right controls ===
        main_frame = tk.Frame(self.root, bg=COLOR_BG)
        main_frame.pack(fill='both', expand=True, padx=10, pady=8)

        left_frame = tk.LabelFrame(main_frame, text=" Parameters ", bg=COLOR_BG, fg=COLOR_TEXT,
                                    font=FONT_SECTION, padx=10, pady=8)
        left_frame.pack(side='left', fill='y')
        self._create_params(left_frame)

        right_frame = tk.Frame(main_frame, bg=COLOR_BG)
        right_frame.pack(side='right', fill='both', expand=True, padx=10)
        self._create_controls(right_frame)

        # === Bottom: Status bar ===
        status_frame = tk.Frame(self.root, bg=COLOR_PRIMARY, height=28)
        status_frame.pack(fill='x', side='bottom')
        self.status_indicator = tk.Label(status_frame, text="  Ready", fg='white', bg=COLOR_PRIMARY,
                                          font=FONT_BODY, anchor='w')
        self.status_indicator.pack(fill='x', padx=10)

    def _create_params(self, parent):
        """Parameter inputs"""
        # Tooltip helper
        def create_tooltip(widget, text):
            """Create a tooltip for a widget"""
            tip = None
            def enter(event):
                nonlocal tip
                x = widget.winfo_rootx() + 25
                y = widget.winfo_rooty() + widget.winfo_height() + 2
                tip = tk.Toplevel(widget)
                tip.wm_overrideredirect(True)
                tip.wm_geometry(f"+{x}+{y}")
                tip.configure(bg='#FFFFDD')
                tk.Label(tip, text=text, bg='#FFFFDD', fg='#333333',
                         font=FONT_BODY, justify='left', padx=6, pady=4,
                         wraplength=320).pack()
            def leave(event):
                nonlocal tip
                if tip:
                    tip.destroy()
                    tip = None
            widget.bind('<Enter>', enter)
            widget.bind('<Leave>', leave)

        # === SMR Section ===
        row = tk.Frame(parent, bg=COLOR_BG)
        row.pack(anchor='w')
        tk.Label(row, text="SMR 熔盐堆 (MSR)", font=FONT_SECTION, bg=COLOR_BG, fg=COLOR_PRIMARY).pack(anchor='w', pady=(0,1))
        tk.Label(row, text="  小型模块化熔盐堆参数，Brayton 循环发电",
                 font=('Microsoft YaHei', 7), bg=COLOR_BG, fg=COLOR_SECONDARY).pack(anchor='w', pady=(0,3))
        self._field(row, "反应堆热功率 [MWth]:", "160.0", "smr_power",
                     "MSR 熔盐堆热功率输出，典型值 100-300 MWth")
        self._field(row, "出口温度 [°C]:", "700.0", "smr_outlet",
                     "反应堆出口熔盐温度，典型值 600-750°C")
        self._field(row, "涡轮入口温度 [°C]:", "650.0", "smr_turbine",
                     "Brayton 循环涡轮入口温度，典型值 550-700°C")
        self._field(row, "循环效率 [-]:", "0.28", "smr_eff",
                     "热-电转换效率，Brayton 循环典型值 0.28 (700°C)")

        # === CST Section ===
        row2 = tk.Frame(parent, bg=COLOR_BG)
        row2.pack(anchor='w', pady=(10,0))
        tk.Label(row2, text="CST 聚光太阳能塔", font=FONT_SECTION, bg=COLOR_BG, fg=COLOR_PRIMARY).pack(anchor='w', pady=(0,1))
        tk.Label(row2, text="  塔式定日镜场，SolarSalt 传热工质",
                 font=('Microsoft YaHei', 7), bg=COLOR_BG, fg=COLOR_SECONDARY).pack(anchor='w', pady=(0,3))
        self._field(row2, "镜场面积 [m²]:", "150000", "cst_area",
                     "定日镜场总面积，Gemasolar 参考值 ≈ 306,000 m²")
        self._field(row2, "聚光比 [-]:", "900", "cst_ratio",
                     "聚光系统聚光比，典型值 600-1,200")
        self._field(row2, "场效率 [-]:", "0.60", "cst_eff",
                     "镜场综合光学效率，含余弦/遮挡/反射等损失，典型值 0.50-0.65")

        # === TES Section ===
        row3 = tk.Frame(parent, bg=COLOR_BG)
        row3.pack(anchor='w', pady=(10,0))
        tk.Label(row3, text="TES 储热系统", font=FONT_SECTION, bg=COLOR_BG, fg=COLOR_PRIMARY).pack(anchor='w', pady=(0,1))
        tk.Label(row3, text="  双罐显热储热，SolarSalt 熔盐或导热油",
                 font=('Microsoft YaHei', 7), bg=COLOR_BG, fg=COLOR_SECONDARY).pack(anchor='w', pady=(0,3))
        self._field(row3, "储罐高度 [m]:", "14.0", "tes_h",
                     "TES 储热罐高度，优化范围 6-20 m，影响成本和效率")
        self._field(row3, "储罐直径 [m]:", "10.0", "tes_d",
                     "TES 储热罐直径，优化范围 5-15 m")

        fr_fluid = tk.Frame(row3, bg=COLOR_BG)
        fr_fluid.pack(fill='x', pady=1)
        tk.Label(fr_fluid, text="储热工质:", bg=COLOR_BG, font=FONT_BODY, width=20, anchor='e').pack(side='left', padx=(0,3))
        self.fluid_var = tk.StringVar(value="SolarSalt")
        combo = ttk.Combobox(fr_fluid, textvariable=self.fluid_var, width=14, state='readonly',
                      values=["Therminol","Dowtherm","SolarSalt","Hitec","HitecXL"])
        combo.pack(side='left')
        create_tooltip(combo, "储热流体类型:\n"
                       "• SolarSalt (60%NaNO₃+40%KNO₃) — 熔盐，工作范围 220-600°C\n"
                       "• Hitec — 三元熔盐，低熔点 142°C\n"
                       "• HitecXL — 低熔点三元盐, 工作范围 120-500°C\n"
                       "• Therminol — 导热油, 工作范围 12-400°C\n"
                       "• Dowtherm — 导热油, 工作范围 15-393°C")

        # === Simulation Section ===
        row4 = tk.Frame(parent, bg=COLOR_BG)
        row4.pack(anchor='w', pady=(10,0))
        tk.Label(row4, text="仿真控制", font=FONT_SECTION, bg=COLOR_BG, fg=COLOR_PRIMARY).pack(anchor='w', pady=(0,1))
        tk.Label(row4, text="  时间步长固定为 1 小时",
                 font=('Microsoft YaHei', 7), bg=COLOR_BG, fg=COLOR_SECONDARY).pack(anchor='w', pady=(0,3))
        self._field(row4, "仿真时长 [h]:", "8760", "hours",
                     "仿真总小时数：全年=8,760h，一季度=2,184h，一周=168h，一天=24h")

        # === Input Data Section ===
        row5 = tk.Frame(parent, bg=COLOR_BG)
        row5.pack(anchor='w', pady=(10,0))
        tk.Label(row5, text="输入气象与负荷数据 (CSV)", font=FONT_SECTION, bg=COLOR_BG, fg=COLOR_PRIMARY).pack(anchor='w', pady=(0,1))
        tk.Label(row5, text="  可选，每列格式: 时间[h]，数值。留空则使用合成默认数据",
                 font=('Microsoft YaHei', 7), bg=COLOR_BG, fg=COLOR_SECONDARY).pack(anchor='w', pady=(0,3))
        data_tooltips = {
            "DNI": "CSV 格式: 时间[h], DNI[W/m²]\n留空将自动生成 Ottawa 地区合成 DNI 辐照数据",
            "P_elec": "CSV 格式: 时间[h], 电力需求[MW]\n留空将自动生成 Ontario 电网合成电力负荷曲线",
            "Q_heat": "CSV 格式: 时间[h], 热负荷[MW]\n留空将自动生成住宅供暖热需求合成数据",
            "T_amb": "CSV 格式: 时间[h], 环境温度[°C]\n留空将自动生成 Ottawa 合成气温数据",
        }
        for label, key in [("DNI 直射辐照 [W/m²]:", "DNI"), ("电力负荷 [MW]:", "P_elec"),
                            ("热负荷 [MW]:", "Q_heat"), ("环境温度 [°C]:", "T_amb")]:
            fr = tk.Frame(row5, bg=COLOR_BG)
            fr.pack(fill='x', pady=1)
            tk.Label(fr, text=label, bg=COLOR_BG, font=FONT_BODY, width=20, anchor='e').pack(side='left')
            btn = tk.Button(fr, text="选择文件", font=FONT_BODY, bg='#ECEFF1', padx=4,
                            command=lambda k=key: self._select_file(k))
            btn.pack(side='left', padx=3)
            create_tooltip(btn, data_tooltips.get(key, "选择 CSV 数据文件，留空使用默认合成数据"))
            info_btn = tk.Button(fr, text="?", font=('Microsoft YaHei', 8, 'bold'), bg='#B0BEC5', fg='white',
                                 width=2, padx=0, pady=0, relief='flat',
                                 command=lambda k=key: messagebox.showinfo(f"{k} 格式说明", data_tooltips.get(k, "")))
            info_btn.pack(side='left', padx=1)
            lbl = tk.Label(fr, text="(自动)", bg=COLOR_BG, fg=COLOR_SECONDARY, font=FONT_BODY)
            lbl.pack(side='left')
            setattr(self, f"lbl_{key}", lbl)
            setattr(self, f"btn_{key}", btn)

    @staticmethod
    def _create_tooltip(widget, text):
        """Create a tooltip for a widget"""
        tip = None
        def enter(event):
            nonlocal tip
            x = widget.winfo_rootx() + 25
            y = widget.winfo_rooty() + widget.winfo_height() + 2
            tip = tk.Toplevel(widget)
            tip.wm_overrideredirect(True)
            tip.wm_geometry(f"+{x}+{y}")
            tip.configure(bg='#FFFFDD')
            tk.Label(tip, text=text, bg='#FFFFDD', fg='#333333',
                     font=FONT_BODY, justify='left', padx=6, pady=4,
                     wraplength=320).pack()
        def leave(event):
            nonlocal tip
            if tip:
                tip.destroy()
                tip = None
        widget.bind('<Enter>', enter)
        widget.bind('<Leave>', leave)

    def _field(self, parent, label, default, name, tooltip=None):
        fr = tk.Frame(parent, bg=COLOR_BG)
        fr.pack(fill='x', pady=1)
        tk.Label(fr, text=label, bg=COLOR_BG, font=FONT_BODY, width=20, anchor='e').pack(side='left', padx=(0,3))
        var = tk.StringVar(value=default)
        setattr(self, name, var)
        entry = tk.Entry(fr, textvariable=var, width=12, font=FONT_BODY)
        entry.pack(side='left')
        if tooltip:
            self._create_tooltip(entry, tooltip)

    def _select_file(self, key):
        path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if path:
            self.data_files[key] = path
            getattr(self, f"lbl_{key}").config(text=os.path.basename(path), fg=COLOR_ACCENT)

    def _create_controls(self, parent):
        """Control buttons + results area"""
        btn_frame = tk.Frame(parent, bg=COLOR_BG)
        btn_frame.pack(fill='x', pady=(0,8))

        self.start_btn = tk.Button(btn_frame, text="  Start Simulation  ", font=FONT_SECTION,
                                    bg=COLOR_ACCENT, fg='white', padx=20, pady=6, border=0,
                                    command=self.start_simulation, cursor='hand2')
        self.start_btn.pack(side='left', padx=4)

        self.plot_btn = tk.Button(btn_frame, text="  Generate Charts  ", font=FONT_SECTION,
                                   state='disabled', bg='#ECEFF1', padx=12, pady=6, border=0,
                                   command=self.generate_plots, cursor='hand2')
        self.plot_btn.pack(side='left', padx=4)

        self.open_btn = tk.Button(btn_frame, text="  Open Results  ", font=FONT_SECTION,
                                   state='disabled', bg='#ECEFF1', padx=12, pady=6, border=0,
                                   command=self.open_results_dir, cursor='hand2')
        self.open_btn.pack(side='left', padx=4)

        # Progress
        prog_frame = tk.Frame(parent, bg=COLOR_BG)
        prog_frame.pack(fill='x', pady=(0,5))
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(prog_frame, variable=self.progress_var, maximum=100, mode='determinate')
        self.progress_bar.pack(fill='x')
        self.status_label = tk.Label(prog_frame, text="Ready", font=FONT_BODY, bg=COLOR_BG, fg=COLOR_SECONDARY)
        self.status_label.pack()

        # Log area
        log_frame = tk.LabelFrame(parent, text=" Log & Results ", bg=COLOR_BG, fg=COLOR_TEXT, font=FONT_SECTION, padx=5, pady=5)
        log_frame.pack(fill='both', expand=True)
        self.log_text = scrolledtext.ScrolledText(log_frame, width=65, height=18,
                                                    font=FONT_MONO, bg='#263238', fg='#B2FF59',
                                                    insertbackground='white')
        self.log_text.pack(fill='both', expand=True)

    def log(self, msg):
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert('end', f"[{ts}] {msg}\n")
        self.log_text.see('end')
        self.root.update_idletasks()

    def get_config(self):
        return {
            'simulation': {'time_step_hours': 1, 'total_hours': int(float(self.hours.get()))},
            'system': {'fossil_backup_active': False},
            'smr': {
                'reactor_thermal_power_MWth': float(self.smr_power.get()),
                'reactor_outlet_temperature_C': float(self.smr_outlet.get()),
                'turbine_inlet_temperature_C': float(self.smr_turbine.get()),
                'cycle_efficiency': float(self.smr_eff.get()),
                'heat_exchanger_loss_frac': 0.05, 'heat_exchanger_TTD_C': 5.0,
            },
            'cst': {
                'field_area_m2': float(self.cst_area.get()),
                'concentration_ratio': float(self.cst_ratio.get()),
                'field_efficiency': float(self.cst_eff.get()),
                'receiver_emissivity': 0.88, 'view_factor': 0.80, 'reflectivity': 0.06,
                'tube_outer_diameter_m': 0.04, 'tube_thermal_conductivity_WpmK': 23.9,
                'wind_velocity_mps': 7.0, 'ambient_temperature_C': 25.0,
                'max_fluid_velocity_mps': 4.0, 'receiver_inlet_temperature_C': 290.0,
                'receiver_outlet_temperature_C': 565.0, 'control_strategy': 'FT',
                'dni_threshold_Wpm2': 350.0,
            },
            'tes': {
                'tank_height_m': float(self.tes_h.get()),
                'tank_diameter_m': float(self.tes_d.get()),
                'tank_insulation_U_Wpm2K': 0.5,
                'storage_fluid': self.fluid_var.get(),
                'auxiliary_heater_efficiency': 0.95,
            },
            'economics': self._load_economics_config(),
        }

    def _load_economics_config(self) -> dict:
        """Load economics parameters from config.yaml (CNY)"""
        default_econ = {
            'lifetime_years': 20, 'interest_rate': 0.05,
            'smr_cost_per_kWe_CNY': 36909.6, 'natural_gas_cost_per_kWe_CNY': 8054.8,
            'cst_cost_per_kWe_CNY': 48880.0,
            'electricity_peak_rate_CNY_per_kWh': 0.78,
            'electricity_offpeak_rate_CNY_per_kWh': 0.416,
            'natural_gas_fuel_cost_CNY_per_MWh': 156.0,
        }
        config_path = os.path.join(PROJECT_DIR, 'config.yaml')
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    yaml_config = yaml.safe_load(f)
                if yaml_config and 'economics' in yaml_config:
                    econ_yaml = yaml_config['economics']
                    default_econ['lifetime_years'] = econ_yaml.get('lifetime_years', default_econ['lifetime_years'])
                    default_econ['interest_rate'] = econ_yaml.get('interest_rate', default_econ['interest_rate'])
                    default_econ['smr_cost_per_kWe_CNY'] = econ_yaml.get('smr_cost_per_kWe_CNY', default_econ['smr_cost_per_kWe_CNY'])
                    default_econ['natural_gas_cost_per_kWe_CNY'] = econ_yaml.get('natural_gas_cost_per_kWe_CNY', default_econ['natural_gas_cost_per_kWe_CNY'])
                    default_econ['cst_cost_per_kWe_CNY'] = econ_yaml.get('cst_cost_per_kWe_CNY', default_econ['cst_cost_per_kWe_CNY'])
                    default_econ['electricity_peak_rate_CNY_per_kWh'] = econ_yaml.get('electricity_peak_rate_CNY_per_kWh', default_econ['electricity_peak_rate_CNY_per_kWh'])
                    default_econ['electricity_offpeak_rate_CNY_per_kWh'] = econ_yaml.get('electricity_offpeak_rate_CNY_per_kWh', default_econ['electricity_offpeak_rate_CNY_per_kWh'])
                    default_econ['natural_gas_fuel_cost_CNY_per_MWh'] = econ_yaml.get('natural_gas_fuel_cost_CNY_per_MWh', default_econ['natural_gas_fuel_cost_CNY_per_MWh'])
        except Exception as e:
            self.log(f"Warning: failed to load config.yaml, using default CNY values: {e}")
        return default_econ

    def start_simulation(self):
        if self.sim_running: return messagebox.showwarning("Busy", "Simulation is already running.")
        self.sim_running = True
        self.start_btn.config(state='disabled', text=" Running... ")
        self.plot_btn.config(state='disabled')
        self.open_btn.config(state='disabled')
        self.status_indicator.config(text="  Running...", fg='#FFD600')
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        try:
            config = self.get_config()
            n_hours = config['simulation']['total_hours']
            self.log("=" * 60)
            self.log("  SMR + CST + TES Transient Simulation Started")
            self.log("=" * 60)

            self.log("[Step 1/5] Initializing models...")
            self.update_progress(5, "Initializing...")

            fluid = get_fluid(config['tes']['storage_fluid'])
            config['tes']['cold_tank_temperature_C'] = fluid.T_cold_tank
            config['tes']['hot_tank_initial_temperature_C'] = fluid.T_out_solar

            smr = NuclearPowerCycle(config['smr'])
            self.log(f"  MSR: {smr.Q_th_MW:.0f} MWth, P_net={smr.P_net_MW:.1f} MWe, eta={smr.eta_design*100:.1f}%")

            cst = ConcentratedSolarTower(config['cst'], storage_fluid=fluid)
            tes = ThermalEnergyStorage(config['tes'], storage_fluid=fluid)
            self.log(f"  TES: H={tes.H_m:.1f}m D={tes.D_m:.1f}m V={tes.volume_m3:.0f}m3")

            ihex = IntermediateHeatExchanger(effectiveness=0.85, heat_loss_frac=0.05, TTD_C=5.0)
            fossil = FossilFuelBackup()
            econ = EconomicAnalyzer(config['economics'])

            self.log(f"[Step 2/5] Loading input data ({n_hours} hours)...")
            self.update_progress(10, "Loading data...")

            seed = int(time.time()) % 1000
            dni = self._load_data('DNI', lambda: generate_ottawa_dni(n_hours, seed=seed))
            P_demand = self._load_data('P_elec', lambda: generate_ontario_electric_demand(n_hours, base_load_MW=smr.P_net_MW*0.7, seed=seed+1))
            H_demand = self._load_data('Q_heat', lambda: generate_residential_heat_demand(n_hours, seed=seed+2))
            T_amb = self._load_data('T_amb', lambda: generate_ambient_temperature(n_hours, seed=seed+3))

            self.log(f"  DNI mean={np.mean(dni):.1f} W/m2, P_demand mean={np.mean(P_demand):.1f} MW")

            self.log(f"[Step 3/5] Running solver...")
            self.update_progress(15, "Simulating...")

            solver = NRHESTransientSolver(smr=smr, cst=cst, tes=tes, ihex=ihex,
                                          fossil=fossil, economics=econ, dt_hours=1.0, verbose=False,
                                          fossil_backup_active=False)
            gap = max(1, n_hours//50)
            for t in range(n_hours):
                solver.step(t_h=t, DNI_Wpm2=dni[t], T_amb_C=T_amb[t],
                           P_demand_MW=P_demand[t], H_demand_MW=H_demand[t])
                if t % gap == 0: self.update_progress(15+(t/n_hours)*65, f"Hour {t+1}/{n_hours}")

            summary = solver.get_summary()
            self.update_progress(80, "Done.")

            # Economic
            self.log("[Step 4/5] Economic analysis...")
            self.update_progress(85, "Economics...")
            fluid_cost = fluid.cost_per_m3 * tes.volume_m3 * 2.0
            tes_costs = econ.tes_capital_cost(fluid_cost, tes.H_m, tes.D_m)
            tes_annual = econ.tes_total_annual_cost(tes_costs['total_capital_CNY'])
            tes_profit = econ.tes_profit(tes.total_discharge_MWh, tes.round_trip_efficiency, tes_annual)
            I_pct = econ.performance_index(tes_profit, summary['yearly_average_combined_efficiency_percent']/100,
                                            summary['tes_storage_efficiency_percent']/100,
                                            econ.annualize_capital(tes_costs['total_capital_CNY']))

            # === 全部 14+ 项关键指标 ===
            s = summary
            annual_gen = s.get('annual_total_electric_generation_MWh', solver.total_nuclear_generation_MWh)
            total_heat_demand = s.get('total_heat_demand_MWh', 0)
            total_thermal_input = s.get('total_thermal_input_MWh', 0)

            # 经济指标
            total_capital = econ.system_total_capital(smr.P_net_MW, 0, 0, tes_costs['total_capital_CNY'])
            annualized_capital = econ.system_annualized_cost(total_capital)
            annual_revenue = econ.annual_revenue(annual_gen, total_heat_demand)
            annual_fuel_cost = econ.ng_fuel_cost_per_MWh * (s.get('total_fossil_electric_MWh', 0) + s.get('total_fossil_thermal_MWh', 0))
            annual_cashflow = annual_revenue - annual_fuel_cost - tes_annual
            npv_val = econ.npv(total_capital, annual_cashflow)
            irr_val = econ.irr(total_capital, annual_cashflow)
            lcoe_val = econ.lcoe(annualized_capital + tes_annual, annual_gen, annual_fuel_cost)

            self.log("=" * 60)
            self.log("  技术性能指标")
            self.log("=" * 60)
            self.log(f"  熔盐堆净发电功率:        {s.get('net_electric_power_MW', smr.P_net_MW):.1f} MW")
            self.log(f"  熔盐堆净发电效率:        {s.get('net_cycle_efficiency_percent', smr.eta_design*100):.1f}%")
            self.log(f"  额定蒸汽流量:            {s.get('rated_steam_flow_kgps', 0):.1f} kg/s")
            self.log(f"  热罐平均温度:            {s.get('avg_hot_tank_temperature_C', s.get('tes_average_hot_tank_temperature_C', 0)):.1f} °C")
            self.log(f"  储能往返效率:            {s.get('tes_round_trip_efficiency_percent', s.get('tes_storage_efficiency_percent', 0)):.1f}%")
            self.log(f"  年总充热量:              {s.get('annual_total_charge_MWh', s.get('tes_total_charge_MWh', 0)):.0f} MWh")
            self.log(f"  平均放电功率:            {s.get('average_discharge_power_MW', 0):.1f} MW")
            self.log(f"  日总发电量(典型运行日):  {s.get('daily_generation_typical_MWh', 0):.0f} MWh")
            self.log(f"  年总发电量:              {annual_gen:.0f} MWh")
            self.log(f"  系统综合热效率(电/热输入): {s.get('system_overall_efficiency_percent', s.get('yearly_average_combined_efficiency_percent', 0)):.1f}%")

            self.log("=" * 60)
            self.log("  经济指标 (CNY)")
            self.log("=" * 60)
            self.log(f"  系统总投资:              {total_capital:,.0f} CNY")
            self.log(f"  年化资本成本:            {annualized_capital:,.0f} CNY/yr")
            self.log(f"  年收益(售电+售热):       {annual_revenue:,.0f} CNY/yr")
            self.log(f"  年现金流(收入-成本):     {annual_cashflow:,.0f} CNY/yr")
            self.log(f"  净现值 NPV:              {npv_val:,.0f} CNY")
            self.log(f"  内部收益率 IRR:           {irr_val:.2f}%")
            self.log(f"  平准化度电成本 LCOE:     {lcoe_val:.2f} CNY/MWh ({lcoe_val/1000:.4f} CNY/kWh)")
            self.log(f"  性能指数 I:              {I_pct:.2f}%")

            self.log("[Step 5/5] Saving...")
            self.update_progress(95, "Saving...")
            out_dir = os.path.join(PROJECT_DIR, 'results')
            os.makedirs(out_dir, exist_ok=True)
            solver.save_results_csv(os.path.join(out_dir, 'simulation_results.csv'))
            with open(os.path.join(out_dir, 'summary.txt'), 'w', encoding='utf-8') as f:
                f.write("=== NR-HES 技术经济指标 (CNY) ===\n")
                f.write(f"熔盐堆净发电功率:         {s.get('net_electric_power_MW', smr.P_net_MW):.1f} MW\n")
                f.write(f"熔盐堆净发电效率:         {s.get('net_cycle_efficiency_percent', smr.eta_design*100):.1f}%\n")
                f.write(f"额定蒸汽流量:             {s.get('rated_steam_flow_kgps', 0):.1f} kg/s\n")
                f.write(f"热罐平均温度:             {s.get('avg_hot_tank_temperature_C', s.get('tes_average_hot_tank_temperature_C', 0)):.1f} C\n")
                f.write(f"储能往返效率:             {s.get('tes_round_trip_efficiency_percent', s.get('tes_storage_efficiency_percent', 0)):.1f}%\n")
                f.write(f"年总充热量:               {s.get('annual_total_charge_MWh', s.get('tes_total_charge_MWh', 0)):.0f} MWh\n")
                f.write(f"平均放电功率:             {s.get('average_discharge_power_MW', 0):.1f} MW\n")
                f.write(f"日总发电量(典型运行日):   {s.get('daily_generation_typical_MWh', 0):.0f} MWh\n")
                f.write(f"年总发电量:               {annual_gen:.0f} MWh\n")
                f.write(f"系统综合热效率:           {s.get('system_overall_efficiency_percent', s.get('yearly_average_combined_efficiency_percent', 0)):.1f}%\n")
                f.write(f"\n--- 经济指标 ---\n")
                f.write(f"系统总投资:               {total_capital:,.0f} CNY\n")
                f.write(f"年收益(售电+售热):        {annual_revenue:,.0f} CNY/yr\n")
                f.write(f"净现值 NPV:               {npv_val:,.0f} CNY\n")
                f.write(f"内部收益率 IRR:            {irr_val:.2f}%\n")
                f.write(f"平准化度电成本 LCOE:      {lcoe_val:.2f} CNY/MWh ({lcoe_val/1000:.4f} CNY/kWh)\n")
                f.write(f"性能指数 I:               {I_pct:.2f}%\n")

            self.results = {'summary': summary, 'solver': solver, 'dni': dni, 'P_demand': P_demand,
                           'H_demand': H_demand, 'T_amb': T_amb}

            self.update_progress(100, "Done!")
            self.status_indicator.config(text="  Complete", fg=COLOR_ACCENT)
            self.log("\n*** Simulation complete! Click [Generate Charts] for figures. ***")
            messagebox.showinfo("Complete", f"Simulation finished!\n\nCycle eff: {summary['yearly_average_combined_efficiency_percent']:.1f}%\nTES Profit: {tes_profit:,.0f} CNY/yr")

        except Exception as e:
            self.log(f"ERROR: {e}")
            import traceback; self.log(traceback.format_exc())
            messagebox.showerror("Error", str(e))
        finally:
            self.sim_running = False
            self.start_btn.config(state='normal', text="  Start Simulation  ")
            self.plot_btn.config(state='normal' if self.results else 'disabled', bg=COLOR_ACCENT if self.results else '#ECEFF1')
            self.open_btn.config(state='normal')

    def _load_data(self, key, default_gen):
        path = self.data_files.get(key)
        if path:
            try:
                return np.loadtxt(path, delimiter=',', skiprows=1)[:, 1] if os.path.exists(path) else default_gen()
            except:
                self.log(f"Warning: failed to load {key} from {path}, using default.")
        return default_gen()

    def generate_plots(self):
        if not self.results: return messagebox.showwarning("No Data", "Run simulation first.")
        self.log("Generating charts...")
        try:
            viz = NRHESVisualizer(os.path.join(PROJECT_DIR, 'results', 'figures'))
            viz.generate_all(self.results['solver'].results, self.results['summary'],
                             self.results['dni'], self.results['T_amb'],
                             self.results['P_demand'], self.results['H_demand'])
            self.log("Charts saved to results/figures/")
            messagebox.showinfo("Done", "5 charts generated in results/figures/")
        except Exception as e:
            self.log(f"Chart error: {e}")

    def open_results_dir(self):
        import subprocess
        d = os.path.join(PROJECT_DIR, 'results'); os.makedirs(d, exist_ok=True)
        subprocess.Popen(f'explorer "{d}"')

    def update_progress(self, val, msg):
        self.progress_var.set(val)
        self.status_label.config(text=msg)
        self.root.update_idletasks()


def main():
    root = tk.Tk()
    SimulationGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()