"""
SMR + CST + TES Transient Simulation — GUI (MSR + SolarSalt, CNY)
Double-click to run, fill in parameters, click Start, auto-generate results and charts
Author: 袁航 (Yuan Hang)
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


# ── Modern Color Palette ─────────────────────────────────────────
C_BG       = '#f0f2f5'
C_SIDEBAR  = '#1e293b'
C_PRIMARY  = '#3b82f6'
C_ACCENT   = '#10b981'
C_WARNING  = '#f59e0b'
C_DANGER   = '#ef4444'
C_TEXT     = '#1e293b'
C_SUBTEXT  = '#64748b'
C_WHITE    = '#ffffff'
C_BORDER   = '#e2e8f0'
C_INPUT_BG = '#f8fafc'
C_CARD     = '#ffffff'
C_PROGRESS_TROUGH = '#e2e8f0'

FONT_TITLE   = ('Microsoft YaHei UI', 15, 'bold')
FONT_SUBTITLE = ('Microsoft YaHei UI', 9)
FONT_SECTION = ('Microsoft YaHei UI', 10, 'bold')
FONT_BODY    = ('Microsoft YaHei UI', 9)
FONT_MONO    = ('Cascadia Code', 9)
FONT_SMALL   = ('Microsoft YaHei UI', 8)
FONT_AUTHOR  = ('Microsoft YaHei UI', 8, 'italic')


# ── Custom ttk Styles ────────────────────────────────────────────
def setup_styles():
    style = ttk.Style()
    style.theme_use('clam')

    # Frame
    style.configure('Card.TFrame', background=C_CARD, relief='solid', borderwidth=1)
    style.configure('Sidebar.TFrame', background=C_SIDEBAR)

    # Label
    style.configure('CardTitle.TLabel', font=FONT_SECTION, foreground=C_TEXT, background=C_CARD)
    style.configure('CardDesc.TLabel', font=FONT_SMALL, foreground=C_SUBTEXT, background=C_CARD)
    style.configure('Sidebar.TLabel', font=FONT_BODY, foreground='#cbd5e1', background=C_SIDEBAR)

    # Entry (simulated via white background)
    style.configure('TEntry', fieldbackground=C_INPUT_BG, borderwidth=1, relief='solid')

    # Button
    style.configure('Primary.TButton', font=FONT_SECTION, background=C_PRIMARY,
                    foreground=C_WHITE, borderwidth=0, padding=(20, 8))
    style.map('Primary.TButton', background=[('active', '#2563eb')])

    style.configure('Accent.TButton', font=FONT_SECTION, background=C_ACCENT,
                    foreground=C_WHITE, borderwidth=0, padding=(16, 7))
    style.map('Accent.TButton', background=[('active', '#059669')])

    style.configure('Outline.TButton', font=FONT_BODY, background=C_WHITE,
                    foreground=C_TEXT, borderwidth=1, relief='solid', padding=(12, 6))
    style.map('Outline.TButton', background=[('active', C_BG)])

    # Progressbar
    style.configure('TProgressbar', thickness=6, troughcolor=C_PROGRESS_TROUGH,
                    background=C_ACCENT, borderwidth=0)

    # Combobox
    style.configure('TCombobox', fieldbackground=C_INPUT_BG, borderwidth=1, relief='solid')

    # LabelFrame
    style.configure('Card.TLabelframe', background=C_CARD, borderwidth=1, relief='solid')
    style.configure('Card.TLabelframe.Label', font=FONT_SECTION, foreground=C_TEXT, background=C_CARD)

    return style


def create_rounded_button(parent, text, command, bg, fg='white', font=FONT_BODY, width=None, padx=12, pady=5):
    """Create a flat, rounded-feel button using tk canvas or simple tk Button with relief=flat."""
    btn = tk.Button(parent, text=text, font=font, bg=bg, fg=fg, padx=padx, pady=pady,
                    border=0, relief='flat', cursor='hand2',
                    activebackground=_darken(bg, 0.9), activeforeground=fg,
                    command=command, width=width)
    return btn


def _darken(hex_color, factor=0.9):
    """Simple darken helper for button hover."""
    c = hex_color.lstrip('#')
    r, g, b = int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
    r, g, b = int(r * factor), int(g * factor), int(b * factor)
    return f'#{r:02x}{g:02x}{b:02x}'


class SimulationGUI:
    """NR-HES Simulation GUI — Modern Edition"""

    def __init__(self, root):
        self.root = root
        self.root.title("SMR + CST + TES 瞬态仿真系统 — 袁航")
        self.root.geometry("1060x740")
        self.root.minsize(960, 640)
        self.root.configure(bg=C_BG)

        self.style = setup_styles()
        self.sim_running = False
        self.results = None
        self.data_files = {'DNI': None, 'P_elec': None, 'Q_heat': None, 'T_amb': None}

        self._create_layout()
        self.log("✦ 系统就绪 — 设置参数后点击 [开始仿真]")

    # ── Layout ──────────────────────────────────────────────────
    def _create_layout(self):
        # === Header Bar ===
        self._create_header()

        # === Main Body: sidebar + content ===
        body = tk.Frame(self.root, bg=C_BG)
        body.pack(fill='both', expand=True)
        body.grid_rowconfigure(0, weight=1)
        body.grid_columnconfigure(1, weight=1)

        # Sidebar (params)
        sidebar = tk.Frame(body, bg=C_SIDEBAR, width=340)
        sidebar.pack(side='left', fill='y')
        sidebar.pack_propagate(False)
        self._create_sidebar(sidebar)

        # Main content (controls + log)
        content = tk.Frame(body, bg=C_BG)
        content.pack(side='left', fill='both', expand=True, padx=16, pady=12)
        self._create_content(content)

        # === Footer ===
        self._create_footer()

    # ── Header ──────────────────────────────────────────────────
    def _create_header(self):
        header = tk.Frame(self.root, bg=C_SIDEBAR, height=56)
        header.pack(fill='x')
        header.pack_propagate(False)

        left = tk.Frame(header, bg=C_SIDEBAR)
        left.pack(side='left', fill='y', padx=20, pady=8)

        tk.Label(left, text="⚡ SMR + CST + TES 耦合瞬态仿真系统", font=FONT_TITLE,
                 fg=C_WHITE, bg=C_SIDEBAR).pack(anchor='w')
        tk.Label(left, text="MSR 熔盐堆 | Gemasolar CST | SolarSalt 双罐储热 | 袁航",
                 font=FONT_SUBTITLE, fg='#94a3b8', bg=C_SIDEBAR).pack(anchor='w')

        right = tk.Frame(header, bg=C_SIDEBAR)
        right.pack(side='right', fill='y', padx=20, pady=8)
        tk.Label(right, text="v3.0", font=('Consolas', 11, 'bold'), fg=C_ACCENT, bg=C_SIDEBAR).pack(pady=(4,0))
        tk.Label(right, text="© 2026 袁航", font=FONT_AUTHOR, fg='#64748b', bg=C_SIDEBAR).pack()

    # ── Sidebar ─────────────────────────────────────────────────
    def _create_sidebar(self, parent):
        canvas = tk.Canvas(parent, bg=C_SIDEBAR, highlightthickness=0, width=340)
        scrollbar = ttk.Scrollbar(parent, orient='vertical', command=canvas.yview)
        scroll_frame = tk.Frame(canvas, bg=C_SIDEBAR)

        scroll_frame.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.create_window((0, 0), window=scroll_frame, anchor='nw', width=326)
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')

        # Bind mousewheel scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), 'units')
        canvas.bind_all('<MouseWheel>', _on_mousewheel)

        # SMR Card
        self._card(scroll_frame, "🔬 SMR 熔盐堆 (MSR)", "小型模块化熔盐堆 · Brayton 循环发电", [
            ("反应堆热功率 [MWth]", "160.0", "smr_power", "MSR 熔盐堆热功率输出, 典型 100-300 MWth"),
            ("出口温度 [°C]", "700.0", "smr_outlet", "反应堆出口熔盐温度, 600-750°C"),
            ("涡轮入口温度 [°C]", "650.0", "smr_turbine", "Brayton 循环涡轮入口, 550-700°C"),
            ("循环效率 [-]", "0.28", "smr_eff", "热-电转换效率, Brayton ~0.28 (@700°C)"),
        ])

        # CST Card
        self._card(scroll_frame, "☀ CST 聚光太阳能塔", "塔式定日镜场 · SolarSalt 传热", [
            ("镜场面积 [m²]", "150000", "cst_area", "定日镜场总面积, Gemasolar ≈306,000 m²"),
            ("聚光比 [-]", "900", "cst_ratio", "聚光系统聚光比, 典型 600-1,200"),
            ("场效率 [-]", "0.60", "cst_eff", "含余弦/遮挡/反射损失, 典型 0.50-0.65"),
        ])

        # TES Card
        self._card(scroll_frame, "🔋 TES 储热系统", "双罐显热储热 · 可选多种介质", [
            ("储罐高度 [m]", "14.0", "tes_h", "储热罐高度, 优化范围 6-20 m"),
            ("储罐直径 [m]", "10.0", "tes_d", "储热罐直径, 优化范围 5-15 m"),
        ], with_fluid=True)

        # Simulation Card
        self._card(scroll_frame, "⚙ 仿真控制", "时间步长固定为 1 小时", [
            ("仿真时长 [h]", "8760", "hours", "全年=8,760h | 一季度=2,184h | 一周=168h"),
        ])

        # Data Card
        self._card(scroll_frame, "📂 输入数据 (可选)", "留空自动生成 Ottawa 合成数据",
                   [], with_data=True)

    # ── Card Builder ────────────────────────────────────────────
    def _card(self, parent, title, desc, fields, with_fluid=False, with_data=False):
        card = tk.Frame(parent, bg=C_CARD, bd=0, highlightthickness=0, padx=16, pady=14)
        card.pack(fill='x', padx=8, pady=5)
        # Shadow effect via a darker border frame beneath
        shadow = tk.Frame(parent, bg=C_BORDER)
        shadow.pack_forget()  # skip complex shadow for simplicity

        tk.Label(card, text=title, font=FONT_SECTION, fg=C_TEXT, bg=C_CARD).pack(anchor='w')
        tk.Label(card, text=desc, font=FONT_SMALL, fg=C_SUBTEXT, bg=C_CARD).pack(anchor='w', pady=(1, 8))

        for lbl, dfl, name, tip in fields:
            fr = tk.Frame(card, bg=C_CARD)
            fr.pack(fill='x', pady=2)
            tk.Label(fr, text=lbl, bg=C_CARD, font=FONT_BODY, fg=C_TEXT,
                     width=18, anchor='e').pack(side='left', padx=(0, 6))
            var = tk.StringVar(value=dfl)
            setattr(self, name, var)
            entry = tk.Entry(fr, textvariable=var, width=13, font=FONT_BODY,
                             bg=C_INPUT_BG, fg=C_TEXT, bd=1, relief='solid',
                             highlightthickness=0, insertbackground=C_PRIMARY)
            entry.pack(side='left')
            if tip:
                self._tooltip(entry, tip)

        if with_fluid:
            fr = tk.Frame(card, bg=C_CARD)
            fr.pack(fill='x', pady=2)
            tk.Label(fr, text="储热工质:", bg=C_CARD, font=FONT_BODY, fg=C_TEXT,
                     width=18, anchor='e').pack(side='left', padx=(0, 6))
            self.fluid_var = tk.StringVar(value="SolarSalt")
            combo = ttk.Combobox(fr, textvariable=self.fluid_var, width=11, state='readonly',
                                  values=["Therminol","Dowtherm","SolarSalt","Hitec","HitecXL"],
                                  font=FONT_BODY)
            combo.pack(side='left')
            self._tooltip(combo, "SolarSalt (60%NaNO₃+40%KNO₃) — 熔盐 220-600°C\n"
                                 "Hitec — 三元熔盐 低熔点 142°C\nHitecXL — 120-500°C\n"
                                 "Therminol / Dowtherm — 导热油 12-400°C")

        if with_data:
            tips = {
                "DNI": "CSV: 时间[h], DNI[W/m²]",
                "P_elec": "CSV: 时间[h], 电力需求[MW]",
                "Q_heat": "CSV: 时间[h], 热负荷[MW]",
                "T_amb": "CSV: 时间[h], 环境温度[°C]",
            }
            for key, (label, tip) in [("DNI", "DNI 辐照 [W/m²]"), ("P_elec", "电力负荷 [MW]"),
                                       ("Q_heat", "热负荷 [MW]"), ("T_amb", "环境温度 [°C]")]:
                fr = tk.Frame(card, bg=C_CARD)
                fr.pack(fill='x', pady=2)
                tk.Label(fr, text=f"{label}:", bg=C_CARD, font=FONT_BODY, fg=C_TEXT,
                         width=18, anchor='e').pack(side='left', padx=(0, 6))
                btn = tk.Button(fr, text="选择文件", font=FONT_SMALL, bg='#f1f5f9', fg=C_TEXT,
                                padx=6, pady=1, border=0, relief='flat', cursor='hand2',
                                command=lambda k=key: self._select_file(k))
                btn.pack(side='left', padx=2)
                self._tooltip(btn, tips.get(key, ""))
                lbl_w = tk.Label(fr, text="(自动)", bg=C_CARD, fg=C_SUBTEXT, font=FONT_SMALL)
                lbl_w.pack(side='left', padx=4)
                setattr(self, f"lbl_{key}", lbl_w)

    # ── Content (Controls + Log) ───────────────────────────────
    def _create_content(self, parent):
        # Button Bar
        bar = tk.Frame(parent, bg=C_CARD, bd=1, relief='solid', highlightthickness=0,
                       highlightbackground=C_BORDER)
        bar.pack(fill='x', pady=(0, 10), padx=2, ipady=6)

        btn_frame = tk.Frame(bar, bg=C_CARD)
        btn_frame.pack(pady=8, padx=12)

        self.start_btn = create_rounded_button(btn_frame, "▶  开始仿真", self.start_simulation,
                                                C_PRIMARY, font=FONT_SECTION, padx=18, pady=7)
        self.start_btn.pack(side='left', padx=4)

        self.plot_btn = create_rounded_button(btn_frame, "📊 生成图表", self.generate_plots,
                                               '#f1f5f9', fg=C_SUBTEXT, font=FONT_SECTION, padx=14, pady=7)
        self.plot_btn.pack(side='left', padx=4)
        self.plot_btn.config(state='disabled')

        self.open_btn = create_rounded_button(btn_frame, "📁 打开结果目录", self.open_results_dir,
                                               '#f1f5f9', fg=C_SUBTEXT, font=FONT_SECTION, padx=14, pady=7)
        self.open_btn.pack(side='left', padx=4)
        self.open_btn.config(state='disabled')

        # Progress
        prog_frame = tk.Frame(parent, bg=C_BG)
        prog_frame.pack(fill='x', pady=(0, 8))
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(prog_frame, variable=self.progress_var,
                                             maximum=100, mode='determinate',
                                             style='TProgressbar')
        self.progress_bar.pack(fill='x')
        self.status_label = tk.Label(prog_frame, text="等待中 ...", font=FONT_SMALL,
                                     bg=C_BG, fg=C_SUBTEXT)
        self.status_label.pack(anchor='w', pady=(2, 0))

        # Log Area
        log_card = tk.Frame(parent, bg=C_CARD, bd=1, relief='solid',
                            highlightbackground=C_BORDER, highlightthickness=0)
        log_card.pack(fill='both', expand=True, padx=2)

        log_header = tk.Frame(log_card, bg='#f8fafc', height=28)
        log_header.pack(fill='x')
        log_header.pack_propagate(False)
        tk.Label(log_header, text="  📋 运行日志 & 结果", font=FONT_SECTION,
                 fg=C_TEXT, bg='#f8fafc').pack(side='left', padx=10)
        tk.Label(log_header, text="🟢 就绪", font=FONT_SMALL, fg=C_ACCENT,
                 bg='#f8fafc', name='log_status').pack(side='right', padx=10, pady=3)
        self.log_status_label = log_header.nametowidget('log_status')

        self.log_text = scrolledtext.ScrolledText(log_card, width=62, height=16,
                                                    font=FONT_MONO, bg='#0f172a', fg='#e2e8f0',
                                                    insertbackground='#38bdf8',
                                                    border=0, padx=10, pady=8,
                                                    relief='flat')
        self.log_text.pack(fill='both', expand=True)
        self.log_text.tag_config('info', foreground='#94a3b8')
        self.log_text.tag_config('success', foreground='#34d399')
        self.log_text.tag_config('warn', foreground='#fbbf24')
        self.log_text.tag_config('err', foreground='#f87171')
        self.log_text.tag_config('title', foreground='#38bdf8', font=FONT_SECTION)

    # ── Footer ─────────────────────────────────────────────────
    def _create_footer(self):
        footer = tk.Frame(self.root, bg=C_SIDEBAR, height=24)
        footer.pack(fill='x', side='bottom')
        footer.pack_propagate(False)
        tk.Label(footer, text="  SMR + CST + TES 耦合瞬态仿真系统  |  © 2026 袁航  |  All rights reserved",
                 font=('Microsoft YaHei UI', 7), fg='#64748b', bg=C_SIDEBAR).pack(side='left', padx=12)

    # ── Tooltip ─────────────────────────────────────────────────
    @staticmethod
    def _tooltip(widget, text):
        tip = None
        def enter(e):
            nonlocal tip
            x = widget.winfo_rootx() + 12
            y = widget.winfo_rooty() + widget.winfo_height() + 6
            tip = tk.Toplevel(widget)
            tip.wm_overrideredirect(True)
            tip.wm_geometry(f"+{x}+{y}")
            tip.configure(bg='#1e293b')
            tk.Label(tip, text=text, bg='#1e293b', fg='#e2e8f0',
                     font=FONT_SMALL, justify='left', padx=8, pady=6,
                     wraplength=300).pack()
            tip.lift()
        def leave(e):
            nonlocal tip
            if tip:
                tip.destroy()
                tip = None
        widget.bind('<Enter>', enter)
        widget.bind('<Leave>', leave)

    # ── File Selector ──────────────────────────────────────────
    def _select_file(self, key):
        path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if path:
            self.data_files[key] = path
            getattr(self, f"lbl_{key}").config(text=os.path.basename(path)[:12], fg=C_ACCENT)

    # ── Logging ─────────────────────────────────────────────────
    def log(self, msg, tag=None):
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert('end', f"[{ts}] {msg}\n", tag or 'info')
        self.log_text.see('end')
        self.root.update_idletasks()

    def update_progress(self, val, msg):
        self.progress_var.set(val)
        self.status_label.config(text=msg)
        self.root.update_idletasks()

    # ── Config ──────────────────────────────────────────────────
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
                    yc = yaml.safe_load(f)
                if yc and 'economics' in yc:
                    em = yc['economics']
                    for k in default_econ:
                        if k in em:
                            default_econ[k] = em[k]
        except Exception:
            pass
        return default_econ

    # ── Simulation ──────────────────────────────────────────────
    def start_simulation(self):
        if self.sim_running: return messagebox.showwarning("Busy", "仿真正在运行中...")
        self.sim_running = True
        self.start_btn.config(state='disabled', text="  ⟳ 运行中 ...  ", bg=_darken(C_PRIMARY))
        self.plot_btn.config(state='disabled', bg='#f1f5f9', fg=C_SUBTEXT)
        self.open_btn.config(state='disabled', bg='#f1f5f9', fg=C_SUBTEXT)
        self.log_status_label.config(text="🟡 运行中")
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        try:
            config = self.get_config()
            n_hours = config['simulation']['total_hours']
            self.log("", 'title')
            self.log("╔══════════════════════════════════════════╗")
            self.log("║  SMR + CST + TES 耦合瞬态仿真启动       ║")
            self.log("╚══════════════════════════════════════════╝")

            self.log("[1/5] 初始化模型 ...")
            self.update_progress(5, "初始化中...")

            fluid = get_fluid(config['tes']['storage_fluid'])
            config['tes']['cold_tank_temperature_C'] = fluid.T_cold_tank
            config['tes']['hot_tank_initial_temperature_C'] = fluid.T_out_solar

            smr = NuclearPowerCycle(config['smr'])
            self.log(f"  MSR: {smr.Q_th_MW:.0f} MWth | P_net={smr.P_net_MW:.1f} MWe | η={smr.eta_design*100:.1f}%", 'info')
            cst = ConcentratedSolarTower(config['cst'], storage_fluid=fluid)
            tes = ThermalEnergyStorage(config['tes'], storage_fluid=fluid)
            self.log(f"  TES: H={tes.H_m:.1f}m D={tes.D_m:.1f}m V={tes.volume_m3:.0f}m³ | 工质={fluid.name}", 'info')
            ihex = IntermediateHeatExchanger(effectiveness=0.85, heat_loss_frac=0.05, TTD_C=5.0)
            fossil = FossilFuelBackup()
            econ = EconomicAnalyzer(config['economics'])

            self.log(f"[2/5] 加载输入数据 ({n_hours} h) ...")
            self.update_progress(10, "加载数据...")
            seed = int(time.time()) % 1000
            dni = self._load_data('DNI', lambda: generate_ottawa_dni(n_hours, seed=seed))
            P_demand = self._load_data('P_elec', lambda: generate_ontario_electric_demand(n_hours, base_load_MW=smr.P_net_MW*0.7, seed=seed+1))
            H_demand = self._load_data('Q_heat', lambda: generate_residential_heat_demand(n_hours, seed=seed+2))
            T_amb = self._load_data('T_amb', lambda: generate_ambient_temperature(n_hours, seed=seed+3))
            self.log(f"  DNI avg={np.mean(dni):.0f} W/m² | P_demand avg={np.mean(P_demand):.1f} MW", 'info')

            self.log(f"[3/5] 运行瞬态求解器 ...")
            self.update_progress(15, "仿真中...")
            solver = NRHESTransientSolver(smr=smr, cst=cst, tes=tes, ihex=ihex, fossil=fossil,
                                          economics=econ, dt_hours=1.0, verbose=False,
                                          fossil_backup_active=False)
            gap = max(1, n_hours // 50)
            for t in range(n_hours):
                solver.step(t_h=t, DNI_Wpm2=dni[t], T_amb_C=T_amb[t],
                           P_demand_MW=P_demand[t], H_demand_MW=H_demand[t])
                if t % gap == 0:
                    self.update_progress(15 + (t / n_hours) * 65, f"仿真 {t+1}/{n_hours} h")

            summary = solver.get_summary()
            self.update_progress(80, "仿真完成")

            self.log("[4/5] 经济分析 ...")
            self.update_progress(85, "经济分析...")
            fluid_cost = fluid.cost_per_m3 * tes.volume_m3 * 2.0
            tes_costs = econ.tes_capital_cost(fluid_cost, tes.H_m, tes.D_m)
            tes_annual = econ.tes_total_annual_cost(tes_costs['total_capital_CNY'])
            tes_profit = econ.tes_profit(tes.total_discharge_MWh, tes.round_trip_efficiency, tes_annual)
            I_pct = econ.performance_index(tes_profit, summary['yearly_average_combined_efficiency_percent']/100,
                                           summary['tes_storage_efficiency_percent']/100,
                                           econ.annualize_capital(tes_costs['total_capital_CNY']))

            s = summary
            annual_gen = s.get('annual_total_electric_generation_MWh', solver.total_nuclear_generation_MWh)
            total_heat_demand = s.get('total_heat_demand_MWh', 0)
            total_capital = econ.system_total_capital(smr.P_net_MW, 0, 0, tes_costs['total_capital_CNY'])
            annualized_capital = econ.system_annualized_cost(total_capital)
            annual_revenue = econ.annual_revenue(annual_gen, total_heat_demand)
            annual_fuel_cost = econ.ng_fuel_cost_per_MWh * (s.get('total_fossil_electric_MWh', 0) + s.get('total_fossil_thermal_MWh', 0))
            annual_cashflow = annual_revenue - annual_fuel_cost - tes_annual
            npv_val = econ.npv(total_capital, annual_cashflow)
            irr_val = econ.irr(total_capital, annual_cashflow)
            lcoe_val = econ.lcoe(annualized_capital + tes_annual, annual_gen, annual_fuel_cost)

            self.log("", 'title')
            self.log("── 技术性能指标 ──", 'title')
            self.log(f"  熔盐堆净发电功率:        {s.get('net_electric_power_MW', smr.P_net_MW):.1f} MW", 'success')
            self.log(f"  熔盐堆净发电效率:        {s.get('net_cycle_efficiency_percent', smr.eta_design*100):.1f}%", 'success')
            self.log(f"  额定蒸汽流量:            {s.get('rated_steam_flow_kgps', 0):.1f} kg/s", 'success')
            self.log(f"  热罐平均温度:            {s.get('avg_hot_tank_temperature_C', s.get('tes_average_hot_tank_temperature_C', 0)):.1f} °C", 'success')
            self.log(f"  储能往返效率:            {s.get('tes_round_trip_efficiency_percent', s.get('tes_storage_efficiency_percent', 0)):.1f}%", 'success')
            self.log(f"  年总充热量:              {s.get('annual_total_charge_MWh', s.get('tes_total_charge_MWh', 0)):.0f} MWh", 'info')
            self.log(f"  平均放电功率:            {s.get('average_discharge_power_MW', 0):.1f} MW", 'info')
            self.log(f"  日总发电量(典型运行日):  {s.get('daily_generation_typical_MWh', 0):.0f} MWh", 'success')
            self.log(f"  年总发电量:              {annual_gen:.0f} MWh", 'success')
            self.log(f"  系统综合热效率:          {s.get('system_overall_efficiency_percent', s.get('yearly_average_combined_efficiency_percent', 0)):.1f}%", 'success')
            self.log("")
            self.log("── 经济指标 (CNY) ──", 'title')
            self.log(f"  系统总投资:              {total_capital:,.0f} CNY", 'success')
            self.log(f"  年收益(售电+售热):       {annual_revenue:,.0f} CNY/yr", 'success')
            self.log(f"  年现金流(收入-成本):     {annual_cashflow:,.0f} CNY/yr", 'success')
            self.log(f"  净现值 NPV:              {npv_val:,.0f} CNY", 'success' if npv_val > 0 else 'warn')
            self.log(f"  内部收益率 IRR:           {irr_val:.2f}%", 'success' if irr_val > 5 else 'warn')
            self.log(f"  平准化度电成本 LCOE:     {lcoe_val:.2f} CNY/MWh", 'success')
            self.log(f"  性能指数 I:              {I_pct:.2f}%", 'info')

            self.log("[5/5] 保存结果 ...")
            self.update_progress(95, "保存中...")
            out_dir = os.path.join(PROJECT_DIR, 'results')
            os.makedirs(out_dir, exist_ok=True)
            solver.save_results_csv(os.path.join(out_dir, 'simulation_results.csv'))
            with open(os.path.join(out_dir, 'summary.txt'), 'w', encoding='utf-8') as f:
                f.write("=== NR-HES 技术经济指标 (CNY) - 作者: 袁航 ===\n\n")
                f.write(f"熔盐堆净发电功率:         {s.get('net_electric_power_MW', smr.P_net_MW):.1f} MW\n")
                f.write(f"热罐平均温度:             {s.get('avg_hot_tank_temperature_C', s.get('tes_average_hot_tank_temperature_C', 0)):.1f} C\n")
                f.write(f"日总发电量(典型运行日):   {s.get('daily_generation_typical_MWh', 0):.0f} MWh\n")
                f.write(f"年总发电量:               {annual_gen:.0f} MWh\n")
                f.write(f"\n系统总投资:               {total_capital:,.0f} CNY\n")
                f.write(f"LCOE:                      {lcoe_val:.2f} CNY/MWh\n")

            self.results = {'summary': summary, 'solver': solver, 'dni': dni, 'P_demand': P_demand,
                           'H_demand': H_demand, 'T_amb': T_amb}
            self.update_progress(100, "完成!")
            self.log_status_label.config(text="🟢 完成", fg=C_ACCENT)
            self.log("\n✦ 仿真完成！点击 [生成图表] 查看可视化结果。", 'success')
            messagebox.showinfo("完成", f"仿真成功!\n\n系统效率: {summary['yearly_average_combined_efficiency_percent']:.1f}%\nTES 利润: {tes_profit:,.0f} CNY/yr")

        except Exception as e:
            self.log(f"✕ 错误: {e}", 'err')
            import traceback; self.log(traceback.format_exc(), 'err')
            messagebox.showerror("错误", str(e))
        finally:
            self.sim_running = False
            self.start_btn.config(state='normal', text="▶  开始仿真", bg=C_PRIMARY)
            self.plot_btn.config(state='normal' if self.results else 'disabled',
                                 bg=C_ACCENT if self.results else '#f1f5f9',
                                 fg=C_WHITE if self.results else C_SUBTEXT)
            self.open_btn.config(state='normal', bg='#e2e8f0', fg=C_TEXT)

    def _load_data(self, key, default_gen):
        path = self.data_files.get(key)
        if path:
            try:
                return np.loadtxt(path, delimiter=',', skiprows=1)[:, 1] if os.path.exists(path) else default_gen()
            except:
                self.log(f"⚠ 未能加载 {key} 文件, 使用默认数据", 'warn')
        return default_gen()

    def generate_plots(self):
        if not self.results:
            messagebox.showwarning("无数据", "请先运行仿真。")
            return
        self.log("生成图表中 ...", 'info')
        try:
            viz = NRHESVisualizer(os.path.join(PROJECT_DIR, 'results', 'figures'))
            viz.generate_all(self.results['solver'].results, self.results['summary'],
                             self.results['dni'], self.results['T_amb'],
                             self.results['P_demand'], self.results['H_demand'])
            self.log("✦ 图表已保存至 results/figures/", 'success')
            messagebox.showinfo("完成", "5 张图表已生成至 results/figures/")
        except Exception as e:
            self.log(f"✕ 图表生成错误: {e}", 'err')
            messagebox.showerror("错误", str(e))

    def open_results_dir(self):
        import subprocess
        d = os.path.join(PROJECT_DIR, 'results'); os.makedirs(d, exist_ok=True)
        subprocess.Popen(f'explorer "{d}"')


def main():
    root = tk.Tk()
    SimulationGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()