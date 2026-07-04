"""
SMR + CST + TES Transient Simulation — GUI (MSR + SolarSalt, CNY)
Industrial dark-science theme · 深色主题 · 专业仿真界面
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

# ═══════════════════════════════════════════════════════════════
#  DARK INDUSTRIAL THEME  ·  深色工业科技风
# ═══════════════════════════════════════════════════════════════

# ── Base Colors ────────────────────────────────────────────────
C_BG         = '#121826'
C_CARD_BG    = '#1A2233'
C_INPUT_BG   = '#0F1522'
C_TEXT_PRIME = '#E6EDF7'
C_TEXT_SEC   = '#9AA7BE'
C_TEXT_DIM   = '#6B7790'
C_BORDER     = '#2A354B'
C_DIVIDER    = '#232D40'

# ── Brand / Function Colors ───────────────────────────────────
C_PRIMARY    = '#2F7BED'
C_PRIMARY_H  = '#1E69D9'
C_PRIMARY_A  = '#1858B8'
C_WHITE      = '#ffffff'
C_SUCCESS    = '#10B981'
C_WARNING    = '#F59E0B'
C_DANGER     = '#EF4444'

# ── Module Accent Colors ──────────────────────────────────────
C_SMR    = '#2F7BED'   # 核电蓝
C_CST    = '#F59E0B'   # 能源橙
C_TES    = '#14B8A6'   # 储热青
C_SIM    = '#8B5CF6'   # 控制紫

# ── Fonts ─────────────────────────────────────────────────────
FONT_TITLE    = ('Microsoft YaHei UI', 18, 'bold')
FONT_SECTION  = ('Microsoft YaHei UI', 15, 'bold')
FONT_BODY     = ('Microsoft YaHei UI', 14)
FONT_SMALL    = ('Microsoft YaHei UI', 12)
FONT_TINY     = ('Microsoft YaHei UI', 11)
FONT_MONO     = ('Cascadia Code', 11)
FONT_AUTHOR   = ('Microsoft YaHei UI', 10, 'italic')


# ── Custom ttk Styles ──────────────────────────────────────────
def setup_theme():
    style = ttk.Style()
    style.theme_use('clam')

    style.configure('.', background=C_BG, foreground=C_TEXT_PRIME)
    style.configure('TFrame', background=C_BG)
    style.configure('TLabel', background=C_BG, foreground=C_TEXT_PRIME)
    style.configure('TEntry', fieldbackground=C_INPUT_BG, foreground=C_TEXT_PRIME,
                    borderwidth=1, relief='solid')
    style.map('TEntry', bordercolor=[('focus', C_PRIMARY)], relief=[('focus', 'solid')])

    style.configure('TCombobox', fieldbackground=C_INPUT_BG, foreground=C_TEXT_PRIME,
                    borderwidth=1, relief='solid', arrowcolor=C_TEXT_PRIME)

    style.configure('TProgressbar', thickness=6, troughcolor=C_BORDER,
                    background=C_PRIMARY, borderwidth=0)
    style.configure('TNotebook', background=C_CARD_BG, borderwidth=0)
    style.configure('TNotebook.Tab', background=C_CARD_BG, foreground=C_TEXT_SEC,
                    padding=(16, 6), font=FONT_SMALL)
    style.map('TNotebook.Tab', background=[('selected', C_BG)],
              foreground=[('selected', C_PRIMARY)])
    style.configure('Vertical.TScrollbar', background=C_BG, troughcolor=C_BG,
                    arrowcolor=C_TEXT_SEC, borderwidth=0)
    style.map('Vertical.TScrollbar', background=[('active', C_BORDER)])

    return style


# ── Helper: darken/lighten color ──────────────────────────────
def _darken(hex_color, factor=0.85):
    c = hex_color.lstrip('#')
    r, g, b = int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
    r, g, b = max(0, int(r * factor)), max(0, int(g * factor)), max(0, int(b * factor))
    return f'#{r:02x}{g:02x}{b:02x}'

def _lighten(hex_color, factor=1.2):
    c = hex_color.lstrip('#')
    r, g, b = int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
    r, g, b = min(255, int(r * factor)), min(255, int(g * factor)), min(255, int(b * factor))
    return f'#{r:02x}{g:02x}{b:02x}'


# ── Button Builder ─────────────────────────────────────────────
def _btn(parent, text, command, variant='primary', font=FONT_BODY, padx=16, pady=0,
          width=None, state='normal'):
    """Create styled button: primary | secondary | text"""
    common = {
        'font': font, 'border': 0, 'relief': 'flat', 'cursor': 'hand2',
        'command': command, 'width': width,
    }
    if variant == 'primary':
        bg, fg = C_PRIMARY, C_WHITE
        h_bg, a_bg = C_PRIMARY_H, C_PRIMARY_A
        d_bg, d_fg = C_BORDER, C_TEXT_DIM
    elif variant == 'secondary':
        bg, fg = C_CARD_BG, C_TEXT_PRIME
        h_bg, a_bg = _lighten(C_CARD_BG, 1.1), C_BORDER
        d_bg, d_fg = C_CARD_BG, C_TEXT_DIM
    else:  # text
        bg, fg = C_BG, C_TEXT_SEC
        h_bg, a_bg = C_CARD_BG, C_BORDER
        d_bg, d_fg = C_BG, C_TEXT_DIM

    btn = tk.Button(parent, text=text, bg=bg, fg=fg, padx=padx, pady=pady,
                    activebackground=h_bg, activeforeground=fg, **common)
    if state == 'disabled':
        btn.config(state='disabled', bg=d_bg, fg=d_fg)
    btn._hover_bg, btn._active_bg = h_bg, a_bg
    btn._normal_bg, btn._normal_fg = bg, fg

    def on_enter(e):
        if btn['state'] != 'disabled':
            btn.config(bg=h_bg)
    def on_leave(e):
        if btn['state'] != 'disabled':
            btn.config(bg=bg)
    def on_press(e):
        if btn['state'] != 'disabled':
            btn.config(bg=a_bg)
    def on_release(e):
        if btn['state'] != 'disabled':
            btn.config(bg=h_bg)

    btn.bind('<Enter>', on_enter)
    btn.bind('<Leave>', on_leave)
    btn.bind('<ButtonPress-1>', on_press)
    btn.bind('<ButtonRelease-1>', on_release)
    return btn


# ═══════════════════════════════════════════════════════════════
#  MAIN GUI CLASS
# ═══════════════════════════════════════════════════════════════
class SimulationGUI:
    """NR-HES Simulation GUI — Industrial Dark Theme"""

    def __init__(self, root):
        self.root = root
        self.root.title("SMR + CST + TES 耦合瞬态仿真系统 — 袁航")
        self.root.geometry("1120x760")
        self.root.minsize(1000, 680)
        self.root.configure(bg=C_BG)

        self.style = setup_theme()
        self.sim_running = False
        self.sim_paused = False
        self.results = None
        self.data_files = {'DNI': None, 'P_elec': None, 'Q_heat': None, 'T_amb': None}
        self.start_time = None

        self._create_layout()
        self._log("✦ 系统就绪 — 设置参数后点击 [开始仿真]", 'info')

    # ══ Layout ═════════════════════════════════════════════════
    def _create_layout(self):
        self._create_header()
        body = tk.Frame(self.root, bg=C_BG)
        body.pack(fill='both', expand=True)

        # Left sidebar 35%
        sidebar = tk.Frame(body, bg=C_BG, width=470)
        sidebar.pack(side='left', fill='y')
        sidebar.pack_propagate(False)
        self._create_sidebar(sidebar)

        # Right content 65%
        content = tk.Frame(body, bg=C_BG)
        content.pack(side='left', fill='both', expand=True, padx=(8, 14), pady=12)
        self._create_content(content)

        self._create_footer()

    # ══ Header ═════════════════════════════════════════════════
    def _create_header(self):
        h = tk.Frame(self.root, bg=_darken(C_BG, 0.7), height=62)
        h.pack(fill='x')
        h.pack_propagate(False)

        left = tk.Frame(h, bg=h['bg'])
        left.pack(side='left', fill='y', padx=18, pady=8)
        tk.Label(left, text="⚡ SMR + CST + TES 耦合瞬态仿真系统", font=FONT_TITLE,
                 fg=C_TEXT_PRIME, bg=h['bg']).pack(anchor='w')

        right = tk.Frame(h, bg=h['bg'])
        right.pack(side='right', fill='y', padx=18, pady=6)
        tk.Label(right, text="v3.1", font=('Consolas', 11, 'bold'), fg=C_PRIMARY,
                 bg=h['bg']).pack(pady=(3,0))
        tk.Label(right, text="© 2026 袁航", font=FONT_AUTHOR, fg=C_TEXT_DIM,
                 bg=h['bg']).pack()

    # ══ Sidebar ════════════════════════════════════════════════
    def _create_sidebar(self, parent):
        canvas = tk.Canvas(parent, bg=C_BG, highlightthickness=0, width=456)
        scrollbar = ttk.Scrollbar(parent, orient='vertical', command=canvas.yview,
                                   style='Vertical.TScrollbar')
        sf = tk.Frame(canvas, bg=C_BG)

        sf.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.create_window((0, 0), window=sf, anchor='nw', width=446)
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')

        def _mw(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), 'units')
        canvas.bind_all('<MouseWheel>', _mw)

        # ── SMR Card ──
        self._param_card(sf, "🔬 SMR 熔盐堆 (MSR)", "小型模块化熔盐堆 · Brayton 循环发电", C_SMR,
            [("反应堆热功率 [MWth]", "160.0", "smr_power", "典型 100-300 MWth"),
             ("出口温度 [°C]", "700.0", "smr_outlet", "600-750°C"),
             ("涡轮入口温度 [°C]", "650.0", "smr_turbine", "550-700°C"),
             ("循环效率 [-]", "0.28", "smr_eff", "Brayton ~0.28")])

        # ── CST Card ──
        self._param_card(sf, "☀ CST 聚光太阳能塔", "塔式定日镜场 · SolarSalt 传热", C_CST,
            [("镜场面积 [m²]", "150000", "cst_area", "Gemasolar ≈306k"),
             ("聚光比 [-]", "900", "cst_ratio", "600-1,200"),
             ("场效率 [-]", "0.60", "cst_eff", "0.50-0.65")])

        # ── TES Card ──
        self._param_card(sf, "🔋 TES 储热系统", "双罐显热储热 · 可选多种介质", C_TES,
            [("储罐高度 [m]", "14.0", "tes_h", "6-20 m"),
             ("储罐直径 [m]", "10.0", "tes_d", "5-15 m")],
            with_fluid=True)

        # ── Simulation Card ──
        self._param_card(sf, "⚙ 仿真控制", "时间步长固定为 1 小时", C_SIM,
            [("仿真时长 [h]", "8760", "hours", "全年=8,760h | 一周=168h")])

        # ── Data Card ──
        self._param_card(sf, "📂 输入数据 (可选)", "留空自动使用默认数据", C_TEXT_DIM,
            [], with_data=True)

    # ── Param Card Builder ─────────────────────────────────────
    def _param_card(self, parent, title, desc, accent, fields, with_fluid=False, with_data=False):
        card = tk.Frame(parent, bg=C_CARD_BG, bd=1, relief='solid',
                        highlightbackground=C_BORDER, highlightthickness=1,
                        padx=18, pady=14)
        card.pack(fill='x', padx=6, pady=7)

        # Accent bar + title
        title_row = tk.Frame(card, bg=C_CARD_BG)
        title_row.pack(fill='x', pady=(0, 6))
        tk.Frame(title_row, bg=accent, width=3, height=20).pack(side='left', padx=(0, 8))
        tk.Label(title_row, text=title, font=FONT_SECTION, fg=C_TEXT_PRIME,
                 bg=C_CARD_BG).pack(side='left')
        tk.Label(card, text=desc, font=('Microsoft YaHei UI', 13, 'bold'), fg=C_WARNING,
                 bg=C_CARD_BG).pack(anchor='w', pady=(0, 8))

        for lbl, dfl, name, tip in fields:
            fr = tk.Frame(card, bg=C_CARD_BG)
            fr.pack(fill='x', pady=3)
            tk.Label(fr, text=lbl, bg=C_CARD_BG, font=FONT_SMALL, fg=C_TEXT_SEC,
                     width=20, anchor='e').pack(side='left', padx=(0, 8))
            var = tk.StringVar(value=dfl)
            setattr(self, name, var)
            # Extract unit after [ and remove ]
            unit = ''
            if '[' in lbl:
                unit_start = lbl.index('[')
                unit_end = lbl.index(']')
                unit = lbl[unit_start:unit_end+1]
            ent = tk.Entry(fr, textvariable=var, width=12, font=FONT_BODY,
                           bg=C_INPUT_BG, fg=C_TEXT_PRIME, bd=1, relief='solid',
                           insertbackground=C_PRIMARY, highlightthickness=0)
            ent.pack(side='left')
            tk.Label(fr, text=unit, bg=C_CARD_BG, font=FONT_SMALL,
                     fg=C_TEXT_DIM).pack(side='left', padx=(4, 0))
            if tip:
                self._tooltip(ent, tip)

        if with_fluid:
            fr = tk.Frame(card, bg=C_CARD_BG)
            fr.pack(fill='x', pady=3)
            tk.Label(fr, text="储热工质:", bg=C_CARD_BG, font=FONT_SMALL, fg=C_TEXT_SEC,
                     width=20, anchor='e').pack(side='left', padx=(0, 8))
            self.fluid_var = tk.StringVar(value="SolarSalt")
            cb = ttk.Combobox(fr, textvariable=self.fluid_var, width=13, state='readonly',
                               values=["Therminol","Dowtherm","SolarSalt","Hitec","HitecXL"],
                               font=FONT_BODY)
            cb.pack(side='left')
            self._tooltip(cb, "SolarSalt (60%NaNO₃+40%KNO₃) 220-600°C\n"
                              "Hitec 三元熔盐 142°C · HitecXL 120-500°C\n"
                              "Therminol/Dowtherm 导热油 12-400°C")

        if with_data:
            tips = {"DNI":"CSV: 时间[h], DNI[W/m²]","P_elec":"CSV: 时间[h], 电力[MW]",
                    "Q_heat":"CSV: 时间[h], 热负荷[MW]","T_amb":"CSV: 时间[h], 气温[°C]"}
            for key, label in [("DNI","DNI 辐照"),("P_elec","电力负荷"),
                               ("Q_heat","热负荷"),("T_amb","环境温度")]:
                fr = tk.Frame(card, bg=C_CARD_BG)
                fr.pack(fill='x', pady=3)
                tk.Label(fr, text=f"{label}:", bg=C_CARD_BG, font=FONT_SMALL, fg=C_TEXT_SEC,
                         width=20, anchor='e').pack(side='left', padx=(0, 8))
                b = _btn(fr, "点击选择文件", lambda k=key: self._select_file(k),
                         variant='secondary', font=FONT_TINY, padx=10)
                b.pack(side='left', padx=2)
                self._tooltip(b, tips.get(key, ""))
                lb = tk.Label(fr, text="(自动)", bg=C_CARD_BG, fg=C_TEXT_DIM, font=FONT_SMALL)
                lb.pack(side='left', padx=4)
                setattr(self, f"lbl_{key}", lb)

    # ══ Content (右区) ═════════════════════════════════════════
    def _create_content(self, parent):
        # ── Button Bar ──
        bar = tk.Frame(parent, bg=C_CARD_BG, bd=1, relief='solid',
                       highlightbackground=C_BORDER, highlightthickness=1)
        bar.pack(fill='x', pady=(0, 10), ipady=6)

        btn_row = tk.Frame(bar, bg=C_CARD_BG)
        btn_row.pack(pady=8, padx=14)

        self.start_btn = _btn(btn_row, "▶  开始仿真", self.start_simulation,
                               variant='primary', font=FONT_SECTION, padx=20)
        self.start_btn.pack(side='left', padx=4)

        self.plot_btn = _btn(btn_row, "📊 生成图表", self.generate_plots,
                              variant='secondary', font=FONT_SECTION, padx=14)
        self.plot_btn.pack(side='left', padx=4)
        self.plot_btn.config(state='disabled')

        self.open_btn = _btn(btn_row, "📁 打开结果", self.open_results_dir,
                              variant='secondary', font=FONT_SECTION, padx=14)
        self.open_btn.pack(side='left', padx=4)
        self.open_btn.config(state='disabled')

        # ── Progress ──
        pf = tk.Frame(parent, bg=C_BG)
        pf.pack(fill='x', pady=(0, 6))
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(pf, variable=self.progress_var,
                                             maximum=100, mode='determinate')
        self.progress_bar.pack(fill='x')
        pl = tk.Frame(pf, bg=C_BG)
        pl.pack(fill='x')
        self.progress_pct = tk.Label(pl, text="0%", font=FONT_SMALL, fg=C_TEXT_SEC, bg=C_BG)
        self.progress_pct.pack(side='left')
        self.progress_elapsed = tk.Label(pl, text="", font=FONT_SMALL, fg=C_TEXT_DIM, bg=C_BG)
        self.progress_elapsed.pack(side='right')
        self.progress_remain = tk.Label(pl, text="", font=FONT_SMALL, fg=C_TEXT_DIM, bg=C_BG)
        self.progress_remain.pack(side='right', padx=(0, 12))

        # ── Status Indicator ──
        sf2 = tk.Frame(parent, bg=C_BG)
        sf2.pack(fill='x', pady=(0, 6), anchor='w')
        self.status_dot = tk.Label(sf2, text="●", font=('Segoe UI', 10), fg=C_SUCCESS, bg=C_BG)
        self.status_dot.pack(side='left', padx=(2, 4))
        self.status_label = tk.Label(sf2, text="就绪", font=FONT_SMALL, fg=C_SUCCESS, bg=C_BG)
        self.status_label.pack(side='left')

        # ── Tabbed Log ──
        log_card = tk.Frame(parent, bg=C_CARD_BG, bd=1, relief='solid',
                            highlightbackground=C_BORDER, highlightthickness=1)
        log_card.pack(fill='both', expand=True)

        self.notebook = ttk.Notebook(log_card)
        self.notebook.pack(fill='both', expand=True)

        # Tab 1: 运行日志
        log_tab = tk.Frame(self.notebook, bg=C_CARD_BG)
        self.notebook.add(log_tab, text="  运行日志  ")

        log_toolbar = tk.Frame(log_tab, bg=C_CARD_BG)
        log_toolbar.pack(fill='x', padx=8, pady=(6, 2))
        tk.Label(log_toolbar, text="📋 实时日志", font=FONT_SMALL, fg=C_TEXT_PRIME,
                 bg=C_CARD_BG).pack(side='left')
        _btn(log_toolbar, "清空", self._clear_log, variant='text', font=FONT_TINY, padx=8)
        _btn(log_toolbar, "复制", self._copy_log, variant='text', font=FONT_TINY, padx=8)
        _btn(log_toolbar, "导出", self._export_log, variant='text', font=FONT_TINY, padx=8)

        self.log_text = scrolledtext.ScrolledText(log_tab, width=58, height=14,
            font=FONT_MONO, bg=C_INPUT_BG, fg=C_TEXT_SEC,
            insertbackground=C_PRIMARY, border=0, padx=10, pady=6, relief='flat')
        self.log_text.pack(fill='both', expand=True, padx=8, pady=4)
        self.log_text.tag_config('i', foreground=C_TEXT_DIM)
        self.log_text.tag_config('s', foreground=C_SUCCESS)
        self.log_text.tag_config('w', foreground=C_WARNING)
        self.log_text.tag_config('e', foreground=C_DANGER)
        self.log_text.tag_config('t', foreground=C_PRIMARY, font=FONT_SECTION)

        # Tab 2: 结果汇总
        self.summary_tab = tk.Frame(self.notebook, bg=C_CARD_BG)
        self.notebook.add(self.summary_tab, text="  结果汇总  ")
        self.summary_text = scrolledtext.ScrolledText(self.summary_tab, width=58, height=14,
            font=FONT_MONO, bg=C_INPUT_BG, fg=C_TEXT_SEC, border=0, padx=10, pady=6, relief='flat')
        self.summary_text.pack(fill='both', expand=True, padx=8, pady=4)
        self.summary_text.tag_config('k', foreground=C_PRIMARY)
        self.summary_text.tag_config('v', foreground=C_SUCCESS)

        # Tab 3: 源码展示
        self._create_source_tab()

    # ══ Footer ═════════════════════════════════════════════════
    def _create_footer(self):
        f = tk.Frame(self.root, bg=_darken(C_BG, 0.7), height=22)
        f.pack(fill='x', side='bottom')
        f.pack_propagate(False)
        self.footer_status = tk.Label(f, text="  ● 就绪", font=FONT_TINY, fg=C_SUCCESS, bg=f['bg'])
        self.footer_status.pack(side='left', padx=10)
        tk.Label(f, text="SMR+CST+TES 耦合瞬态仿真系统  |  © 2026 袁航  |  v3.1",
                 font=('Microsoft YaHei UI', 7), fg=C_TEXT_DIM, bg=f['bg']).pack(side='right', padx=10)

    # ══ Tooltip ════════════════════════════════════════════════
    @staticmethod
    def _tooltip(widget, text):
        tip = None
        def enter(e):
            nonlocal tip
            x = widget.winfo_rootx() + 10
            y = widget.winfo_rooty() + widget.winfo_height() + 4
            tip = tk.Toplevel(widget)
            tip.wm_overrideredirect(True)
            tip.wm_geometry(f"+{x}+{y}")
            tip.configure(bg=C_CARD_BG, bd=1, relief='solid', highlightbackground=C_BORDER)
            tk.Label(tip, text=text, bg=C_CARD_BG, fg=C_TEXT_PRIME, font=FONT_TINY,
                     justify='left', padx=8, pady=6, wraplength=300).pack()
            tip.lift()
        def leave(e):
            nonlocal tip
            if tip: tip.destroy(); tip = None
        widget.bind('<Enter>', enter)
        widget.bind('<Leave>', leave)

    # ══ Helpers ════════════════════════════════════════════════
    def _select_file(self, key):
        path = filedialog.askopenfilename(filetypes=[("CSV", "*.csv"), ("All", "*.*")])
        if path:
            self.data_files[key] = path
            getattr(self, f"lbl_{key}").config(text=os.path.basename(path)[:10], fg=C_SUCCESS)

    def _clear_log(self):
        self.log_text.delete('1.0', 'end')
    def _copy_log(self):
        self.root.clipboard_clear()
        self.root.clipboard_append(self.log_text.get('1.0', 'end-1c'))
    def _export_log(self):
        p = filedialog.asksaveasfilename(defaultextension='.txt',
              filetypes=[("Text", "*.txt")])
        if p:
            with open(p, 'w', encoding='utf-8') as f:
                f.write(self.log_text.get('1.0', 'end-1c'))

    def _log(self, msg, tag='i'):
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert('end', f"[{ts}] {msg}\n", tag)
        self.log_text.see('end')
        self.root.update_idletasks()

    def _update_progress(self, val, msg=''):
        self.progress_var.set(val)
        self.progress_pct.config(text=f"{val:.0f}%")
        if self.start_time and val > 0 and val < 100:
            elapsed = time.time() - self.start_time
            remain = elapsed / val * (100 - val) if val > 0 else 0
            self.progress_elapsed.config(text=f"已用 {elapsed:.0f}s")
            self.progress_remain.config(text=f"剩余 {remain:.0f}s")
        if msg:
            self.status_label.config(text=msg)
        self.root.update_idletasks()

    def _set_status(self, state):
        if state == 'ready':
            self.status_dot.config(fg=C_SUCCESS)
            self.status_label.config(fg=C_SUCCESS, text="就绪")
            self.footer_status.config(text="  ● 就绪", fg=C_SUCCESS)
        elif state == 'running':
            self.status_dot.config(fg=C_PRIMARY)
            self.status_label.config(fg=C_PRIMARY, text="运行中")
            self.footer_status.config(text="  ● 运行中", fg=C_PRIMARY)
        elif state == 'error':
            self.status_dot.config(fg=C_DANGER)
            self.status_label.config(fg=C_DANGER, text="异常")
            self.footer_status.config(text="  ● 异常", fg=C_DANGER)

    # ══ Config ═════════════════════════════════════════════════
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
            'economics': self._load_econ(),
        }

    def _load_econ(self):
        d = {'lifetime_years':20,'interest_rate':0.05,'smr_cost_per_kWe_CNY':36909.6,
             'natural_gas_cost_per_kWe_CNY':8054.8,'cst_cost_per_kWe_CNY':48880.0,
             'electricity_peak_rate_CNY_per_kWh':0.78,
             'electricity_offpeak_rate_CNY_per_kWh':0.416,
             'natural_gas_fuel_cost_CNY_per_MWh':156.0}
        try:
            cp = os.path.join(PROJECT_DIR, 'config.yaml')
            if os.path.exists(cp):
                with open(cp, encoding='utf-8') as f:
                    yc = yaml.safe_load(f)
                if yc and 'economics' in yc:
                    for k in d:
                        if k in yc['economics']: d[k] = yc['economics'][k]
        except: pass
        return d

    # ══ Simulation ═════════════════════════════════════════════
    def start_simulation(self):
        if self.sim_running:
            messagebox.showwarning("运行中", "仿真正在运行，请等待完成。")
            return
        self.sim_running = True
        self.start_btn.config(state='disabled', text="  ⟳ 仿真中 ...  ",
                              bg=C_BORDER, fg=C_TEXT_DIM)
        self.plot_btn.config(state='disabled')
        self.open_btn.config(state='disabled')
        self._set_status('running')
        self.start_time = time.time()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        try:
            config = self.get_config()
            n_hours = config['simulation']['total_hours']
            self._log("╔══════════════════════════════════════════╗", 't')
            self._log("║  SMR + CST + TES 耦合瞬态仿真启动       ║", 't')
            self._log("╚══════════════════════════════════════════╝", 't')

            self._log("[1/5] 初始化模型 ...", 'i')
            self._update_progress(5, "初始化...")

            fluid = get_fluid(config['tes']['storage_fluid'])
            config['tes']['cold_tank_temperature_C'] = fluid.T_cold_tank
            config['tes']['hot_tank_initial_temperature_C'] = fluid.T_out_solar

            smr = NuclearPowerCycle(config['smr'])
            self._log(f"  MSR: {smr.Q_th_MW:.0f}MWth P_net={smr.P_net_MW:.1f}MWe η={smr.eta_design*100:.1f}%", 'i')
            cst = ConcentratedSolarTower(config['cst'], storage_fluid=fluid)
            tes = ThermalEnergyStorage(config['tes'], storage_fluid=fluid)
            self._log(f"  TES: {tes.H_m:.1f}m×{tes.D_m:.1f}m {tes.volume_m3:.0f}m³ {fluid.name}", 'i')
            ihex = IntermediateHeatExchanger(effectiveness=0.85, heat_loss_frac=0.05, TTD_C=5.0)
            fossil = FossilFuelBackup()
            econ = EconomicAnalyzer(config['economics'])

            self._log(f"[2/5] 加载数据 ({n_hours}h) ...", 'i')
            self._update_progress(10)
            seed = int(time.time()) % 1000
            dni   = self._load('DNI',   lambda: generate_ottawa_dni(n_hours, seed=seed))
            P_dem = self._load('P_elec',lambda: generate_ontario_electric_demand(n_hours, base_load_MW=smr.P_net_MW*0.7, seed=seed+1))
            H_dem = self._load('Q_heat',lambda: generate_residential_heat_demand(n_hours, seed=seed+2))
            T_amb = self._load('T_amb', lambda: generate_ambient_temperature(n_hours, seed=seed+3))
            self._log(f"  DNI avg={np.mean(dni):.0f} W/m²  P_demand avg={np.mean(P_dem):.1f}MW", 'i')

            self._log("[3/5] 瞬态求解器运行中 ...", 'i')
            self._update_progress(15)
            solver = NRHESTransientSolver(smr=smr, cst=cst, tes=tes, ihex=ihex, fossil=fossil,
                                          economics=econ, dt_hours=1.0, verbose=False)
            gap = max(1, n_hours // 50)
            for t in range(n_hours):
                solver.step(t_h=t, DNI_Wpm2=dni[t], T_amb_C=T_amb[t],
                           P_demand_MW=P_dem[t], H_demand_MW=H_dem[t])
                if t % gap == 0:
                    self._update_progress(15+(t/n_hours)*65)

            summary = solver.get_summary()
            self._update_progress(80, "求解完成")

            # Economic analysis
            self._log("[4/5] 经济分析 ...", 'i')
            self._update_progress(85)
            fc = fluid.cost_per_m3 * tes.volume_m3 * 2.0
            tcosts = econ.tes_capital_cost(fc, tes.H_m, tes.D_m)
            tann = econ.tes_total_annual_cost(tcosts['total_capital_CNY'])
            tprofit = econ.tes_profit(tes.total_discharge_MWh, tes.round_trip_efficiency, tann)
            Ipct = econ.performance_index(tprofit, summary['yearly_average_combined_efficiency_percent']/100,
                                          summary['tes_storage_efficiency_percent']/100,
                                          econ.annualize_capital(tcosts['total_capital_CNY']))

            s = summary
            annual_gen = s.get('annual_total_electric_generation_MWh', solver.total_nuclear_generation_MWh)
            hdem = s.get('total_heat_demand_MWh', 0)
            total_cap = econ.system_total_capital(smr.P_net_MW, 0, 0, tcosts['total_capital_CNY'])
            ann_cap = econ.system_annualized_cost(total_cap)
            ann_rev = econ.annual_revenue(annual_gen, hdem)
            fcost = econ.ng_fuel_cost_per_MWh*(s.get('total_fossil_electric_MWh',0)+s.get('total_fossil_thermal_MWh',0))
            ann_cf = ann_rev - fcost - tann
            npv_val = econ.npv(total_cap, ann_cf)
            irr_val = econ.irr(total_cap, ann_cf)
            lcoe_val = econ.lcoe(ann_cap+tann, annual_gen, fcost)

            # Build summary text
            slines = []
            slines.append("══ 技术性能指标 ══")
            slines.append(f"  熔盐堆净发电功率:        {s.get('net_electric_power_MW', smr.P_net_MW):.1f} MW")
            slines.append(f"  净发电效率:              {s.get('net_cycle_efficiency_percent', smr.eta_design*100):.1f}%")
            slines.append(f"  额定蒸汽流量:            {s.get('rated_steam_flow_kgps', 0):.1f} kg/s")
            slines.append(f"  热罐平均温度:            {s.get('avg_hot_tank_temperature_C', s.get('tes_average_hot_tank_temperature_C', 0)):.1f} °C")
            slines.append(f"  储能往返效率:            {s.get('tes_round_trip_efficiency_percent', s.get('tes_storage_efficiency_percent', 0)):.1f}%")
            slines.append(f"  年总充热量:              {s.get('annual_total_charge_MWh', s.get('tes_total_charge_MWh', 0)):.0f} MWh")
            slines.append(f"  平均放电功率:            {s.get('average_discharge_power_MW', 0):.1f} MW")
            slines.append(f"  日总发电量(典型运行日):  {s.get('daily_generation_typical_MWh', 0):.0f} MWh")
            slines.append(f"  周期总发电量:            {annual_gen:.0f} MWh")
            slines.append(f"  系统综合热效率:          {s.get('system_overall_efficiency_percent', s.get('yearly_average_combined_efficiency_percent', 0)):.1f}%")
            slines.append("")
            slines.append("══ 经济指标 (CNY) ══")
            slines.append(f"  系统总投资:              {total_cap:,.0f} CNY")
            slines.append(f"  年化资本成本:            {ann_cap:,.0f} CNY/yr")
            slines.append(f"  年收益(售电+售热):       {ann_rev:,.0f} CNY/yr")
            slines.append(f"  年现金流:                {ann_cf:,.0f} CNY/yr")
            slines.append(f"  净现值 NPV:              {npv_val:,.0f} CNY")
            slines.append(f"  内部收益率 IRR:           {irr_val:.2f}%")
            slines.append(f"  LCOE:                     {lcoe_val:.2f} CNY/MWh")
            slines.append(f"  性能指数 I:              {Ipct:.2f}%")

            self._log("")
            for line in slines:
                self._log(line, 's')

            # Fill summary tab
            self.summary_text.delete('1.0', 'end')
            for line in slines:
                if line.startswith('══'):
                    self.summary_text.insert('end', line + '\n', 'k')
                else:
                    self.summary_text.insert('end', line + '\n', 'v')

            self._log("[5/5] 保存结果 ...", 'i')
            self._update_progress(95)
            out_dir = os.path.join(PROJECT_DIR, 'results')
            os.makedirs(out_dir, exist_ok=True)
            solver.save_results_csv(os.path.join(out_dir, 'simulation_results.csv'))
            with open(os.path.join(out_dir, 'summary.txt'), 'w', encoding='utf-8') as f:
                f.write("=== NR-HES 技术经济指标 (CNY) - 作者: 袁航 ===\n\n")
                for line in slines:
                    f.write(line + '\n')

            self.results = {'summary': summary, 'solver': solver, 'dni': dni,
                           'P_demand': P_dem, 'H_demand': H_dem, 'T_amb': T_amb}
            self._update_progress(100, "完成!")
            self._set_status('ready')
            self._log("\n✦ 仿真完成！点击 [生成图表] 查看可视化。", 's')
            self.notebook.select(self.summary_tab)
            messagebox.showinfo("完成",
                f"仿真成功!\n系统效率: {summary['yearly_average_combined_efficiency_percent']:.1f}%\nTES 利润: {tprofit:,.0f} CNY/yr")

        except Exception as e:
            self._log(f"✕ 错误: {e}", 'e')
            import traceback; self._log(traceback.format_exc(), 'e')
            self._set_status('error')
            messagebox.showerror("错误", str(e))
        finally:
            self.sim_running = False
            self.start_time = None
            self.start_btn.config(state='normal', text="▶  开始仿真",
                                  bg=C_PRIMARY, fg=C_WHITE)
            self.plot_btn.config(state='normal' if self.results else 'disabled',
                                 bg=C_CARD_BG if self.results else C_CARD_BG,
                                 fg=C_TEXT_PRIME if self.results else C_TEXT_DIM)
            self.open_btn.config(state='normal')

    def _load(self, key, default_gen):
        path = self.data_files.get(key)
        if path:
            try:
                return np.loadtxt(path, delimiter=',', skiprows=1)[:,1] if os.path.exists(path) else default_gen()
            except:
                self._log(f"⚠ 加载 {key} 失败, 使用默认数据", 'w')
        return default_gen()

    def generate_plots(self):
        if not self.results:
            messagebox.showwarning("无数据", "请先运行仿真。")
            return
        self._log("生成图表 ...", 'i')
        try:
            viz = NRHESVisualizer(os.path.join(PROJECT_DIR, 'results', 'figures'))
            viz.generate_all(self.results['solver'].results, self.results['summary'],
                             self.results['dni'], self.results['T_amb'],
                             self.results['P_demand'], self.results['H_demand'])
            self._log("✦ 图表已保存至 results/figures/", 's')
            messagebox.showinfo("完成", "4 张图表已生成至 results/figures/")
        except Exception as e:
            self._log(f"✕ 图表错误: {e}", 'e')

    def open_results_dir(self):
        import subprocess
        d = os.path.join(PROJECT_DIR, 'results'); os.makedirs(d, exist_ok=True)
        subprocess.Popen(f'explorer "{d}"')

    # ══ Source Code Viewer Tab ═════════════════════════════════
    def _create_source_tab(self):
        """创建源码展示标签页"""
        source_tab = tk.Frame(self.notebook, bg=C_CARD_BG)
        self.notebook.add(source_tab, text="  源码展示  ")

        # Left panel: file list (35%)
        left_panel = tk.Frame(source_tab, bg=C_CARD_BG, width=200)
        left_panel.pack(side='left', fill='y')
        left_panel.pack_propagate(False)

        # File list header
        lh = tk.Frame(left_panel, bg=C_CARD_BG)
        lh.pack(fill='x', padx=8, pady=(6, 2))
        tk.Label(lh, text="📁 项目源文件", font=FONT_SMALL, fg=C_TEXT_PRIME,
                 bg=C_CARD_BG).pack(side='left')

        # File listbox
        self.source_file_listbox = tk.Listbox(left_panel, bg=C_INPUT_BG, fg=C_TEXT_SEC,
            font=FONT_TINY, bd=0, relief='flat', selectbackground=C_PRIMARY,
            selectforeground=C_WHITE, activestyle='none', highlightthickness=0)
        self.source_file_listbox.pack(fill='both', expand=True, padx=6, pady=4)

        # Right panel: source view
        right_panel = tk.Frame(source_tab, bg=C_CARD_BG)
        right_panel.pack(side='left', fill='both', expand=True, padx=(4, 6), pady=6)

        # Source header
        rh = tk.Frame(right_panel, bg=C_CARD_BG)
        rh.pack(fill='x', pady=(0, 2))
        self.source_file_label = tk.Label(rh, text="选择文件以查看源码", font=FONT_SMALL,
            fg=C_TEXT_DIM, bg=C_CARD_BG)
        self.source_file_label.pack(side='left', padx=4)

        # Line count label
        self.source_line_count = tk.Label(rh, text="", font=FONT_TINY, fg=C_TEXT_DIM,
            bg=C_CARD_BG)
        self.source_line_count.pack(side='right', padx=6)

        # Source code text widget with line numbers
        code_bg = tk.Frame(right_panel, bg=C_INPUT_BG)
        code_bg.pack(fill='both', expand=True)

        self.source_line_nums = tk.Text(code_bg, width=4, font=FONT_MONO, bg=C_INPUT_BG,
            fg=C_TEXT_DIM, bd=0, relief='flat', padx=6, pady=6,
            state='disabled', wrap='none')
        self.source_line_nums.pack(side='left', fill='y')

        self.source_text = tk.Text(code_bg, font=FONT_MONO,
            bg=C_INPUT_BG, fg=C_TEXT_SEC, bd=0, relief='flat', padx=10, pady=6,
            insertbackground=C_PRIMARY, wrap='none')
        self.source_text.pack(side='left', fill='both', expand=True)

        # Shared scrollbar for line numbers + source text
        self.source_scrollbar = tk.Scrollbar(code_bg, orient='vertical',
            bg=C_BG, troughcolor=C_BG, activebackground=C_BORDER)
        self.source_scrollbar.pack(side='right', fill='y')
        self.source_line_nums.config(yscrollcommand=self.source_scrollbar.set)
        self.source_text.config(yscrollcommand=self.source_scrollbar.set)
        self.source_scrollbar.config(command=self._source_scroll_sync)

        # Configure syntax-highlight tags
        self.source_text.tag_config('kw', foreground='#FF7B72')     # keywords (red)
        self.source_text.tag_config('str', foreground='#A5D6FF')    # strings (light blue)
        self.source_text.tag_config('cmt', foreground='#8B949E')    # comments (grey)
        self.source_text.tag_config('num', foreground='#79C0FF')    # numbers (blue)
        self.source_text.tag_config('fn', foreground='#D2A8FF')     # function names
        self.source_text.tag_config('dec', foreground='#FFA657')    # decorators (orange)
        self.source_text.tag_config('cls', foreground='#FFA657')    # class (orange)
        self.source_text.tag_config('imp', foreground='#7EE787')    # imports (green)
        self.source_text.tag_config('const', foreground='#56D364')  # constants (green)

        # Scan and populate file list
        self._scan_source_files()
        self.source_file_listbox.bind('<<ListboxSelect>>', self._on_source_file_select)

    def _scan_source_files(self):
        """扫描项目目录中的所有 .py 文件"""
        self.source_files = {}
        self.source_file_listbox.delete(0, 'end')

        exclude_dirs = {'__pycache__', '.git', 'results', 'build_temp', 'build',
                        'build_main', 'dist', 'figures', 'logs'}
        py_files = []

        for root_dir, dirs, files in os.walk(PROJECT_DIR, topdown=True):
            dirs[:] = [d for d in dirs if d not in exclude_dirs and not d.startswith('.')]
            for f in files:
                if f.endswith('.py'):
                    full_path = os.path.join(root_dir, f)
                    rel_path = os.path.relpath(full_path, PROJECT_DIR)
                    py_files.append(rel_path)

        py_files.sort(key=lambda x: (os.path.dirname(x), os.path.basename(x)))

        for rel_path in py_files:
            self.source_files[rel_path] = os.path.join(PROJECT_DIR, rel_path)
            # Display icon based on directory
            if os.path.dirname(rel_path):
                display = f"  📄 {rel_path}"
            else:
                display = f"  📄 {rel_path}"
            self.source_file_listbox.insert('end', display)

    def _on_source_file_select(self, event):
        """当用户选择文件列表中的文件时"""
        selection = self.source_file_listbox.curselection()
        if not selection:
            return

        idx = selection[0]
        rel_path = list(self.source_files.keys())[idx]
        full_path = self.source_files[rel_path]

        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            content = f"# Error loading file: {e}"

        self.source_file_label.config(text=f"📝 {rel_path}", fg=C_SUCCESS)
        self._display_source_with_highlight(content)

    def _source_scroll_sync(self, *args):
        """同步行号和源码的滚动"""
        if args[0] == 'moveto':
            self.source_line_nums.yview_moveto(args[1])
            self.source_text.yview_moveto(args[1])
        elif args[0] == 'scroll':
            self.source_line_nums.yview_scroll(int(args[1]), args[2])
            self.source_text.yview_scroll(int(args[1]), args[2])

    def _display_source_with_highlight(self, content):
        """显示源代码并应用语法高亮"""
        self.source_text.config(state='normal')
        self.source_text.delete('1.0', 'end')
        self.source_line_nums.config(state='normal')
        self.source_line_nums.delete('1.0', 'end')

        lines = content.split('\n')
        total_lines = len(lines)
        self.source_line_count.config(text=f"共 {total_lines} 行")

        # Build line numbers
        max_digits = max(3, len(str(total_lines)))
        line_num_text = '\n'.join(f"{i+1:>{max_digits}} " for i in range(total_lines))
        self.source_line_nums.insert('1.0', line_num_text)
        self.source_line_nums.config(state='disabled')

        # Insert text and apply syntax highlighting
        self.source_text.insert('1.0', content)

        # Python keywords
        KEYWORDS = {
            'import', 'from', 'as', 'def', 'class', 'return', 'if', 'elif',
            'else', 'try', 'except', 'finally', 'raise', 'for', 'while',
            'break', 'continue', 'pass', 'and', 'or', 'not', 'in', 'is',
            'with', 'yield', 'lambda', 'assert', 'del', 'global', 'nonlocal',
            'True', 'False', 'None', 'self', 'async', 'await',
        }

        # Built-in functions
        BUILTINS = {'print', 'len', 'range', 'int', 'float', 'str', 'list',
                    'dict', 'set', 'tuple', 'bool', 'type', 'zip', 'map',
                    'filter', 'enumerate', 'super', 'isinstance', 'issubclass',
                    'hasattr', 'getattr', 'setattr', 'property', 'staticmethod',
                    'classmethod', 'abs', 'all', 'any', 'bin', 'chr', 'dir',
                    'divmod', 'eval', 'exec', 'format', 'frozenset', 'hex',
                    'id', 'input', 'iter', 'max', 'min', 'next', 'oct', 'open',
                    'ord', 'pow', 'repr', 'reversed', 'round', 'slice', 'sorted',
                    'sum', 'vars', '__import__'}

        # Highlight keywords and built-ins
        for word in KEYWORDS | BUILTINS:
            start = '1.0'
            while True:
                start = self.source_text.search(f'\\m{word}\\M', start,
                    stopindex='end', regexp=True)
                if not start:
                    break
                end = f"{start}+{len(word)}c"
                tags = self.source_text.tag_names(start)
                if 'str' not in tags and 'cmt' not in tags:
                    self.source_text.tag_add('kw', start, end)
                start = end

        # Highlight import statements (line-level)
        start = '1.0'
        while True:
            start = self.source_text.search(r'^\s*(import|from)\s', start,
                stopindex='end', regexp=True)
            if not start:
                break
            line_end = self.source_text.search(r'$', start, stopindex='end', regexp=True)
            if line_end:
                self.source_text.tag_add('imp', start, line_end)
            start = line_end if line_end else f"{start}+1c"

        # Highlight decorators
        start = '1.0'
        while True:
            start = self.source_text.search(r'^[ \t]*@\w+', start,
                stopindex='end', regexp=True)
            if not start:
                break
            line_end = self.source_text.search(r'$', start, stopindex='end', regexp=True)
            if line_end:
                self.source_text.tag_add('dec', start, line_end)
            start = line_end if line_end else f"{start}+1c"

        # Highlight class names
        start = '1.0'
        while True:
            start = self.source_text.search(r'class\s+(\w+)', start,
                stopindex='end', regexp=True)
            if not start:
                break
            end = self.source_text.search(r'[(:]', start, stopindex='end', regexp=True)
            if end:
                self.source_text.tag_add('cls', start, end)
            start = end if end else f"{start}+1c"

        # Highlight triple-quoted strings (docstrings) - manual state machine
        in_triple_single = False
        in_triple_double = False
        doc_start = None
        for i, line in enumerate(lines):
            line_idx = i + 1  # 1-based
            stripped = line.lstrip()
            if in_triple_double:
                if '"""' in line:
                    end_pos = line.index('"""')
                    self.source_text.tag_add('str', doc_start,
                        f"{line_idx}.{end_pos + 3}")
                    in_triple_double = False
                    doc_start = None
            elif in_triple_single:
                if "'''" in line:
                    end_pos = line.index("'''")
                    self.source_text.tag_add('str', doc_start,
                        f"{line_idx}.{end_pos + 3}")
                    in_triple_single = False
                    doc_start = None
            else:
                # Check for triple-quote starts
                dq_pos = line.find('"""')
                sq_pos = line.find("'''")
                first_triple = None
                first_type = None
                if dq_pos >= 0 and (sq_pos < 0 or dq_pos < sq_pos):
                    first_triple = dq_pos
                    first_type = 'double'
                elif sq_pos >= 0:
                    first_triple = sq_pos
                    first_type = 'single'

                if first_triple is not None:
                    # Check if there's another triple quote on the same line
                    remaining = line[first_triple + 3:]
                    if first_type == 'double' and '"""' in remaining:
                        end_pos = remaining.index('"""') + first_triple + 6
                        self.source_text.tag_add('str',
                            f"{line_idx}.{first_triple}",
                            f"{line_idx}.{end_pos}")
                    elif first_type == 'single' and "'''" in remaining:
                        end_pos = remaining.index("'''") + first_triple + 6
                        self.source_text.tag_add('str',
                            f"{line_idx}.{first_triple}",
                            f"{line_idx}.{end_pos}")
                    else:
                        # Multi-line docstring starts
                        if first_type == 'double':
                            in_triple_double = True
                        else:
                            in_triple_single = True
                        doc_start = f"{line_idx}.{first_triple}"

        # Regular strings (single/double quoted) - simple approach
        for pattern in [r"'[^'\\n]*'", r'"[^"\\n]*"']:
            start = '1.0'
            while True:
                start = self.source_text.search(pattern, start,
                    stopindex='end', regexp=True)
                if not start:
                    break
                match_text = self.source_text.get(start, f'{start} lineend')
                end = f"{start}+{len(match_text)}c"
                tags = self.source_text.tag_names(start)
                if 'str' not in tags and 'cmt' not in tags:
                    self.source_text.tag_add('str', start, end)
                start = end

        # Highlight comments (last, so strings take priority)
        start = '1.0'
        while True:
            start = self.source_text.search(r'#.*$', start, stopindex='end', regexp=True)
            if not start:
                break
            end = f"{start} lineend"
            if 'str' not in self.source_text.tag_names(start):
                self.source_text.tag_add('cmt', start, end)
            start = end

        # Highlight numbers
        start = '1.0'
        while True:
            start = self.source_text.search(r'\b\d+\.?\d*\b', start,
                stopindex='end', regexp=True)
            if not start:
                break
            match_text = self.source_text.get(start, f'{start} lineend')
            end = f"{start}+{len(match_text)}c"
            tags = self.source_text.tag_names(start)
            if 'str' not in tags and 'cmt' not in tags:
                self.source_text.tag_add('num', start, end)
            start = end

        self.source_text.config(state='disabled')


def main():
    root = tk.Tk()
    SimulationGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()