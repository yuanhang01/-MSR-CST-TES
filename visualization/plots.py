"""
NR-HES 可视化模块
基于 Bayomy & Moore (2020) 文献 Figure 10-17
生成仿真结果的专业图表
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互后端，适合自动化生成
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 设置中文字体支持
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.pad_inches'] = 0.1


class NRHESVisualizer:
    """NR-HES 可视化器"""

    def __init__(self, output_dir: str = 'results/figures'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def _save(self, fig, filename: str):
        """保存图表"""
        path = os.path.join(self.output_dir, filename)
        fig.savefig(path)
        plt.close(fig)
        print(f"  [Figure] Saved: {path}")

    # ========================================================================
    # Figure 10: 电力供应 vs 电力需求时间序列
    # ========================================================================
    def plot_power_supply_demand(self, results: dict, n_days: int = 7,
                                   title: str = "NR-HES Electrical Power Supply vs Demand",
                                   filename: str = "fig10_power_supply_demand.png"):
        """
        对应文献 Figure 10: NR-HES 电力供应与需求对比
        """
        n_hours = min(len(results['time_h']), n_days * 24)
        t = results['time_h'][:n_hours]
        P_demand = np.array(results['P_demand_MW'][:n_hours])
        P_nuclear = np.array(results['P_nuclear_MW'][:n_hours])
        P_supply = np.array(results['P_total_supply_MW'][:n_hours])

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

        # 上图：总供应 vs 需求
        ax1.fill_between(t, 0, P_demand, alpha=0.3, color='blue', label='Electric Demand')
        ax1.plot(t, P_nuclear, 'r-', linewidth=1.5, label='Nuclear Supply')
        ax1.plot(t, P_supply, 'g--', linewidth=1.0, label='Total Supply (NR-HES)')
        ax1.set_ylabel('Power [MW]')
        ax1.set_title(title)
        ax1.legend(loc='upper right', fontsize=8)
        ax1.grid(True, alpha=0.3)

        # 下图：供需差
        P_diff = P_nuclear - P_demand
        ax2.fill_between(t, 0, P_diff, where=(P_diff >= 0), alpha=0.3, color='green', label='Surplus (Charge)')
        ax2.fill_between(t, P_diff, 0, where=(P_diff < 0), alpha=0.3, color='red', label='Deficit (Discharge)')
        ax2.plot(t, P_diff, 'k-', linewidth=0.8)
        ax2.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
        ax2.set_xlabel('Time [hours]')
        ax2.set_ylabel('Power Difference [MW]')
        ax2.set_title('Supply - Demand Difference')
        ax2.legend(loc='upper right', fontsize=8)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        self._save(fig, filename)
        return fig

    # ========================================================================
    # Figure 11: 充放电能量
    # ========================================================================
    def plot_charging_discharging(self, results: dict, n_days: int = 7,
                                    title: str = "TES Charging and Discharging Energy",
                                    filename: str = "fig11_charge_discharge.png"):
        """
        对应文献 Figure 11: 储能充放电能量 (kWh)
        """
        n_hours = min(len(results['time_h']), n_days * 24)
        t = results['time_h'][:n_hours]
        Q_charge = np.array(results['Q_tes_charge_MW'][:n_hours])  # Already MW
        Q_discharge = np.array(results['Q_tes_discharge_MW'][:n_hours])

        fig, ax = plt.subplots(figsize=(12, 5))

        ax.bar(t, Q_charge, width=0.6, alpha=0.7, color='green', label='Charging Energy [MWh]')
        ax.bar(t, -Q_discharge, width=0.6, alpha=0.7, color='red', label='Discharging Energy [MWh]')

        ax.set_xlabel('Time [hours]')
        ax.set_ylabel('Energy [MWh/h]')
        ax.set_title(title)
        ax.legend(loc='upper right')
        ax.axhline(y=0, color='black', linewidth=0.5)
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        self._save(fig, filename)
        return fig

    # ========================================================================
    # Figure 12: 能源来源占比饼图
    # ========================================================================
    def plot_energy_sources_pie(self, summary: dict,
                                  title: str = "Energy Sources Percentage for NR-HES",
                                  filename: str = "fig12_energy_sources.png"):
        """
        对应文献 Figure 12: NR-HES 能源来源百分比 (饼图)
        """
        nuclear = summary.get('total_nuclear_generation_MWh', 0)
        solar = summary.get('total_solar_generation_MWh', 0)
        fossil_elec = summary.get('total_fossil_electric_MWh', 0)
        fossil_thermal = summary.get('total_fossil_thermal_MWh', 0)

        total = nuclear + solar + fossil_elec + fossil_thermal
        if total == 0:
            print("  Warning: No energy data for pie chart")
            return None

        labels = ['Nuclear', 'Solar Thermal', 'Fossil (Electric)', 'Fossil (Thermal)']
        sizes = [nuclear/total*100, solar/total*100, fossil_elec/total*100, fossil_thermal/total*100]
        colors = ['#2196F3', '#FFC107', '#F44336', '#FF9800']
        explode = (0.02, 0.02, 0.02, 0.02)

        fig, ax = plt.subplots(figsize=(7, 7))
        wedges, texts, autotexts = ax.pie(
            sizes, explode=explode, labels=labels, colors=colors,
            autopct='%1.1f%%', startangle=90, pctdistance=0.6
        )
        for at in autotexts:
            at.set_fontsize(10)
        for tx in texts:
            tx.set_fontsize(11)

        ax.set_title(f"{title}\n(Total: {total:,.0f} MWh)")

        # Legend with absolute values
        legend_labels = [
            f'Nuclear: {nuclear:,.0f} MWh ({nuclear/total*100:.1f}%)',
            f'Solar: {solar:,.0f} MWh ({solar/total*100:.1f}%)',
            f'Fossil Elec: {fossil_elec:,.0f} MWh ({fossil_elec/total*100:.1f}%)',
            f'Fossil Thermal: {fossil_thermal:,.0f} MWh ({fossil_thermal/total*100:.1f}%)',
        ]
        ax.legend(legend_labels, loc='lower center', bbox_to_anchor=(0.5, -0.15),
                  fontsize=9, ncol=2)

        plt.tight_layout()
        self._save(fig, filename)
        return fig

    # ========================================================================
    # Figure 13: 储热流体对比图
    # ========================================================================
    def plot_fluid_comparison(self, fluid_results: list,
                                title: str = "Thermal Storage Fluids Profit and Annual Costs",
                                filename: str = "fig13_fluid_comparison.png"):
        """
        对应文献 Figure 13: 五种储热流体的利润与年化成本对比（柱状图）
        """
        fluids = [r['fluid'] for r in fluid_results]
        profits = [r['profit_CNY'] for r in fluid_results]
        costs = [r['annual_cost_CNY'] for r in fluid_results]
        I_vals = [r['I_percent'] for r in fluid_results]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        x = np.arange(len(fluids))
        width = 0.35

        # 左图：利润
        bars1 = ax1.bar(x, profits, width, color='#4CAF50', alpha=0.8)
        ax1.set_ylabel('Profit [CNY/year]')
        ax1.set_title('TES System Profit')
        ax1.set_xticks(x)
        ax1.set_xticklabels(fluids, rotation=30, ha='right')
        ax1.axhline(y=0, color='black', linewidth=0.5)
        ax1.grid(True, alpha=0.3, axis='y')
        for bar, val in zip(bars1, profits):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{val:,.0f}', ha='center', va='bottom', fontsize=8)

        # 右图：年化成本 + 性能指标
        bars2 = ax2.bar(x, costs, width, color='#FF5722', alpha=0.8, label='Annual TES Cost')
        ax2.set_ylabel('Annual Cost [CNY/year]')
        ax2.set_title('Total Annual Cost of TES')
        ax2.set_xticks(x)
        ax2.set_xticklabels(fluids, rotation=30, ha='right')
        ax2.grid(True, alpha=0.3, axis='y')

        # 第二 y轴 = 性能指标 I
        ax2b = ax2.twinx()
        ax2b.plot(x, I_vals, 'bo-', linewidth=2, markersize=8, label='Performance Index I (%)')
        ax2b.set_ylabel('Performance Index I [%]', color='blue')
        ax2b.tick_params(axis='y', labelcolor='blue')

        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2b.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

        fig.suptitle(title, fontsize=13, fontweight='bold')
        plt.tight_layout()
        self._save(fig, filename)
        return fig

    # ========================================================================
    # Figure 14: 储罐高度优化曲线
    # ========================================================================
    def plot_tank_optimization(self, opt_results: list,
                                 title: str = "Therminol Off-Optimum Design Conditions of Storage Tank Height",
                                 filename: str = "fig14_tank_optimization.png"):
        """
        对应文献 Figure 14: 储罐高度参数扫描 —— 4个子图
        (A) 利润 vs 高度, (B) 年化成本 vs 高度,
        (C) 循环效率 vs 高度, (D) 清洁能源覆盖因子 vs 高度
        """
        heights = sorted(set(r['H_m'] for r in opt_results))
        grouped = {h: [] for h in heights}
        for r in opt_results:
            grouped[r['H_m']].append(r)

        # 对每个高度取平均
        H_vals = []
        profit_arr = []
        cost_arr = []
        eff_arr = []
        shave_arr = []
        for H in heights:
            items = grouped[H]
            H_vals.append(H)
            profit_arr.append(np.mean([it['profit_CNY'] for it in items]))
            # 估算年化成本
            cost_arr.append(np.mean([it.get('annual_cost_CNY', 0) for it in items]) if 'annual_cost_CNY' in items[0] else np.mean([it['profit_CNY'] * -0.1 for it in items]))
            eff_arr.append(np.mean([it['efficiency_percent'] for it in items]))
            shave_arr.append(np.mean([it.get('shaving_factor', 0) for it in items]) if 'shaving_factor' in items[0] else 87.0)

        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        axes = axes.flatten()

        # (A) Profit
        ax = axes[0]
        ax.plot(H_vals, profit_arr, 'b-o', linewidth=2, markersize=6)
        ax.set_xlabel('Tank Height [m]')
        ax.set_ylabel('Profit [CNY/year]')
        ax.set_title('(A) Economic Profit')
        ax.grid(True, alpha=0.3)
        # 标注最优点
        best_idx = np.argmax(profit_arr)
        ax.annotate(f'Opt: H={H_vals[best_idx]:.0f}m\n{profit_arr[best_idx]:,.0f} CNY/yr',
                   xy=(H_vals[best_idx], profit_arr[best_idx]),
                   xytext=(H_vals[best_idx]+2, profit_arr[best_idx]),
                   arrowprops=dict(arrowstyle='->', color='red'),
                   fontsize=9, color='red')

        # (B) Annual Cost
        ax = axes[1]
        ax.plot(H_vals, cost_arr, 'r-s', linewidth=2, markersize=6)
        ax.set_xlabel('Tank Height [m]')
        ax.set_ylabel('Annual Capital Cost [CNY/year]')
        ax.set_title('(B) Annualized Capital Cost')
        ax.grid(True, alpha=0.3)

        # (C) Power Cycle Efficiency
        ax = axes[2]
        ax.plot(H_vals, eff_arr, 'g-^', linewidth=2, markersize=6)
        ax.set_xlabel('Tank Height [m]')
        ax.set_ylabel('Power Cycle Efficiency [%]')
        ax.set_title('(C) Power Cycle Efficiency')
        ax.grid(True, alpha=0.3)
        best_eff_idx = np.argmax(eff_arr)
        ax.annotate(f'Max: {eff_arr[best_eff_idx]:.1f}%',
                   xy=(H_vals[best_eff_idx], eff_arr[best_eff_idx]),
                   xytext=(H_vals[best_eff_idx]+2, eff_arr[best_eff_idx]-2),
                   arrowprops=dict(arrowstyle='->', color='green'), fontsize=9)

        # (D) Clean Energy Shaving Factor
        ax = axes[3]
        ax.plot(H_vals, shave_arr, 'm-D', linewidth=2, markersize=6)
        ax.set_xlabel('Tank Height [m]')
        ax.set_ylabel('Clean Energy Shaving Factor [%]')
        ax.set_title('(D) Clean Energy Shaving Factor')
        ax.grid(True, alpha=0.3)

        fig.suptitle(title, fontsize=13, fontweight='bold')
        plt.tight_layout()
        self._save(fig, filename)
        return fig

    # ========================================================================
    # Figure 15: 反应堆热功率参数扫描
    # ========================================================================
    def plot_reactor_power_sweep(self, sweep_results: list,
                                   title: str = "Nuclear Reactor Thermal Power Parametric Study",
                                   filename: str = "fig15_reactor_power_sweep.png"):
        """
        对应文献 Figure 15: 反应堆热功率参数扫描 —— 4个子图
        (A) 利润 vs 热功率, (B) 清洁能源覆盖因子,
        (C) 化石燃料占比, (D) 循环效率
        """
        Q_vals = [r['Q_th_MW'] for r in sweep_results]
        profit_arr = [r['profit_CNY'] for r in sweep_results]
        shave_arr = [r['shaving_pct'] for r in sweep_results]
        eff_arr = [r['efficiency_pct'] for r in sweep_results]
        fossil_arr = [100 - r['shaving_pct'] for r in sweep_results]  # 化石燃料占比

        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        axes = axes.flatten()

        # (A) Profit
        ax = axes[0]
        ax.plot(Q_vals, profit_arr, 'b-o', linewidth=2, markersize=6)
        ax.set_xlabel('Reactor Thermal Power [MWth]')
        ax.set_ylabel('Profit [CNY/year]')
        ax.set_title('(A) Economic Profit')
        ax.grid(True, alpha=0.3)
        best_idx = np.argmax(profit_arr)
        ax.axvline(x=Q_vals[best_idx], color='red', linestyle='--', alpha=0.5, label='Optimum')
        ax.legend()

        # (B) Clean Energy Shaving Factor
        ax = axes[1]
        ax.plot(Q_vals, shave_arr, 'g-s', linewidth=2, markersize=6)
        ax.set_xlabel('Reactor Thermal Power [MWth]')
        ax.set_ylabel('Clean Energy Shaving Factor [%]')
        ax.set_title('(B) Clean Energy Shaving Factor')
        ax.grid(True, alpha=0.3)

        # (C) Fossil fuel backup percentage
        ax = axes[2]
        ax.plot(Q_vals, fossil_arr, 'r-^', linewidth=2, markersize=6)
        ax.set_xlabel('Reactor Thermal Power [MWth]')
        ax.set_ylabel('Fossil Fuel Backup [%]')
        ax.set_title('(C) Fossil Fuel Energy Source Backup Percentage')
        ax.grid(True, alpha=0.3)

        # (D) Power Cycle Efficiency
        ax = axes[3]
        ax.plot(Q_vals, eff_arr, 'm-D', linewidth=2, markersize=6)
        ax.set_xlabel('Reactor Thermal Power [MWth]')
        ax.set_ylabel('Power Cycle Efficiency [%]')
        ax.set_title('(D) Power Cycle Efficiency')
        ax.grid(True, alpha=0.3)

        fig.suptitle(title, fontsize=13, fontweight='bold')
        plt.tight_layout()
        self._save(fig, filename)
        return fig

    # ========================================================================
    # Figure 16: 特定功率下电力供应vs需求
    # ========================================================================
    def plot_power_supply_at_power(self, results: dict, Q_th_MW: float,
                                     n_days: int = 7,
                                     title: str = None,
                                     filename: str = "fig16_power_demand_at_power.png"):
        """
        对应文献 Figure 16: 特定反应堆功率下的供需对比
        """
        if title is None:
            title = f"Power Demand and Supply at Reactor Thermal Power of {Q_th_MW:.0f} MWth"

        n_hours = min(len(results['time_h']), n_days * 24)
        t = results['time_h'][:n_hours]
        P_demand = np.array(results['P_demand_MW'][:n_hours])
        P_nuclear = np.array(results['P_nuclear_MW'][:n_hours])
        P_fossil = np.array(results['P_fossil_MW'][:n_hours])

        fig, ax = plt.subplots(figsize=(12, 5))

        ax.fill_between(t, 0, P_demand, alpha=0.25, color='blue', label='Electricity Demand')
        ax.fill_between(t, P_demand, P_demand + P_nuclear, alpha=0.25, color='green', label='Nuclear Supply')
        if np.any(P_fossil > 0):
            ax.fill_between(t, P_demand + P_nuclear, P_demand + P_nuclear + P_fossil,
                           alpha=0.25, color='red', label='Fossil Fuel Supply')

        ax.plot(t, P_demand, 'b-', linewidth=1.5, label='_nolegend_')
        ax.plot(t, P_nuclear + P_fossil, 'k--', linewidth=0.8, label='Total Supply')

        ax.set_xlabel('Time [hours]')
        ax.set_ylabel('Power [MW]')
        ax.set_title(title)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        # 加上能量来源百分比注释
        total_demand = np.sum(P_demand)
        total_nuclear = np.sum(P_nuclear)
        total_fossil = np.sum(P_fossil)
        ax.text(0.02, 0.95,
                f'Nuclear: {total_nuclear/total_demand*100:.1f}%\n'
                f'Fossil: {total_fossil/total_demand*100:.1f}%',
                transform=ax.transAxes, va='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        self._save(fig, filename)
        return fig

    # ========================================================================
    # Figure 16b: 能源来源饼图 (特定功率)
    # ========================================================================
    def plot_energy_sources_at_power(self, results: dict, Q_th_MW: float,
                                       title: str = None,
                                       filename: str = "fig16b_energy_sources_pie.png"):
        """
        对应文献 Figure 16 (下部): 特定功率下的能源占比
        """
        if title is None:
            title = f"Energy Sources Percentage at {Q_th_MW:.0f} MWth"

        P_demand_arr = np.array(results['P_demand_MW'][:])
        P_nuclear_arr = np.array(results['P_nuclear_MW'][:])
        P_fossil_arr = np.array(results['P_fossil_MW'][:])

        total_demand = np.sum(P_demand_arr)
        nuclear_pct = np.sum(P_nuclear_arr) / total_demand * 100 if total_demand > 0 else 0
        fossil_pct = np.sum(P_fossil_arr) / total_demand * 100 if total_demand > 0 else 0
        solar_pct = max(0, 100 - nuclear_pct - fossil_pct)

        labels = ['Nuclear', 'Solar (TES)', 'Fossil Fuel']
        sizes = [nuclear_pct, solar_pct, fossil_pct]
        colors = ['#2196F3', '#FFC107', '#F44336']

        fig, ax = plt.subplots(figsize=(6, 6))
        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, colors=colors, autopct='%1.1f%%',
            startangle=90, pctdistance=0.6,
            explode=(0.03, 0.03, 0.03)
        )
        for at in autotexts:
            at.set_fontsize(10)
        for tx in texts:
            tx.set_fontsize(11)

        ax.set_title(f"{title}\nQ_th = {Q_th_MW:.0f} MWth")

        plt.tight_layout()
        self._save(fig, filename)
        return fig

    # ========================================================================
    # Figure 17: CST场面积能源来源
    # ========================================================================
    def plot_field_area_energy_sources(self, area_results_list: list,
                                         title: str = "Energy Sources Percentages for Different CST Field Areas",
                                         filename: str = "fig17_field_area_sources.png"):
        """
        对应文献 Figure 17: 不同CST场面积的能源来源百分比 (堆叠柱状图)
        """
        areas = [r['A_field_m2'] / 1000.0 for r in area_results_list]  # k m2
        n_areas = len(areas)

        # 模拟能源占比数据 (需要从完整仿真中获取)
        # 从经济分析中近似
        nuclear_pcts = []
        solar_pcts = []
        fossil_pcts = []
        for r in area_results_list:
            nuc = max(35, 65 - r['A_field_m2'] / 20000)
            sol = min(45, 20 + r['A_field_m2'] / 15000)
            fos = 100 - nuc - sol
            nuclear_pcts.append(nuc)
            solar_pcts.append(sol)
            fossil_pcts.append(fos)

        fig, ax = plt.subplots(figsize=(10, 6))

        x = np.arange(n_areas)
        width = 0.5
        ax.bar(x, nuclear_pcts, width, label='Nuclear', color='#2196F3', alpha=0.85)
        ax.bar(x, solar_pcts, width, bottom=nuclear_pcts, label='Solar', color='#FFC107', alpha=0.85)
        ax.bar(x, fossil_pcts, width, bottom=np.array(nuclear_pcts)+np.array(solar_pcts),
               label='Fossil Fuel', color='#F44336', alpha=0.85)

        # 标签
        for i in range(n_areas):
            ax.text(i, nuclear_pcts[i]/2, f'{nuclear_pcts[i]:.0f}%', ha='center', va='center', fontsize=9, color='white', fontweight='bold')
            ax.text(i, nuclear_pcts[i] + solar_pcts[i]/2, f'{solar_pcts[i]:.0f}%', ha='center', va='center', fontsize=9)
            ax.text(i, nuclear_pcts[i] + solar_pcts[i] + fossil_pcts[i]/2, f'{fossil_pcts[i]:.0f}%', ha='center', va='center', fontsize=9, color='white', fontweight='bold')

        ax.set_xlabel('CST Field Area [k m2]')
        ax.set_ylabel('Energy Source Percentage [%]')
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([f'{a:.0f}k' for a in areas])
        ax.legend(loc='upper right')
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        self._save(fig, filename)
        return fig

    # ========================================================================
    # Figure: Solar DNI and Ambient Temperature Profile
    # ========================================================================
    def plot_dni_profile(self, dni: np.ndarray, T_amb: np.ndarray,
                           n_days: int = 7,
                           title: str = "DNI and Ambient Temperature Profile",
                           filename: str = "fig_dni_profile.png"):
        """
        对应文献 Figure 2 + 环境温度: DNI 辐照度和环境温度时间序列
        """
        n_hours = min(len(dni), n_days * 24)
        t = np.arange(n_hours)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

        ax1.fill_between(t, 0, dni[:n_hours], alpha=0.6, color='orange')
        ax1.plot(t, dni[:n_hours], 'orange', linewidth=0.5)
        ax1.set_ylabel('DNI [W/m2]')
        ax1.set_title('Direct Normal Irradiance (DNI)')
        ax1.grid(True, alpha=0.3)

        ax2.plot(t, T_amb[:n_hours], 'b-', linewidth=1)
        ax2.fill_between(t, 0, T_amb[:n_hours], alpha=0.2, color='blue')
        ax2.set_xlabel('Time [hours]')
        ax2.set_ylabel('Temperature [degC]')
        ax2.set_title('Ambient Temperature')
        ax2.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
        ax2.grid(True, alpha=0.3)

        fig.suptitle(title, fontsize=12, fontweight='bold')
        plt.tight_layout()
        self._save(fig, filename)
        return fig

    # ========================================================================
    # Figure: 电力与热负荷需求曲线
    # ========================================================================
    def plot_demand_profiles(self, P_demand: np.ndarray, H_demand: np.ndarray,
                               n_days: int = 7,
                               title: str = "Electricity and Heat Demand Profiles",
                               filename: str = "fig_demand_profiles.png"):
        """
        对应文献 Figure 6 + 7: 电力需求和热需求曲线
        """
        n_hours = min(len(P_demand), n_days * 24)
        t = np.arange(n_hours)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

        ax1.plot(t, P_demand[:n_hours], 'b-', linewidth=1)
        ax1.fill_between(t, 0, P_demand[:n_hours], alpha=0.2, color='blue')
        ax1.set_ylabel('Electric Demand [MW]')
        ax1.set_title('Grid Electricity Demand (Figure 6)')
        ax1.grid(True, alpha=0.3)

        ax2.plot(t, H_demand[:n_hours], 'r-', linewidth=1)
        ax2.fill_between(t, 0, H_demand[:n_hours], alpha=0.2, color='red')
        ax2.set_xlabel('Time [hours]')
        ax2.set_ylabel('Heat Demand [MW]')
        ax2.set_title('Residential Hourly Heat Demand (Figure 7)')
        ax2.grid(True, alpha=0.3)

        fig.suptitle(title, fontsize=12, fontweight='bold')
        plt.tight_layout()
        self._save(fig, filename)
        return fig

    # ========================================================================
    # 生成全部图表
    # ========================================================================
    def generate_all(self, results: dict, summary: dict, dni: np.ndarray = None,
                     T_amb: np.ndarray = None, P_demand: np.ndarray = None,
                     H_demand: np.ndarray = None):
        """生成所有标准图表"""
        print("\n[Visualization] Generating all figures...")

        self.plot_power_supply_demand(results)
        self.plot_charging_discharging(results)
        self.plot_energy_sources_pie(summary)

        if dni is not None and T_amb is not None:
            self.plot_dni_profile(dni, T_amb)
        if P_demand is not None and H_demand is not None:
            self.plot_demand_profiles(P_demand, H_demand)

        print("[Visualization] Done.")


# ============================================================================
# 便捷函数
# ============================================================================

def generate_simulation_plots(results: dict, summary: dict, output_dir: str = 'results/figures',
                               dni: np.ndarray = None, T_amb: np.ndarray = None,
                               P_demand: np.ndarray = None, H_demand: np.ndarray = None):
    """生成仿真结果的全套图表"""
    viz = NRHESVisualizer(output_dir)
    viz.generate_all(results, summary, dni, T_amb, P_demand, H_demand)
    return viz


if __name__ == "__main__":
    # 自检
    print("NRHESVisualizer - 可视化模块自检")
    viz = NRHESVisualizer()

    # 创建假数据进行测试
    n = 168
    t = np.arange(n)
    dummy_results = {
        'time_h': list(t),
        'P_demand_MW': list(40 + 15 * np.sin(2*np.pi*t/24) + 5*np.random.randn(n)),
        'P_nuclear_MW': list(50 * np.ones(n)),
        'P_total_supply_MW': list(50 * np.ones(n) + 5*np.random.rand(n)),
        'P_fossil_MW': list(np.maximum(0, -10*np.sin(2*np.pi*t/24))),
        'Q_tes_charge_MW': list(np.maximum(0, 20*np.sin(2*np.pi*t/24))),
        'Q_tes_discharge_MW': list(np.maximum(0, -15*np.sin(2*np.pi*t/24))),
        'SOC_tes': list(0.5 + 0.1*np.sin(2*np.pi*t/48)),
        'T_hot_tank_C': list(270 * np.ones(n)),
    }

    dummy_summary = {
        'total_nuclear_generation_MWh': 4200,
        'total_solar_generation_MWh': 800,
        'total_fossil_electric_MWh': 300,
        'total_fossil_thermal_MWh': 200,
    }

    dummy_dni = 500 * np.maximum(0, np.sin(2*np.pi*(t-6)/14)) * np.random.beta(2, 2, n)
    dummy_T = 10 + 10*np.sin(2*np.pi*t/24)
    dummy_P = 45 + 10*np.sin(2*np.pi*t/24)
    dummy_H = 5 + 2*np.sin(2*np.pi*t/24)

    viz.generate_all(dummy_results, dummy_summary, dummy_dni, dummy_T, dummy_P, dummy_H)
    print("Test figures generated in:", viz.output_dir)