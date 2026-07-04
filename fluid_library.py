"""
Thermal storage fluid library — Based on Bayomy & Moore (2020) Table 7
Supports 5 fluids: Therminol 66, Dowtherm T, Solar Salt, Hitec, Hitec XL
Prices in CNY (RMB), converted from CAD at ~5.2 CNY/CAD
"""

import numpy as np

FLUID_PROPERTIES = {
    "Therminol": {
        "density_kgpm3": 941.0,
        "specific_heat_kJpkgK": 1.91,
        "volumetric_heat_capacity_kJpm3K": 1797.3,
        "thermal_conductivity_WpmK": 0.1121,
        "dynamic_viscosity_Pa_s": 0.00242,
        "T_max_C": 315.0,
        "T_min_C": -3.0,
        "melting_temperature_C": None,
        "cost_per_liter_CNY": 20.18,
        "density_coeffs": {"rho0": 941.0, "T_ref": 47.0, "alpha": -0.61},
        "cp_coeffs": {"a": 1.70, "b": 0.0035},
        "k_coeffs": {"a": 0.1273, "b": 0.00026},
        "mu_coeffs": {"a": -3.75, "b": 892.0, "c": 0.0},
    },
    "Dowtherm": {
        "density_kgpm3": 807.0, "specific_heat_kJpkgK": 2.2,
        "volumetric_heat_capacity_kJpm3K": 1775.4, "thermal_conductivity_WpmK": 0.11,
        "dynamic_viscosity_Pa_s": 0.002365, "T_max_C": 345.0, "T_min_C": -10.0,
        "melting_temperature_C": None, "cost_per_liter_CNY": 37.02,
        "density_coeffs": {"rho0": 807.0, "T_ref": 47.0, "alpha": -0.67},
        "cp_coeffs": {"a": 2.00, "b": 0.0030},
        "k_coeffs": {"a": 0.130, "b": 0.00025},
        "mu_coeffs": {"a": -4.10, "b": 950.0, "c": 0.0},
    },
    "SolarSalt": {
        "density_kgpm3": 1899.0, "specific_heat_kJpkgK": 1.495,
        "volumetric_heat_capacity_kJpm3K": 2839.0, "thermal_conductivity_WpmK": 0.55,
        "dynamic_viscosity_Pa_s": 0.00326, "T_max_C": 585.0, "T_min_C": 220.0,
        "melting_temperature_C": 220.0, "cost_per_liter_CNY": 16.59,
        "density_coeffs": {"rho0": 1899.0, "T_ref": 400.0, "alpha": -0.64},
        "cp_coeffs": {"a": 1.443, "b": 0.000172},
        "k_coeffs": {"a": 0.443, "b": 0.00019},
        "mu_coeffs": {"a": -4.343, "b": 2905.0, "c": 0.0},
    },
    "Hitec": {
        "density_kgpm3": 1860.0, "specific_heat_kJpkgK": 1.56,
        "volumetric_heat_capacity_kJpm3K": 2901.6, "thermal_conductivity_WpmK": 0.60,
        "dynamic_viscosity_Pa_s": 0.00316, "T_max_C": 538.0, "T_min_C": 142.0,
        "melting_temperature_C": 142.0, "cost_per_liter_CNY": 24.18,
        "density_coeffs": {"rho0": 1860.0, "T_ref": 340.0, "alpha": -0.65},
        "cp_coeffs": {"a": 1.560, "b": 0.00000},
        "k_coeffs": {"a": 0.600, "b": -0.0001},
        "mu_coeffs": {"a": -4.20, "b": 2700.0, "c": 0.0},
    },
    "HitecXL": {
        "density_kgpm3": 1992.0, "specific_heat_kJpkgK": 1.447,
        "volumetric_heat_capacity_kJpm3K": 2882.4, "thermal_conductivity_WpmK": 0.52,
        "dynamic_viscosity_Pa_s": 0.00637, "T_max_C": 505.0, "T_min_C": 120.0,
        "melting_temperature_C": 120.0, "cost_per_liter_CNY": 22.27,
        "density_coeffs": {"rho0": 1992.0, "T_ref": 310.0, "alpha": -0.64},
        "cp_coeffs": {"a": 1.447, "b": 0.00000},
        "k_coeffs": {"a": 0.520, "b": 0.0000},
        "mu_coeffs": {"a": -3.80, "b": 2600.0, "c": 0.0},
    },
}

FLUID_TEMPERATURE_CONFIG = {
    "Therminol": {"T_cold_tank_C": 25.0, "T_out_solar_C": 300.0, "T_out_HPT_C": 139.2, "T_out_superheater_C": 295.0},
    "Dowtherm": {"T_cold_tank_C": 25.0, "T_out_solar_C": 300.0, "T_out_HPT_C": 139.2, "T_out_superheater_C": 295.0},
    "SolarSalt": {"T_cold_tank_C": 225.0, "T_out_solar_C": 550.0, "T_out_HPT_C": 403.2, "T_out_superheater_C": 545.0},
    "Hitec": {"T_cold_tank_C": 147.0, "T_out_solar_C": 500.0, "T_out_HPT_C": 359.8, "T_out_superheater_C": 495.0},
    "HitecXL": {"T_cold_tank_C": 125.0, "T_out_solar_C": 500.0, "T_out_HPT_C": 359.8, "T_out_superheater_C": 495.0},
}


class StorageFluid:
    """Storage fluid with temperature-dependent properties"""

    def __init__(self, fluid_name: str):
        if fluid_name not in FLUID_PROPERTIES:
            raise ValueError(f"Unknown fluid: {fluid_name}. Options: {list(FLUID_PROPERTIES.keys())}")
        self.name = fluid_name
        self.props = FLUID_PROPERTIES[fluid_name]
        self.T_config = FLUID_TEMPERATURE_CONFIG[fluid_name]

    def density(self, T_C: float) -> float:
        c = self.props["density_coeffs"]
        return c["rho0"] + c["alpha"] * (T_C - c["T_ref"])

    def specific_heat(self, T_C: float) -> float:
        c = self.props["cp_coeffs"]
        return c["a"] + c["b"] * T_C

    def thermal_conductivity(self, T_C: float) -> float:
        c = self.props["k_coeffs"]
        return c["a"] - c["b"] * T_C

    def dynamic_viscosity(self, T_C: float) -> float:
        c = self.props["mu_coeffs"]
        T_K = T_C + 273.15
        return np.exp(c["a"] + c["b"] / T_K + c["c"] * T_K**2)

    def volumetric_heat_capacity(self, T_C: float) -> float:
        return self.density(T_C) * self.specific_heat(T_C)

    @property
    def T_max(self) -> float: return self.props["T_max_C"]
    @property
    def T_min(self) -> float: return self.props["T_min_C"]
    @property
    def melting_temperature(self): return self.props["melting_temperature_C"]
    @property
    def cost_per_liter(self) -> float: return self.props["cost_per_liter_CNY"]
    @property
    def cost_per_m3(self) -> float: return self.props["cost_per_liter_CNY"] * 1000.0
    @property
    def T_cold_tank(self) -> float: return self.T_config["T_cold_tank_C"]
    @property
    def T_out_solar(self) -> float: return self.T_config["T_out_solar_C"]
    @property
    def T_out_HPT(self) -> float: return self.T_config["T_out_HPT_C"]
    @property
    def T_out_superheater(self) -> float: return self.T_config["T_out_superheater_C"]

    def needs_auxiliary_heating(self) -> bool:
        return self.T_cold_tank > 50.0

    def get_nominal_properties_table(self) -> dict:
        return {
            "density_kgpm3": self.props["density_kgpm3"],
            "specific_heat_kJpkgK": self.props["specific_heat_kJpkgK"],
            "volumetric_heat_capacity_kJpm3K": self.props["volumetric_heat_capacity_kJpm3K"],
            "thermal_conductivity_WpmK": self.props["thermal_conductivity_WpmK"],
            "dynamic_viscosity_Pa_s": self.props["dynamic_viscosity_Pa_s"],
            "T_max_C": self.props["T_max_C"],
            "T_min_C": self.props["T_min_C"],
            "melting_T_C": self.props["melting_temperature_C"],
            "cost_CNY_per_L": self.props["cost_per_liter_CNY"],
        }

    def __repr__(self):
        return f"StorageFluid({self.name}, T_range=[{self.T_min}, {self.T_max}] C)"


def get_fluid(fluid_name: str) -> StorageFluid:
    return StorageFluid(fluid_name)

def list_available_fluids() -> list:
    return list(FLUID_PROPERTIES.keys())

def print_fluid_table():
    import sys
    if sys.stdout.encoding != 'utf-8':
        try: sys.stdout.reconfigure(encoding='utf-8')
        except: pass
    fluids = [StorageFluid(name) for name in FLUID_PROPERTIES.keys()]
    header = f"{'Property':<35}" + "".join(f"{f.name:<16}" for f in fluids)
    print(header)
    print("-" * len(header))
    rows = [
        ("Density (kg/m3)", "density_kgpm3"),
        ("Specific Heat (kJ/kg.K)", "specific_heat_kJpkgK"),
        ("Vol. Heat Cap. (kJ/m3.K)", "volumetric_heat_capacity_kJpm3K"),
        ("Thermal Cond. (W/m.K)", "thermal_conductivity_WpmK"),
        ("Dyn. Viscosity (Pa.s)", "dynamic_viscosity_Pa_s"),
        ("T_max (degC)", "T_max_C"),
        ("T_min (degC)", "T_min_C"),
        ("Cost (CNY/L)", "cost_CNY_per_L"),
    ]
    for label, key in rows:
        row = f"{label:<35}"
        for f in fluids:
            val = f.get_nominal_properties_table()[key]
            if val is None: row += f"{'N/A':<16}"
            elif isinstance(val, float): row += f"{val:<16.4g}"
            else: row += f"{str(val):<16}"
        print(row)