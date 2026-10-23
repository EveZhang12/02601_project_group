"""
Name: Chengjun Lu, Victoria Liu, Yudi Wu, Eve Zhang
Date: 10/22/2026
"""

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from .diffusion import stokes_einstein_mm2_s

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

@dataclass
class VesselParameters:
    """
    Input settings for one vessel simulation
    """
    radius_mm: float = 0.1              # vessel radius 
    length_mm: float = 10.0             # length of the vessel piece we simulate
    mean_velocity_mm_s: float = 2.0     # average blood speed
    plasma_viscosity_pa_s: float = 1.2e-3   # thickness of plasma at body temperature
    temperature_k: float = 310.15           # 37 degrees C in kelvin
    drug_name: str = "Acetaminophen (paracetamol)"
    molecular_weight_g_mol: float = 151.16
    molecular_radius_nm: float = None  
    inlet_kind: str = "constant"        # can be "constant" or "bolus"
    inlet_concentration_mg_L: float = 10.0
    bolus_duration_s: float = 1.0
    nx: int = 200          # cells along the vessel
    ny: int = 24           # cells across the vessel
    cfl: float = 0.4       # safety factor for the time step (must be < 1)
    duration_s: float = 10.0
    n_frames: int = 101    # how many snapshots solve() stores

    @property
    def diffusion_coefficient_mm2_s(self):
        """
        Determine how fast the drug spreads out in plasma. 

        D = k_B * T / (6 * pi * mu * r)

        Returns:
            float: The diffusion coefficient D in mm^2/s
        """
        D = stokes_einstein_mm2_s(self.molecular_weight_g_mol, self.plasma_viscosity_pa_s,
                                  self.temperature_k, self.molecular_radius_nm)
        return D

    def peclet_number(self):
        """
        Compare how fast blood carries the drug with how fast it spreads sideways.

        Pe = u_mean * R / D

        Returns: 
            float: The Peclet number.
        """
        D = self.diffusion_coefficient_mm2_s
        Pe = self.mean_velocity_mm_s * self.radius_mm / D
        return Pe

    def to_dict(self):
        """
        Turn into a dictionary.
        """
        return asdict(self)

    def summary(self):
        """
        Returns one-line description of the main settings.
        """
        D = self.diffusion_coefficient_mm2_s
        Pe = self.peclet_number()
        text = (self.drug_name + ": R=" + str(self.radius_mm) + " mm, L=" + str(self.length_mm)
                + " mm, u_mean=" + str(self.mean_velocity_mm_s) + " mm/s, D=" + format(D, ".2e")
                + " mm^2/s, Pe=" + format(Pe, ".0f") + ", inlet=" + self.inlet_kind
                + " " + str(self.inlet_concentration_mg_L) + " mg/L")
        return text

def load_parameters(path=DATA_DIR / "vessel_params.json", **overrides):
    """
    Read the settings from the JSON file and check that they make sense.
    
    Args:
        path: location of the JSON file.
    
    Returns: 
        VesselParameters: the settings
    """
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    rename = {"name": "drug_name", "kind": "inlet_kind",
              "concentration_mg_L": "inlet_concentration_mg_L"}
    values = {}
    for key, group in d.items():
        if key.startswith("_"):  # skip "_comment"
            continue
        for k, v in group.items():
            values[rename.get(k, k)] = v
    values.update(overrides)
    p = VesselParameters(**values)
    if min(p.radius_mm, p.length_mm, p.plasma_viscosity_pa_s,
           p.temperature_k, p.molecular_weight_g_mol) <= 0:
        raise ValueError("radius, length, viscosity, temperature and molecular weight must be positive")
    if p.mean_velocity_mm_s < 0 or p.inlet_concentration_mg_L < 0:
        raise ValueError("velocity and inlet concentration must be non-negative")
    return p