"""
Name: Chengjun Lu, Victoria Liu, Yudi Wu, Eve Zhang
Date: 10/22/2026
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import load_parameters, VesselModel, InletProfile, BolusInlet
from src.visualization import VesselVisualizer

def main():
    """
    Run the vessel demos and save all figures and mp4s.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "docs" / "figures"))
    ap.add_argument("--no-movie", action="store_true")
    args = ap.parse_args()
    viz = VesselVisualizer(args.out)

    p = load_parameters()
    print(p.summary())
    model = VesselModel.from_parameters(p)
    print(f"grid {model.nx} x {model.ny}, dx={model.dx:.3f} mm, dy={model.dy:.4f} mm, "
          f"dt={model.dt * 1000:.2f} ms")
    viz.plot_velocity_profile(model, save="velocity_profile.png")

    const = model.solve(InletProfile.from_parameters(p), p.duration_s, p.n_frames)
    print(f"[constant] mass-balance error {const.mass_balance_error():.1e}")
    viz.plot_snapshots(const, times=(1, 2.5, 5, 10), save="washin_snapshots.png")
    viz.plot_profiles(const, times=(1, 2.5, 5, 10), save="washin_profiles.png")

    bolus_inlet = BolusInlet(p.inlet_concentration_mg_L, p.bolus_duration_s)
    bolus = model.solve(bolus_inlet, 12.0, 121)
    print(f"[bolus]    mass-balance error {bolus.mass_balance_error():.1e}")
    viz.plot_snapshots(bolus, times=(1, 3, 5, 8), save="bolus_snapshots.png")

    p_fine = load_parameters(ny=96)
    small_fine = VesselModel.from_parameters(p_fine).solve(bolus_inlet, 12.0, 241)
    p_nano = load_parameters(drug_name="50-nm nanoparticle", molecular_radius_nm=50.0, ny=96)
    nano = VesselModel.from_parameters(p_nano).solve(bolus_inlet, 12.0, 241)
    print(f"[nano]     D = {p_nano.diffusion_coefficient_mm2_s:.2e} mm^2/s, "
          f"Pe = {p_nano.peclet_number():.0f}, mass-balance error {nano.mass_balance_error():.1e}")
    viz.plot_snapshots(nano, times=(1, 3, 5, 8), save="nanoparticle_snapshots.png")
    viz.plot_breakthrough({f"small drug (D = {p.diffusion_coefficient_mm2_s:.1e} mm²/s)": small_fine,
                           f"50-nm nanoparticle (D = {p_nano.diffusion_coefficient_mm2_s:.1e} mm²/s)": nano},
                          save="breakthrough.png",
                          title="1-s pulse: concentration leaving the vessel")

    if not args.no_movie:
        print("saved", viz.animate(const, "washin.mp4", title="Constant inlet:  "))
        print("saved", viz.animate(bolus, "bolus.mp4", title="1-s bolus:  "))

if __name__ == "__main__":
    main()