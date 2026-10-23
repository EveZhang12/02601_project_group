"""
Name: Chengjun Lu, Victoria Liu, Yudi Wu, Eve Zhang
Date: 10/22/2026
"""

from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")     
import matplotlib.pyplot as plt 
from matplotlib import animation  
import numpy as np 
from .vessel import VesselModel, VesselResult 

LINE_COLORS = ["#1f5fa8", "#d9822b", "#2e7d5b", "#8e3b46"]  

class VesselVisualizer:
    """
    Makes figures and movies from vessel simulation results.
    """
    def __init__(self, out_dir: str | Path | None = None):
        self.out = Path(out_dir) if out_dir else None
        if self.out:
            self.out.mkdir(parents=True, exist_ok=True)

    def _finish(self, fig, save):
        """
        Save the figure if a file name is given, return it.
        """
        if save and self.out:
            fig.savefig(self.out / save, dpi=160)
        return fig

    @staticmethod
    def _extent(v: VesselResult):
        """
        Find the edges of the grid so each cell is drawn in the right place.
        """
        dx, dy = v.x[1] - v.x[0], v.y[1] - v.y[0]
        return [0.0, v.x[-1] + 0.5 * dx, v.y[0] - 0.5 * dy, v.y[-1] + 0.5 * dy]

    @staticmethod
    def _frame(v: VesselResult, t: float) -> int:
        """
        Find the saved snapshot closest to time t.
        """
        return int(np.argmin(np.abs(v.time - t)))

    def plot_velocity_profile(self, model: VesselModel, save: str | None = None):
        """
        Plot how fast the blood moves at each point across the vessel.
        """
        fig, ax = plt.subplots(figsize=(4, 3.4), constrained_layout=True)
        y = np.linspace(-model.R, model.R, 201)
        u = 1.5 * model.u_mean * (1 - (y / model.R) ** 2)
        ax.plot(u, y, color="#c0392b", lw=2)
        ax.fill_betweenx(y, 0, u, color="#c0392b", alpha=0.1)
        for yy in np.linspace(-0.8 * model.R, 0.8 * model.R, 7):
            uu = 1.5 * model.u_mean * (1 - (yy / model.R) ** 2)
            ax.annotate("", xy=(uu, yy), xytext=(0, yy), arrowprops=dict(arrowstyle="->", color="0.4"))
        ax.axhline(model.R, color="k", lw=1.5)
        ax.axhline(-model.R, color="k", lw=1.5)
        ax.set_xlabel("blood speed u (mm/s)")
        ax.set_ylabel("position across vessel y (mm)")
        ax.set_title("Blood speed: fastest in the centre", fontsize=10)
        return self._finish(fig, save)

    def plot_frame(self, v: VesselResult, k: int, ax, vmax: float | None = None, title: str = ""):
        """
        Draw one heat map of the drug onto an existing plot.
        """
        vmax = vmax or max(float(np.max(v.inlet_values)), 1e-12)  # full colour = inlet level
        im = ax.imshow(v.concentration[k], origin="lower", extent=self._extent(v), aspect="auto",
                       cmap="Greens", vmin=0, vmax=vmax, interpolation="bilinear")
        for yw in (-v.vessel_radius, v.vessel_radius): # black lines = vessel walls
            ax.axhline(yw, color="k", lw=1.5)
        ax.set_title(f"{title}t = {v.time[k]:.1f} s", fontsize=9)
        ax.set_ylabel("y (mm)")
        return im

    def plot_snapshots(self, v: VesselResult, times=(1, 2.5, 5, 10), title: str = "",
                       save: str | None = None, vmax: float | None = None):
        """
        Plot heat maps at several times, stacked on top of each other.
        """
        fig, axes = plt.subplots(len(times), 1, figsize=(7.5, 1.45 * len(times) + 0.6),
                                 sharex=True, constrained_layout=True)
        axes = np.atleast_1d(axes)
        im = None
        for ax, t in zip(axes, times):
            im = self.plot_frame(v, self._frame(v, t), ax, vmax=vmax, title=title)
        axes[-1].set_xlabel("x (mm) along the vessel  (blood flows left → right)")
        fig.colorbar(im, ax=axes, label="drug concentration (mg/L)", shrink=0.9)
        return self._finish(fig, save)

    def plot_profiles(self, v: VesselResult, times=(1, 2.5, 5, 10), save: str | None = None):
        """
        Plot the drug concentration along two lines through the vessel.
        """
        fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.4), constrained_layout=True)
        i_mid = len(v.x) // 2
        j_mid = int(np.argmin(np.abs(v.y)))
        for n, t in enumerate(times):
            k = self._frame(v, t)
            col = LINE_COLORS[n % len(LINE_COLORS)]
            a.plot(v.concentration[k][:, i_mid], v.y, color=col, lw=2, label=f"t = {v.time[k]:.1f} s")
            b.plot(v.x, v.concentration[k][j_mid, :], color=col, lw=2, label=f"t = {v.time[k]:.1f} s")
        for yw in (-v.vessel_radius, v.vessel_radius):
            a.axhline(yw, color="k", lw=1)
        a.set_xlabel("concentration (mg/L)")
        a.set_ylabel("y (mm)")
        a.set_title(f"Across the vessel at x = {v.x[i_mid]:.1f} mm", fontsize=10)
        b.set_xlabel("x (mm)")
        b.set_ylabel("concentration (mg/L)")
        b.set_title("Along the centre line", fontsize=10)
        a.legend(fontsize=8)
        return self._finish(fig, save)

    def plot_breakthrough(self, results: dict, save: str | None = None, title: str = ""):
        """
        Plot the concentration leaving the vessel over time, for several runs.
        """
        fig, ax = plt.subplots(figsize=(6.2, 3.6), constrained_layout=True)
        for n, (label, v) in enumerate(results.items()):
            ax.plot(v.time, v.outlet_concentration(), color=LINE_COLORS[n % 4], lw=2, label=label)
        first = next(iter(results.values()))
        ax.plot(first.time, first.inlet_values, color="0.5", ls=":", lw=1.5, label="inlet signal")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("concentration leaving the vessel (mg/L)")
        ax.set_title(title or "Breakthrough curves at the outlet", fontsize=10)
        ax.legend(fontsize=8)
        return self._finish(fig, save)

    def animate(self, v: VesselResult, fname: str = "vessel.mp4", fps: int = 12, title: str = ""):
        """
        Make a movie of the heat map changing over time.
        """
        fig, ax = plt.subplots(figsize=(8, 2.8), constrained_layout=True)
        im = self.plot_frame(v, 0, ax, title=title)
        ax.set_xlabel("x (mm) along the vessel")
        fig.colorbar(im, ax=ax, label="c (mg/L)")

        def update(k):                          
            im.set_data(v.concentration[k])
            ax.set_title(f"{title}t = {v.time[k]:.1f} s", fontsize=9)
            return (im,)

        anim = animation.FuncAnimation(fig, update, frames=len(v.time))
        path = (self.out or Path(".")) / fname
        if path.suffix == ".mp4" and animation.writers.is_available("ffmpeg"):
            anim.save(path, writer=animation.FFMpegWriter(fps=fps, bitrate=1600))
        else:
            path = path.with_suffix(".gif")
            anim.save(path, writer=animation.PillowWriter(fps=fps))
        plt.close(fig)
        return path