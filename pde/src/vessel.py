"""
Name: Chengjun Lu, Victoria Liu, Yudi Wu, Eve Zhang
Date: 10/22/2026
"""

from dataclasses import dataclass, field
import numpy as np

@dataclass
class VesselResult:
    """
    Output of one simulation.
    """
    time: np.ndarray
    x: np.ndarray
    y: np.ndarray
    concentration: np.ndarray
    lumen_rows: np.ndarray
    vessel_radius: float
    mass_in: np.ndarray = field(default=None)
    mass_out: np.ndarray = field(default=None)
    mass_eliminated: np.ndarray = field(default=None)
    mass_stored: np.ndarray = field(default=None)
    inlet_values: np.ndarray = field(default=None)
    dt: float = 0.0
    u_profile: np.ndarray = field(default=None)

    def mass_balance_error(self):
        """
        Check that no drug was lost or created by the simulation.
        """
        total_in = self.mass_in[-1]
        left = total_in - self.mass_out[-1] - self.mass_eliminated[-1] - self.mass_stored[-1]
        if total_in == 0:
            return 0.0
        return abs(left) / total_in

    def mean_lumen_concentration(self):
        """
        Average drug concentration inside the vessel at each frame.
        """
        n_frames = self.concentration.shape[0]
        result = np.zeros(n_frames)
        for i in range(n_frames):
            frame = self.concentration[i]
            rows = []
            for j in range(frame.shape[0]):
                if self.lumen_rows[j]:
                    rows.append(frame[j])
            lumen_part = np.array(rows)
            result[i] = lumen_part.mean()
        return result

    def outlet_concentration(self):
        """
        Concentration of the blood leaving the vessel at each frame.
        """
        u = self.u_profile
        total_u = u.sum()
        n_frames = self.concentration.shape[0]
        n_rows = self.concentration.shape[1]
        last_x = self.concentration.shape[2] - 1   # index of the outlet column
        result = np.zeros(n_frames)
        for i in range(n_frames):
            frame = self.concentration[i]
            c_last = np.zeros(n_rows)
            for j in range(n_rows):
                c_last[j] = frame[j, last_x]
            weighted = c_last * u
            if total_u == 0:
                result[i] = 0.0
            else:
                result[i] = weighted.sum() / total_u
        return result

@dataclass
class VesselModel:
    """
    Simulates drug moving through a straight vessel by flow and diffusion.
    """
    diffusion_coefficient: float
    vessel_length: float
    vessel_radius: float
    blood_velocity: float
    nx: int = 200
    ny: int = 24
    cfl: float = 0.4
    wall_permeability: float = 0.0
    tissue_thickness: float = 0.0
    tissue_diffusion: float = field(default=None)
    tissue_uptake: float = 0.0

    def __post_init__(self):
        """
        Set up the grid, blood speeds, wall connections and time step.
        """
        self.D = self.diffusion_coefficient
        self.L = self.vessel_length
        self.R = self.vessel_radius
        self.u_mean = self.blood_velocity
        self.P = self.wall_permeability
        if self.P > 0 and self.tissue_thickness <= 0:
            raise ValueError("wall_permeability > 0 needs tissue_thickness > 0 "
                             "(without a tissue layer the wall stays sealed)")
        if self.tissue_diffusion is not None:
            self.D_t = self.tissue_diffusion
        else:
            self.D_t = 0.3 * self.D
        self.k_t = self.tissue_uptake

        # grid
        self.nx = int(self.nx)
        self.ny_lumen = int(self.ny)
        self.dx = self.L / self.nx
        self.dy = 2.0 * self.R / self.ny_lumen
        if self.tissue_thickness > 0:
            self.n_tissue = int(round(self.tissue_thickness / self.dy))
        else:
            self.n_tissue = 0
        self.ny = self.ny_lumen + 2 * self.n_tissue
        self.x = (np.arange(self.nx) + 0.5) * self.dx
        self.y = (np.arange(self.ny) + 0.5) * self.dy - (self.R + self.n_tissue * self.dy)

        self.lumen_rows = np.zeros(self.ny, dtype=bool)
        self.D_row = np.zeros(self.ny)
        self.k_row = np.zeros(self.ny)
        for j in range(self.ny):
            if abs(self.y[j]) < self.R:
                self.lumen_rows[j] = True
                self.D_row[j] = self.D
                self.k_row[j] = 0.0
            else:
                self.lumen_rows[j] = False
                self.D_row[j] = self.D_t
                self.k_row[j] = self.k_t
        self.u_row = self.velocity_field()[:, 0]
        self.G_y = np.zeros(self.ny + 1)
        h = 0.5 * self.dy
        for j in range(1, self.ny):
            resist = h / self.D_row[j - 1] + h / self.D_row[j]
            if self.lumen_rows[j - 1] != self.lumen_rows[j]:
                if self.P > 0:
                    resist = resist + 1.0 / self.P
                else:
                    resist = np.inf
            if np.isinf(resist):
                self.G_y[j] = 0.0
            else:
                self.G_y[j] = 1.0 / resist

        u_max = max(1.5 * self.u_mean, 1e-12)
        dt_flow = self.dx / u_max
        dt_diff = 1.0 / (2.0 * float(np.max(self.D_row)) * (1.0 / self.dx ** 2 + 1.0 / self.dy ** 2))
        dt = min(dt_flow, dt_diff)
        if self.k_t > 0:
            dt = min(dt, 1.0 / self.k_t)  # uptake must not empty a tissue cell in one step
        self.dt = self.cfl * dt

    def velocity_field(self):
        """
        Blood speed in every cell.
        """
        u = np.zeros((self.ny, self.nx))
        for j in range(self.ny):
            eta = self.y[j] / self.R
            if abs(eta) < 1.0:
                speed = 1.5 * self.u_mean * (1.0 - eta ** 2)
            else:
                speed = 0.0
            u[j, :] = speed
        return u

    def _rhs(self, c, c_in):
        """
        Determine how fast the concentration is changing in every cell.

        Args:
            c (np.ndarray): Current concentration, shape (ny, nx), in mg/L.
            c_in (float): Concentration of the blood entering, in mg/L.
        
        Returns:
            tuple: (dcdt, rates)
            dcdt (np.ndarray): Rate of change in every cell (mg/L per s).
            rates (np.ndarray): Drug per second for the mass check.
        """
        ny = self.ny
        nx = self.nx
        dx = self.dx
        dy = self.dy

        ext = np.empty((ny, nx + 4))
        ext[:, 2:-2] = c
        ext[:, :2] = c_in
        ext[:, -2:] = c[:, -1:]

        c_upup = ext[:, 0:-3]   # two cells upstream of the face
        c_up = ext[:, 1:-2]     # cell just upstream of the face
        c_down = ext[:, 2:-1]   # cell just downstream of the face
        den = c_up - c_upup
        r = np.zeros_like(den)
        ok = np.abs(den) > 1e-14
        r[ok] = (c_down[ok] - c_up[ok]) / den[ok]
        phi = (r + np.abs(r)) / (1.0 + np.abs(r))  # van Leer limiter
        c_face = c_up + 0.5 * phi * den
        Fx = self.u_row[:, None] * c_face

        # axial diffusion
        Fx[:, 1:-1] = Fx[:, 1:-1] - self.D_row[:, None] * (c[:, 1:] - c[:, :-1]) / dx

        # transverse diffusion
        Fy = np.zeros((ny + 1, nx))
        Fy[1:-1, :] = -self.G_y[1:-1, None] * (c[1:, :] - c[:-1, :])

        # net change per cell
        flow_x = (Fx[:, 1:] - Fx[:, :-1]) / dx
        flow_y = (Fy[1:, :] - Fy[:-1, :]) / dy
        uptake = self.k_row[:, None] * c
        dcdt = -flow_x - flow_y - uptake

        sum_in = Fx[:, 0].sum()
        sum_out = Fx[:, nx].sum()
        sum_taken = uptake.sum()
        rates = np.array([sum_in * dy, sum_out * dy, sum_taken * dx * dy])
        return dcdt, rates

    def step(self, concentration, inlet, t=0.0, inlet_next=None, dt=None):
        """
        Move the simulation forward by one time step.

        Args:
            concentration: current concentration
            inlet (float): inlet concentration at the beginning of the step
            t: time in seconds
            inlet_next (float): Inlet concentration at the end of the step
            dt: step size in seconds
        
        Returns:
            tuple: (c_new, moved)
            c_new (np.ndarray): Concentration after the step
            moved (np.ndarray): Drug [entered, left, taken up] during the step.
        """
        if inlet_next is None:
            inlet_next = inlet
        if dt is None:
            dt = self.dt
        k1, r1 = self._rhs(concentration, inlet)
        c1 = concentration + dt * k1
        k2, r2 = self._rhs(c1, inlet_next)
        c_new = 0.5 * (concentration + c1 + dt * k2)
        return c_new, 0.5 * (r1 + r2) * dt

    def solve(self, inlet_concentration, duration, n_frames=121, initial="empty"):
        """
        Run the whole simulation.

        Args: 
            inlet_concentration (float): Inlet concentration in mg/L
            duration (float): time in seconds
            n_frames (int): Number of snapshots to save. Defaults to 121
            initial (str): starts with no drug
        
        Returns:
            VesselResult: the snapshots
        """
        if callable(inlet_concentration):
            cin = inlet_concentration
        else:
            value = float(inlet_concentration)

            def cin(t):
                return value

        c = np.zeros((self.ny, self.nx))
        if initial == "filled":
            c[self.lumen_rows, :] = cin(0.0)

        frame_t = np.linspace(0.0, duration, n_frames)
        frames = np.empty((n_frames, self.ny, self.nx))
        mass_hist = np.zeros((n_frames, 3))
        stored = np.zeros(n_frames)
        totals = np.zeros(3)
        cell = self.dx * self.dy
        stored0 = c.sum() * cell
        frames[0] = c
        stored[0] = stored0

        t = 0.0
        for k in range(1, n_frames):
            # step until we land exactly on the next frame time
            while t < frame_t[k] - 1e-12:
                h = min(self.dt, frame_t[k] - t) 
                c, moved = self.step(c, cin(t), t, cin(t + h), dt=h)
                totals += moved
                t += h
            frames[k] = c
            mass_hist[k] = totals
            stored[k] = c.sum() * cell

        inlet_values = np.zeros(n_frames)
        for i in range(n_frames):
            inlet_values[i] = cin(frame_t[i])

        return VesselResult(time=frame_t, x=self.x, y=self.y, concentration=frames,
                            lumen_rows=self.lumen_rows, vessel_radius=self.R,
                            mass_in=mass_hist[:, 0] + stored0, mass_out=mass_hist[:, 1],
                            mass_eliminated=mass_hist[:, 2], mass_stored=stored,
                            inlet_values=inlet_values, dt=self.dt,
                            u_profile=self.u_row.copy())

    @classmethod
    def from_parameters(cls, p):
        """
        Build a model from a VesselParameter object.
        """
        D = p.diffusion_coefficient_mm2_s
        L = p.length_mm
        R = p.radius_mm
        u_mean = p.mean_velocity_mm_s
        model = cls(D, L, R, u_mean, nx=p.nx, ny=p.ny, cfl=p.cfl)
        return model