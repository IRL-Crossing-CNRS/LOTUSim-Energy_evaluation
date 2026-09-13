"""
Wind Speed Sweep — Changes 8 and 9
====================================
Real NREL 5MW Cp/Ct curves from FLORIS v4 turbine library.
3-turbine aligned row, 7D spacing, 0 degree wind direction.
Wind speeds: 5 to 15 m/s.
Reference: FLORIS v4 with NREL 5MW power curve.
Layout: standard 3-turbine row as used in Niayifar & Porte-Agel (2016).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import yaml
import os
import sys

# Make models/ importable regardless of the working directory the script is
# launched from (needed for `from models.extended_models import ...` below).
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import tempfile

# Figures go to a fixed, repo-relative location so the output does not depend
# on the working directory the script happens to be launched from.
FIG_DIR = os.path.join(REPO_ROOT, 'figures', 'main')
SUPP_DIR = os.path.join(REPO_ROOT, 'figures', 'supplementary')


def tmp_path(name):
    """Scratch file in the platform temp directory (not a hardcoded /tmp)."""
    return os.path.join(tempfile.gettempdir(), name)

# =============================================================================
# JensenWakeModel
class JensenWakeModel:

    def __init__(self, diameter: float, ct: float = 0.8, air_density: float = 1.225, cp: float = 0.35,

                 cut_in: float = 5.0, cut_out: float = 25.0, kw: float = 0.04, ambient_ti=0.8):

        self.diameter = diameter

        self.ct = ct

        self.air_density = air_density

        self.cp = cp

        self.cut_in = cut_in

        self.cut_out = cut_out

        self.kw = kw

        self.ambient_ti = ambient_ti

        return

    def power(self, wind_speed, hub_height=90.0, alpha=0.12):
        if wind_speed < self.cut_in or wind_speed > self.cut_out:
            return 0.0
        area = np.pi * (self.diameter / 2.0) ** 2
        # Use rotor-averaged speed instead of hub-height speed
        U_rotor = self.rotor_averaged_speed(wind_speed, hub_height, alpha)
        return 0.5 * self.air_density * area * self.cp * U_rotor ** 3


    @staticmethod
    def normalise(vector):

        # Normalising the wind vector e.g. converting to unit vector

        vector = np.array(vector, dtype=float)

        magnitude = np.linalg.norm(vector)

        if magnitude == 0:
            raise ValueError("Wind vector cannot be zero.")

        return vector / magnitude

    @staticmethod
    def perpendicular_vect_xz(wind_unit_vector_xz):

        # Perpendicular direction in the horizontal x-z plane

        return np.array([-wind_unit_vector_xz[1], wind_unit_vector_xz[0]])

    @staticmethod
    def wind_speed(wind_vector_multi):

        u_inf = []

        for wind_vector in wind_vector_multi:
            # wind vector is 2D: [x, z]

            speed = np.sqrt(wind_vector[0] ** 2 + wind_vector[1] ** 2)

            u_inf.append(float(f"{speed:.2f}"))

        return u_inf

    def effective_kw(self, U_inf=None):
        """
        TI-dependent wake decay constant.
        Wind-speed-dependent ambient TI following offshore ABL scaling.
        Reference: Niayifar & Porte-Agel (2016), Energies 9(9), 741.
        """
        if U_inf is None:
            return self.kw  # fall back to fixed value
        TI_eff = max(0.04, self.ambient_ti * (10.0 / max(U_inf, 3.0)) ** 0.1)
        return 0.38 * TI_eff + 0.004

    def wake_radius(self, x_dist: float, U_inf=None) -> float:
        kw = self.effective_kw(U_inf)
        return self.diameter / 2 + kw * x_dist

    def in_wake(self, x_dist: float, lateral_dist: float) -> bool:

        r = self.wake_radius(x_dist)

        return abs(lateral_dist) < r

    def wake_deficit(self, ogWind: float, x_dist: float, U_inf=None) -> float:
        if x_dist <= 0:
            return 0.0
        kw = self.effective_kw(U_inf)
        factor = (1.0 - np.sqrt(1.0 - self.ct)) / (1.0 + 2.0 * kw * x_dist / self.diameter) ** 2
        deficit = ogWind * factor
        return deficit

    @staticmethod
    def combined_velocity(ogWind: float, deficits) -> float:

        if not deficits:

            return ogWind

        else:

            total_deficit = np.sqrt(sum((d / ogWind) ** 2 for d in deficits))

            total_deficit = min(total_deficit, 0.999)

            return ogWind * (1.0 - total_deficit)

    def rotational_speed_rpm(

            self,

            wind_speed: float,

            yaw_factor: float = 1.0,

            tip_speed_ratio: float = 7.0

    ) -> float:

        """

        Calculates turbine blade RPM for visual effect.

        Uses yaw-adjusted effective wind speed so lower power = slower rotation.

        """

        visual_wind_speed = wind_speed * yaw_factor

        if visual_wind_speed < self.cut_in or visual_wind_speed > self.cut_out:
            return 0.0

        radius = self.diameter / 2.0

        omega = (tip_speed_ratio * visual_wind_speed) / radius

        rpm = omega * 60.0 / (2.0 * np.pi)

        return rpm

    def shear_adjusted_speed(self, U_hub, z, hub_height, alpha=0.12):
        """
        Power law wind shear profile.
        Returns wind speed at height z given hub-height reference speed.
        alpha = 0.12 is standard for offshore neutral ABL.
        Reference: IEC 61400-1 Ed.3
        """
        return U_hub * (z / hub_height) ** alpha

    def wake_centreline_offset(self, x_dist, yaw_angle_rad):
        """
        Jimenez (2009) wake deflection model.
        Computes lateral deflection of wake centreline at downstream
        distance x due to yaw misalignment between wind and turbine.
        Larger yaw angle = more deflection = less impact on downstream turbines.
        Reference: Jimenez, A. et al. (2009), Wind Energy, 13(6), 559-572.
        """
        if abs(yaw_angle_rad) < 1e-6:
            return 0.0
        deflection = (self.ct / 2.0) * np.sin(yaw_angle_rad) \
                     * np.cos(yaw_angle_rad) ** 2 * x_dist
        return deflection

    def rotor_averaged_speed(self, U_hub, hub_height, alpha=0.12, n_points=20):
        """
        Rotor-disk-averaged wind speed accounting for vertical wind shear.
        Integrates the power law profile across the rotor swept area using
        area-weighted vertical sampling.
        Reference: Honrubia et al. (2012), Wind Energy, 15(6), 825-838.
        """
        R = self.diameter / 2.0
        # Sample heights from bottom to top of rotor disk
        z_samples = np.linspace(hub_height - R, hub_height + R, n_points)
        # Weight each sample by the chord width at that height
        # (rotor sweeps more area in the middle than at edges)
        weights = np.sqrt(np.maximum(0, R ** 2 - (z_samples - hub_height) ** 2))
        # Wind speed at each sample height
        U_samples = np.array([
            self.shear_adjusted_speed(U_hub, z, hub_height, alpha)
            for z in z_samples
        ])
        # Area-weighted average
        if weights.sum() > 0:
            return float(np.average(U_samples, weights=weights))
        return U_hub

    def partial_overlap_factor(self, x_dist, lateral_dist, U_inf=None):
        R_rotor = self.diameter / 2.0
        R_wake = self.wake_radius(x_dist, U_inf)
        d = abs(lateral_dist)

        # Fully outside wake — no overlap
        if d >= R_rotor + R_wake:
            return 0.0

        # Fully inside wake — full overlap
        if d + R_rotor <= R_wake:
            return 1.0

        # Wake fully inside rotor — scale by wake/rotor area ratio
        if d + R_wake <= R_rotor:
            return (R_wake / R_rotor) ** 2

        # Partial overlap — geometric circle-circle intersection
        d1 = (d ** 2 + R_rotor ** 2 - R_wake ** 2) / (2 * d)
        d2 = d - d1

        cos1 = np.clip(d1 / R_rotor, -1, 1)
        cos2 = np.clip(d2 / R_wake, -1, 1)

        A_overlap = (R_rotor ** 2 * np.arccos(cos1)
                     - d1 * np.sqrt(max(0, R_rotor ** 2 - d1 ** 2))
                     + R_wake ** 2 * np.arccos(cos2)
                     - d2 * np.sqrt(max(0, R_wake ** 2 - d2 ** 2)))

        A_rotor = np.pi * R_rotor ** 2
        return float(np.clip(A_overlap / A_rotor, 0.0, 1.0))

    def wind_speeds_full(self, turbines, ogWind: float, wind_vector, debug: bool = False):

        wind_xz = np.array([wind_vector[0], wind_vector[1]], dtype=float)

        if np.linalg.norm(wind_xz) == 0:
            raise ValueError("Horizontal wind vector cannot be zero.")

        w = wind_xz / np.linalg.norm(wind_xz)

        w_perp = self.perpendicular_vect_xz(w)

        # Turbines face south (-z). Yaw factor = how much wind aligns with +z axis.

        # Pure north wind [0,1] → factor=1.0 (full power)

        # Pure east/west [±1,0] → factor=0.0 (no power, rotor not facing wind)

        turbine_facing = np.array([0.0, 1.0])  # +z = into the rotor from the south

        yaw_factor = abs(np.dot(w, turbine_facing))  # 0.0 to 1.0

        turbines_sorted = sorted(

            turbines,

            key=lambda t: np.dot(np.array([t[0], t[2]]), w)

        )

        velocities = []

        rpms = []

        for i, (x_i, y_i, z_i) in enumerate(turbines_sorted):

            # Sequential local velocity superposition
            U_local = ogWind

            for j in range(i):
                x_j, y_j, z_j = turbines_sorted[j]
                delta_xz = np.array([x_i - x_j, z_i - z_j])
                x_dist = np.dot(delta_xz, w)
                lateral_dist = np.dot(delta_xz, w_perp)

                yaw_angle = np.arctan2(w[0], w[1])
                wake_offset = self.wake_centreline_offset(x_dist, yaw_angle)
                effective_lateral = lateral_dist - wake_offset

                if x_dist > 1e-9:
                    overlap = self.partial_overlap_factor(x_dist, effective_lateral, U_inf=ogWind)
                    if overlap > 0:
                        deficit = self.wake_deficit(U_local, x_dist, U_inf=ogWind) * overlap
                        if deficit > 1e-6:
                            U_local = max(0.0, U_local - deficit)

            v_eff = U_local * yaw_factor

            rpm = self.rotational_speed_rpm(v_eff)

            velocities.append(round(v_eff, 2))

            rpms.append(rpm)

            if debug:
                print(f"Turbine {i} at (x={x_i}, z={z_i}): yaw={yaw_factor:.3f}, "

                      f"v_eff={v_eff:.2f} m/s, rpm={rpm:.1f}")

        return turbines_sorted, velocities, rpms

    def multi_speed(self, turbines, wind_vector_multi, times):

        ogWind_speeds = self.wind_speed(wind_vector_multi)

        if not (len(times) == len(wind_vector_multi)):
            raise ValueError("The length of times and wind_vector_multi must match")

        total_energy_kwh = 0.0

        results = []

        for i in range(len(times)):
            ogWind = ogWind_speeds[i]

            wind_vector = wind_vector_multi[i]

            hours = times[i]

            turbines_sorted, velocities, rpms = self.wind_speeds_full(turbines, ogWind, wind_vector)

            powers = [self.power(v) for v in velocities]

            farm_power_w = sum(powers)

            farm_energy_kwh = (farm_power_w / 1000.0) * hours

            total_energy_kwh += farm_energy_kwh

            results.append({

                "interval": i,

                "ogWind": ogWind,

                "wind_vector": wind_vector,

                "hours": hours,

                "turbines": turbines_sorted,

                "velocities": velocities,

                "rpms": rpms,

                "Power_w": powers,

                "farm_power_w": farm_power_w,

                "farm_energy_kwh": farm_energy_kwh,

            })

        return total_energy_kwh, results

    def plot_wind_field(self, turbines, wind_vector):

        """

        Plot wind field in x–z plane:

            x = horizontal spacing

            z = downstream (rows), increasing northward

        """

        # Extract turbine positions (x, z only)

        x_t = [t[0] for t in turbines]

        z_t = [t[2] for t in turbines]

        # Create grid

        x = np.linspace(min(x_t) - 400, max(x_t) + 400, 10)

        z = np.linspace(min(z_t) - 400, max(z_t) + 400, 10)

        X, Z = np.meshgrid(x, z)

        # Normalise wind vector (2D: [x, z])

        w = np.array(wind_vector, dtype=float)

        w = w / np.linalg.norm(w)

        # Uniform vector field

        u = np.full_like(X, w[0])  # x-component

        v = np.full_like(Z, w[1])  # z-component

        plt.figure(figsize=(8, 6))

        # Wind field

        plt.quiver(X, Z, u, v, color='grey', alpha=0.6)

        # Turbines

        plt.scatter(x_t, z_t, s=150, edgecolors='black', color='red')

        # Labels

        for i, (x, y, z) in enumerate(turbines):
            plt.text(x + 10, z + 10, f"T{i}")

        plt.xlabel("X (m) – Turbine spacing")

        plt.ylabel("Z (m) – Downstream direction (North)")

        plt.title("Wind Field and Turbine Layout (Top View, Wind Travelling North)")

        plt.axis("equal")

        plt.grid()

        plt.show()


# GaussianWakeModel
class GaussianWakeModel:

    def __init__(self, diameter: float, ct: float, air_density: float = 1.225, cp: float = 0.35, cut_in: float = 5.0,
                 cut_out: float = 25.0, k_y: float = 0.05, k_z: float = 0.05, eps: float = 0.2,
                 ambient_ti: float = None):

        self.diameter = diameter

        self.ct = ct

        self.air_density = air_density

        self.cp = cp

        self.cut_in = cut_in

        self.cut_out = cut_out

        self.k_y = k_y  # describes the width expansion of the wake downstream

        self.k_z = k_z  # describes the height expansion of the wake downstream

        self.eps = eps  # parameter linked to the starting width of the wake ratio to diameter

        self.ambient_ti = ambient_ti

        # If TI provided, use Niayifar & Porte-Agel (2016) TI-dependent expansion
        # k = 0.38 * TI + 0.004
        # Otherwise use user-supplied k_y and k_z
        if ambient_ti is not None:
            self.k_y = 0.50 * ambient_ti + 0.003
            self.k_z = 0.38 * ambient_ti + 0.004
        else:
            self.k_y = k_y
            self.k_z = k_z

    def power(self, wind_speed, hub_height=90.0, alpha=0.12):
        if wind_speed < self.cut_in or wind_speed > self.cut_out:
            return 0.0
        area = np.pi * (self.diameter / 2.0) ** 2
        # Use rotor-averaged speed instead of hub-height speed
        U_rotor = self.rotor_averaged_speed(wind_speed, hub_height, alpha)
        return 0.5 * self.air_density * area * self.cp * U_rotor ** 3

    def local_ti(self, TI_amb, ct, x_dist, diameter):
        """
        Crespo-Hernandez added turbulence intensity model.
        Computes local TI inside the wake at downstream distance x.
        Higher TI inside wakes speeds up wake recovery for downstream turbines.
        Reference: Crespo & Hernandez (1996), J. Wind Eng. Ind. Aerodyn., 61(1), 71-85.
        """
        if x_dist <= 0:
            return TI_amb
        x_D = x_dist / diameter
        TI_added = 0.5 * (ct * TI_amb) ** 0.25 * x_D ** (-0.32)
        return TI_amb + TI_added

    @staticmethod
    def normalise(vector):

        # Normalising the wind vector e.g. converting to unit vector

        vector = np.array(vector, dtype=float)

        magnitude = np.linalg.norm(vector)

        if magnitude == 0:
            raise ValueError("Wind vector cannot be zero.")

        return vector / magnitude

    @staticmethod
    def perpendicular_vect(wind_unit_vector):

        # perpendicular direction in the horizontal x-z plane

        return np.array([-wind_unit_vector[1], wind_unit_vector[0]])

    def rotational_speed_rpm(self, wind_speed: float, tip_speed_ratio: float = 7.0) -> float:

        """

        Calculates turbine blade RPM from effective wind speed.

        Already accounts for yaw since wind_speed is yaw-adjusted before this is called.

        """

        if wind_speed < self.cut_in or wind_speed > self.cut_out:
            return 0.0

        radius = self.diameter / 2.0

        omega = (tip_speed_ratio * wind_speed) / radius

        return omega * 60.0 / (2.0 * np.pi)

    # start of Gaussian Wake model

    def gaussian_wake_deficit_full(self, ogWind, x_dist, y_dist, z_dist,
                                   sigma_y=None, sigma_z=None):

        if x_dist <= 0:
            return 0.0

        if sigma_y is None:
            sigma_y = self.k_y * x_dist + self.eps * self.diameter
        if sigma_z is None:
            sigma_z = self.k_z * x_dist + self.eps * self.diameter

        if sigma_y <= 0 or sigma_z <= 0:
            return 0.0

        # Amplitude: how much the centre line is reduced

        sigma_d = self.k_y * x_dist / self.diameter + self.eps
        amplitude = 1.0 - np.sqrt(max(0.0, 1.0 - self.ct / (8 * sigma_d ** 2)))

        # Gaussian spatial spread — how the deficit distributes laterally and vertically

        exponent = -0.5 * ((y_dist / sigma_y) ** 2 + (z_dist / sigma_z) ** 2)

        spread = np.exp(exponent)

        return ogWind * amplitude * spread

    # calculating overlapping wakes

    @staticmethod
    def wind_speed(wind_vector_multi):

        u_inf = []

        for wind_vector in wind_vector_multi:
            # wind vector is 2D: [x, z]

            speed = np.sqrt(wind_vector[0] ** 2 + wind_vector[1] ** 2)

            u_inf.append(float(f"{speed:.2f}"))

        return u_inf

    def shear_adjusted_speed(self, U_hub, z, hub_height, alpha=0.12):
        """
        Power law wind shear profile.
        Returns wind speed at height z given hub-height reference speed.
        alpha = 0.12 is standard for offshore neutral ABL.
        Reference: IEC 61400-1 Ed.3
        """
        return U_hub * (z / hub_height) ** alpha

    def wake_centreline_offset(self, x_dist, yaw_angle_rad):
        """
        Jimenez (2009) wake deflection model.python

# Compute yaw angle — angle between wind direction and turbine facing
# Turbines face south (+z), yaw occurs when wind is offset from this
yaw_angle = np.arctan2(w[0], w[1])  # angle from +z axis

# Apply wake deflection to effective lateral distance
wake_offset = self.wake_centreline_offset(x_dist, yaw_angle)
effective_lateral = lateral_dist - wake_offset

Then use effective_lateral instead of lateral_dist in all subsequent wake calculations — the in_wake check, partial overlap, and Gaussian sigma calculations.
        Computes lateral deflection of wake centreline at downstream
        distance x due to yaw misalignment between wind and turbine.
        Larger yaw angle = more deflection = less impact on downstream turbines.
        Reference: Jimenez, A. et al. (2009), Wind Energy, 13(6), 559-572.
        """
        if abs(yaw_angle_rad) < 1e-6:
            return 0.0
        deflection = (self.ct / 2.0) * np.sin(yaw_angle_rad) \
                     * np.cos(yaw_angle_rad) ** 2 * x_dist
        return deflection

    def rotor_averaged_speed(self, U_hub, hub_height, alpha=0.12, n_points=20):
        """
        Rotor-disk-averaged wind speed accounting for vertical wind shear.
        Integrates the power law profile across the rotor swept area using
        area-weighted vertical sampling.
        Reference: Honrubia et al. (2012), Wind Energy, 15(6), 825-838.
        """
        R = self.diameter / 2.0
        # Sample heights from bottom to top of rotor disk
        z_samples = np.linspace(hub_height - R, hub_height + R, n_points)
        # Weight each sample by the chord width at that height
        # (rotor sweeps more area in the middle than at edges)
        weights = np.sqrt(np.maximum(0, R ** 2 - (z_samples - hub_height) ** 2))
        # Wind speed at each sample height
        U_samples = np.array([
            self.shear_adjusted_speed(U_hub, z, hub_height, alpha)
            for z in z_samples
        ])
        # Area-weighted average
        if weights.sum() > 0:
            return float(np.average(U_samples, weights=weights))
        return U_hub

    @staticmethod
    def combined_velocity(ogWind: float, deficits) -> float:

        # combining multiple wake deficits using root sum square

        if not deficits:
            return ogWind

        total_deficit = np.sqrt(sum((d / ogWind) ** 2 for d in deficits))

        total_deficit = min(total_deficit, 0.999)  # to prevent negative wind speeds

        return ogWind * (1.0 - total_deficit)

    # Main section

    def wind_speeds_full(self, turbines, ogWind, wind_vector, debug: bool = False):

        wind_xz = np.array([wind_vector[0], wind_vector[1]], dtype=float)

        if np.linalg.norm(wind_xz) == 0:
            raise ValueError("Wind vector cannot be zero.")

        w = wind_xz / np.linalg.norm(wind_xz)

        w_perp = self.perpendicular_vect(w)

        # Yaw factor — turbines face south (-z), wind must have +z component

        turbine_facing = np.array([0.0, 1.0])

        yaw_factor = abs(np.dot(w, turbine_facing))

        turbines_sorted = sorted(

            turbines,

            key=lambda t: np.dot(np.array([t[0], t[2]]), w)

        )

        velocities = []

        rpms = []

        for i, (x_i, y_i, z_i) in enumerate(turbines_sorted):

            # Sequential local velocity superposition
            U_local = ogWind

            for j in range(i):
                x_j, y_j, z_j = turbines_sorted[j]
                delta_xz = np.array([x_i - x_j, z_i - z_j])
                x_dist = np.dot(delta_xz, w)
                lateral_dist = np.dot(delta_xz, w_perp)
                vertical_dist = y_i - y_j

                yaw_angle = np.arctan2(w[0], w[1])
                wake_offset = self.wake_centreline_offset(x_dist, yaw_angle)
                effective_lateral = lateral_dist - wake_offset

                if x_dist <= 1e-9:
                    continue

                # Compute local TI at target turbine from upstream turbine j
                local_TI = self.local_ti(self.ambient_ti, self.ct, x_dist, self.diameter)
                # Use local TI for wake expansion
                k_local = 0.38 * local_TI + 0.004
                # Override sigma with locally computed expansion
                sigma_y_local = k_local * x_dist + self.eps * self.diameter
                sigma_z_local = k_local * x_dist + self.eps * self.diameter

                deficit = self.gaussian_wake_deficit_full(
                    U_local, x_dist, effective_lateral, vertical_dist,
                    sigma_y=sigma_y_local, sigma_z=sigma_z_local)

                if deficit > 1e-6:
                    U_local = max(0.0, U_local - deficit)

            u_eff = U_local * yaw_factor

            rpm = self.rotational_speed_rpm(u_eff)

            velocities.append(round(u_eff, 2))

            rpms.append(rpm)

        return turbines_sorted, velocities, rpms

    def multi_speed(self, turbines, wind_vector_multi, times):

        ogWind_speeds = self.wind_speed(wind_vector_multi)

        if not (len(times) == len(wind_vector_multi)):
            raise ValueError("The length of times and wind_vector_multi must match")

        total_energy_kwh = 0.0

        results = []

        for i in range(len(times)):
            ogWind = ogWind_speeds[i]

            wind_vector = wind_vector_multi[i]

            hours = times[i]

            turbines_sorted, velocities, rpms = self.wind_speeds_full(turbines, ogWind, wind_vector)

            powers = [self.power(v) for v in velocities]

            farm_power_w = sum(powers)

            farm_energy_kwh = (farm_power_w / 1000.0) * hours

            total_energy_kwh += farm_energy_kwh

            results.append({

                "interval": i,

                "ogWind": ogWind,

                "wind_vector": wind_vector,

                "hours": hours,

                "turbines": turbines_sorted,

                "velocities": velocities,

                "rpms": rpms,

                "Power_w": powers,

                "farm_power_w": farm_power_w,

                "farm_energy_kwh": farm_energy_kwh,

            })

        return total_energy_kwh, results

    def plot_wind_field(self, turbines, wind_vector):

        """

        Top-view plot in the x-z plane.

        x = lateral spacing

        z = row/downstream distance

        y = height, ignored in this plot

        """

        x_t = [t[0] for t in turbines]

        z_t = [t[2] for t in turbines]

        x = np.linspace(min(x_t) - 200, max(x_t) + 200, 10)

        z = np.linspace(min(z_t) - 200, max(z_t) + 200, 10)

        X, Z = np.meshgrid(x, z)

        # wind vector is [x, z]

        w = np.array(wind_vector, dtype=float)

        w = w / np.linalg.norm(w)

        u = np.full_like(X, w[0])

        v = np.full_like(Z, w[1])

        plt.figure(figsize=(8, 6))

        plt.quiver(X, Z, u, v, color='grey', alpha=0.6)

        plt.scatter(x_t, z_t, s=150, edgecolors='black', color='red')

        for i, (x, y, z) in enumerate(turbines):
            plt.text(x + 10, z + 10, f"T{i}")

        plt.xlabel("X (m) - lateral turbine spacing")

        plt.ylabel("Z (m) - row/downstream direction")

        plt.title("Wind Field and Turbine Layout in X-Z Plane")

        plt.axis("equal")

        plt.grid()

        plt.show()


# LarsenWakeModel
class LarsenWakeModel:

    def __init__(

            self,

            diameter: float,

            ct: float,

            air_density: float = 1.225,

            cp: float = 0.35,

            ambient_ti: float = 0.08,  # Turbulence intensity, offshore (0.05-0.1)

            cut_in_speed: float = 3.0,

            cut_out_speed: float = 25.0,

            rated_power: float = None  # rated power if turbine only produce power up to a certain value

    ):

        self.diameter = float(diameter)

        self.radius = self.diameter / 2

        self.area = np.pi * self.radius ** 2

        self.ct = float(ct)

        self.air_density = float(air_density)

        self.cp = float(cp)

        self.ambient_ti = float(ambient_ti)

        self.cut_in = float(cut_in_speed)

        self.cut_out = float(cut_out_speed)

        self.rated_power = rated_power

    # Geometry helpers

    def _unit_vector(self, wind_vector):

        wind_vector = np.array(wind_vector, dtype=float)

        mag = np.linalg.norm(wind_vector)

        if mag == 0:
            raise ValueError("Wind vector cannot be zero.")

        return wind_vector / mag, mag

    def _perpendicular_vector(self, unit_wind):

        return np.array([-unit_wind[1], unit_wind[0]])

    def _project_coordinates(self, turbines, wind_vector):

        turbines = np.array(turbines, dtype=float)

        if turbines.ndim != 2 or turbines.shape[1] != 3:
            raise ValueError("turbines must be shape (n_turbines, 3).")

        wind_xz = np.array([wind_vector[0], wind_vector[1]], dtype=float)

        unit_wind, wind_speed = self._unit_vector(wind_xz)

        perp = self._perpendicular_vector(unit_wind)

        # x-z is the horizontal plane, y is height

        xz = turbines[:, [0, 2]]

        height = turbines[:, 1]

        downstream = xz @ unit_wind

        crosswind = xz @ perp

        # Turbines face south (-z). Yaw factor = alignment with +z (north = into rotor).

        # Pure north [0,1]  → 1.0 (full power)

        # Pure east  [1,0]  → 0.0 (no power)

        # Diagonal   [5,12] → partial power

        turbine_facing = np.array([0.0, 1.0])

        yaw_factor = abs(float(np.dot(unit_wind, turbine_facing)))

        return downstream, crosswind, height, wind_speed, yaw_factor

    # Power

    def power(self, wind_speed, hub_height=90.0, alpha=0.12):
        if wind_speed < self.cut_in or wind_speed > self.cut_out:
            return 0.0
        area = np.pi * (self.diameter / 2.0) ** 2
        # Use rotor-averaged speed instead of hub-height speed
        U_rotor = self.rotor_averaged_speed(wind_speed, hub_height, alpha)
        return 0.5 * self.air_density * area * self.cp * U_rotor ** 3

    def local_ti(self, TI_amb, ct, x_dist, diameter):
        """
        Crespo-Hernandez added turbulence intensity model.
        Computes local TI inside the wake at downstream distance x.
        Higher TI inside wakes speeds up wake recovery for downstream turbines.
        Reference: Crespo & Hernandez (1996), J. Wind Eng. Ind. Aerodyn., 61(1), 71-85.
        """
        if x_dist <= 0:
            return TI_amb
        x_D = x_dist / diameter
        TI_added = 0.5 * (ct * TI_amb) ** 0.25 * x_D ** (-0.32)
        return TI_amb + TI_added

    # Larsen parameters

    def _m(self):

        # induction factor, higher ct, higher m and stronger wake

        return 1.0 / np.sqrt(1.0 - self.ct)

    def _k(self):

        # wake expansion factor, how fast wake radius grows with distance

        m = self._m()

        return np.sqrt((m + 1.0) / 2.0)

    def _R96(self):

        # wake radius at 9.6 D downstream, a1 to a4 and b1 were derived by fitting to wind tunnel measurements

        a1 = 0.435449861
        a2 = 0.797853685
        a3 = -0.124807893
        a4 = 0.136821858
        b1 = 9.5  # originally 15.6298

        return (

                a1

                * np.exp(a2 * self.ct ** 2 + a3 * self.ct + a4)

                * (b1 * self.ambient_ti + 1.0)

                * self.diameter

        )

    def _x0(self):

        # virtual origin distance for 9.6D wake

        R96 = self._R96()

        k = self._k()

        denom = (2.0 * R96 / (k * self.diameter)) ** 3 - 1.0

        if np.isclose(denom, 0.0):
            raise ValueError("Invalid denominator in Larsen x0 calculation.")

        return 9.6 * self.diameter / denom

    def _c1(self):

        # deficit amplitude constant

        k = self._k()

        x0 = self._x0()

        return (

                ((k * self.diameter / 2.0) ** (5.0 / 2.0))

                * ((105.0 / (2.0 * np.pi)) ** (-1.0 / 2.0))

                * ((self.ct * self.area * x0) ** (-5.0 / 6.0))

        )

    def wake_radius(self, x):

        if x <= 0:
            return self.radius

        c1 = self._c1()

        x0 = self._x0()

        return (

                ((105.0 * c1 ** 2) / (2.0 * np.pi)) ** (1.0 / 5.0)

                * (self.ct * self.area * (x + x0)) ** (1.0 / 3.0)

        )

    def shear_adjusted_speed(self, U_hub, z, hub_height, alpha=0.12):
        """
        Power law wind shear profile.
        Returns wind speed at height z given hub-height reference speed.
        alpha = 0.12 is standard for offshore neutral ABL.
        Reference: IEC 61400-1 Ed.3
        """
        return U_hub * (z / hub_height) ** alpha

    def wake_centreline_offset(self, x_dist, yaw_angle_rad):
        """
        Jimenez (2009) wake deflection model.
        Computes lateral deflection of wake centreline at downstream
        distance x due to yaw misalignment between wind and turbine.
        Larger yaw angle = more deflection = less impact on downstream turbines.
        Reference: Jimenez, A. et al. (2009), Wind Energy, 13(6), 559-572.
        """
        if abs(yaw_angle_rad) < 1e-6:
            return 0.0
        deflection = (self.ct / 2.0) * np.sin(yaw_angle_rad) \
                     * np.cos(yaw_angle_rad) ** 2 * x_dist
        return deflection

    def meandering_factor(self, x_dist, r):
        """
        Dynamic wake meandering correction for time-averaged deficit.
        Large-scale atmospheric turbulence causes the wake to meander
        laterally, reducing the time-averaged deficit at a fixed point.
        The correction smears the deficit by convolving with a Gaussian
        displacement distribution with sigma = 0.5 * wake_radius(x).
        Reference: Larsen, G.C. et al. (2008), Wind Energy, 11(4), 377-395.
        """
        R_wake = self.wake_radius(x_dist)
        if R_wake <= 0:
            return 1.0

        sigma_m = 0.5 * R_wake

        # Gaussian weighting — deficit at r is reduced by meandering
        weight = np.exp(-0.5 * (r / (sigma_m + 1e-6)) ** 2)

        # Scale between 0.7 and 1.0 — meandering never fully removes deficit
        return float(np.clip(0.7 + 0.3 * weight, 0.7, 1.0))

    def rotor_averaged_speed(self, U_hub, hub_height, alpha=0.12, n_points=20):
        """
        Rotor-disk-averaged wind speed accounting for vertical wind shear.
        Integrates the power law profile across the rotor swept area using
        area-weighted vertical sampling.
        Reference: Honrubia et al. (2012), Wind Energy, 15(6), 825-838.
        """
        R = self.diameter / 2.0
        # Sample heights from bottom to top of rotor disk
        z_samples = np.linspace(hub_height - R, hub_height + R, n_points)
        # Weight each sample by the chord width at that height
        # (rotor sweeps more area in the middle than at edges)
        weights = np.sqrt(np.maximum(0, R ** 2 - (z_samples - hub_height) ** 2))
        # Wind speed at each sample height
        U_samples = np.array([
            self.shear_adjusted_speed(U_hub, z, hub_height, alpha)
            for z in z_samples
        ])
        # Area-weighted average
        if weights.sum() > 0:
            return float(np.average(U_samples, weights=weights))
        return U_hub

    def deficit(self, x, r, U_inf):

        """
        First-order Larsen deficit.
        Returns positive deficit magnitude.
        """

        if x <= 0:
            return 0.0

        c1 = self._c1()

        x0 = self._x0()

        term1 = -U_inf / 9.0

        term2 = (self.ct * self.area * (x + x0) ** (-2.0)) ** (1.0 / 3.0)

        bracket = (

                r ** (3.0 / 2.0)

                * (3.0 * c1 ** 2 * self.ct * self.area * (x + x0)) ** (-1.0 / 2.0)

                - (35.0 / (2.0 * np.pi)) ** (3.0 / 10.0)

                * (3.0 * c1 ** 2) ** (-1.0 / 5.0)

        )

        delta_u = term1 * term2 * (bracket ** 2)

        return max(0.0, -delta_u)

    def deficit_second_order(self, x, r, U_inf):
        """
        Second-order Larsen wake deficit correction.
        Adds pressure recovery term to the first-order solution,
        improving accuracy at large downstream distances (x > 10D)
        where first-order models underestimate wake recovery.
        The second-order term is derived from the next term in the
        series expansion of the Prandtl boundary layer equations.
        Reference: Larsen, G.C. (2009). Risoe-R-1713(EN), DTU.
        """
        if x <= 0:
            return 0.0

        c1 = self._c1()
        x0 = self._x0()
        A = np.pi * self.radius ** 2

        # Second-order correction term
        # This captures the pressure-driven recovery beyond the near wake
        term1 = (self.ct * A * (x + x0) ** (-2)) ** (1.0 / 3.0)

        bracket = (
                r ** (3.0 / 2.0)
                * (3.0 * c1 ** 2 * self.ct * A * (x + x0)) ** (-1.0 / 2.0)
                - (35.0 / (2.0 * np.pi)) ** (3.0 / 10.0)
                * (3.0 * c1 ** 2) ** (-1.0 / 5.0)
        )

        # Second-order coefficient — empirically derived from
        # Larsen (2009) second-order boundary layer expansion
        alpha_2 = 0.07

        delta_u_2 = -(U_inf / 9.0) * term1 * bracket ** 2 * alpha_2 * (
                1.0 - np.exp(-((x / (10.0 * self.diameter)) ** 2))
        )

        return max(0.0, -delta_u_2)

    def velocity_at_point(self, x, r, U_inf):
        """
        Returns local velocity using first + second order Larsen deficit.
        Second-order term improves accuracy at x > 10D downstream.
        """
        if x <= 0:
            return U_inf

        rw = self.wake_radius(x)
        if abs(r) > rw:
            return U_inf

        # First-order deficit
        d1 = self.deficit(x, r, U_inf)

        # Second-order correction — only significant beyond ~5D
        d2 = self.deficit_second_order(x, r, U_inf) if x > 5 * self.diameter else 0.0

        total_deficit = d1 - d2  # second order reduces the deficit (faster recovery)
        return max(0.0, U_inf - total_deficit)

    def rotational_speed_rpm(self, wind_speed: float, tip_speed_ratio: float = 7.0) -> float:

        if wind_speed < self.cut_in or wind_speed >= self.cut_out:
            return 0.0

        radius = self.diameter / 2.0

        omega = (tip_speed_ratio * wind_speed) / radius

        return omega * 60.0 / (2.0 * np.pi)

    # ------------------------------------------------------------

    # One interval

    # ------------------------------------------------------------

    def wind_speeds_full(self, turbines, wind_vector, debug: bool = False):

        """

        Turbine coordinates:

            x = lateral turbine spacing

            y = height

            z = distance between rows / downstream direction



        Wind vector: [x, z]



        Turbines face south (-z direction).

        Wind travelling north = positive z-component.

        Pure east/west wind produces zero output.



        Returns:

            turbines_sorted : list of (x, y, z) sorted upstream → downstream

            velocities      : yaw-adjusted effective wind speed (m/s) per turbine

            rpms            : blade RPM per turbine

        """

        turbines_arr = np.array(turbines, dtype=float)

        wind_vector = np.array(wind_vector, dtype=float)

        downstream, crosswind, height, U_inf, yaw_factor = self._project_coordinates(

            turbines_arr, wind_vector

        )

        # Sort upstream → downstream

        order = np.argsort(downstream)

        turbine_speeds = np.zeros(len(turbines_arr))

        turbine_rpms = np.zeros(len(turbines_arr))

        for idx in order:

            v_eff = self._effective_speed_at_turbine(idx, turbines_arr, wind_vector)

            turbine_speeds[idx] = v_eff

            turbine_rpms[idx] = self.rotational_speed_rpm(v_eff)

            if debug:
                x, y, z = turbines_arr[idx]

                print(f"Turbine at (x={x}, z={z}): yaw={yaw_factor:.3f}, "

                      f"v_eff={v_eff:.2f} m/s, rpm={turbine_rpms[idx]:.1f}")

        turbines_sorted = turbines_arr[order].tolist()

        velocities = [round(float(turbine_speeds[i]), 2) for i in order]

        rpms = [round(float(turbine_rpms[i]), 1) for i in order]

        return turbines_sorted, velocities, rpms

    def _effective_speed_at_turbine(self, turbine_index, turbines, wind_vector):

        downstream, crosswind, height, U_inf, yaw_factor = self._project_coordinates(
            turbines, wind_vector
        )

        target_down = downstream[turbine_index]
        target_cross = crosswind[turbine_index]
        target_height = height[turbine_index]

        # Sort upstream turbines by downstream distance
        upstream_indices = [
            j for j in range(len(turbines))
            if j != turbine_index and downstream[j] < target_down - 1e-9
        ]
        upstream_indices.sort(key=lambda j: downstream[j])

        # Start with freestream and reduce sequentially
        U_local = U_inf

        for j in upstream_indices:
            dx = target_down - downstream[j]
            dy = target_cross - crosswind[j]
            dh = target_height - height[j]
            r = np.sqrt(dy ** 2 + dh ** 2)

            yaw_angle = np.arctan2(wind_vector[0], wind_vector[1])
            wake_offset = self.wake_centreline_offset(dx, yaw_angle)
            effective_r = np.sqrt((dy - wake_offset) ** 2 + dh ** 2)

            # Check if inside wake radius
            rw = self.wake_radius(dx)
            if abs(r) > rw:
                continue

            # Compute deficit using LOCAL velocity as inflow (not U_inf)
            # This is the key change — Larsen deficit scaled to local conditions
            local_TI = self.local_ti(self.ambient_ti, self.ct, dx, self.diameter)
            k_amb = 0.38 * self.ambient_ti + 0.004
            k_local = 0.38 * local_TI + 0.004
            TI_ratio = (k_amb / k_local) ** (1 / 2)
            meander = self.meandering_factor(dx, effective_r)
            # First-order deficit
            d1 = self.deficit(dx, effective_r, U_local)

            # Second-order correction (faster recovery at large x)
            d2 = self.deficit_second_order(dx, effective_r, U_local) \
                if dx > 5 * self.diameter else 0.0

            deficit = max(0.0, d1 - d2) * TI_ratio * meander

            # Apply deficit sequentially
            U_local = max(0.0, U_local - deficit)

        return U_local * yaw_factor

    def single_speed(self, turbines, wind_vector, hours, interval=0):

        turbines_arr = np.array(turbines, dtype=float)

        wind_vector = np.array(wind_vector, dtype=float)

        _, _, _, ogWind, _ = self._project_coordinates(turbines_arr, wind_vector)

        turbines_sorted, velocities, rpms = self.wind_speeds_full(turbines_arr, wind_vector)

        powers = [self.power(v) for v in velocities]

        farm_power = sum(powers)

        farm_energy_kwh = farm_power * hours / 1000

        return {

            "interval": interval,

            "ogWind": ogWind,

            "wind_vector": wind_vector.tolist(),

            "hours": hours,

            "turbines": turbines_sorted,

            "velocities": velocities,

            "rpms": rpms,

            "Power_w": powers,

            "farm_power_w": farm_power,

            "farm_energy_kwh": farm_energy_kwh

        }

    def multi_speed(self, turbines, wind_vectors, times):

        if len(wind_vectors) != len(times):
            raise ValueError("wind_vectors and times must have the same length.")

        results = []

        total_energy_kwh = 0.0

        for i, (w, t) in enumerate(zip(wind_vectors, times)):
            result = self.single_speed(turbines, w, t, interval=i)

            results.append(result)

            total_energy_kwh += result["farm_energy_kwh"]

        return total_energy_kwh, results

    # Plotting

    def plot_wind_field(self, turbines, wind_vector):

        """

        Top-view plot in the x-z plane.

        x = lateral spacing

        z = row/downstream distance

        y = height, ignored in this plot.

        """

        turbines = np.array(turbines, dtype=float)

        plt.figure(figsize=(8, 6))

        # Plot x-z positions

        plt.scatter(turbines[:, 0], turbines[:, 2], s=100)

        for i, (x, y, z) in enumerate(turbines):
            plt.text(x + 5, z + 5, f"T{i + 1}\ny={y:.1f} m")

        wind_vector = np.array(wind_vector, dtype=float)

        mag = np.linalg.norm(wind_vector)

        if mag > 0:
            unit = wind_vector / mag

            centre = np.mean(turbines[:, [0, 2]], axis=0)

            plt.arrow(

                centre[0], centre[1],

                unit[0] * self.diameter * 2,

                unit[1] * self.diameter * 2,

                head_width=self.diameter * 0.2,

                length_includes_head=True

            )

        plt.xlabel("X position (m) - lateral spacing")

        plt.ylabel("Z position (m) - row/downstream direction")

        plt.title("Larsen Wake Model Wind Field in X-Z Plane")

        plt.axis("equal")

        plt.grid(True)

        plt.show()

# =============================================================================
# NREL 5MW Cp/Ct LOOKUP TABLES
# Source: FLORIS v4 turbine library
# =============================================================================

WIND_SPEEDS = np.array([
    2.9, 3.0, 4.0, 5.0, 6.0, 7.0, 7.1, 7.2, 7.3, 7.4, 7.5,
    7.6, 7.7, 7.8, 7.9, 8.0, 9.0, 10.0, 10.1, 10.2, 10.3, 10.4,
    10.5, 10.6, 10.7, 10.8, 10.9, 11.0, 11.1, 11.2, 11.3, 11.4,
    11.5, 11.6, 11.7, 11.8, 11.9, 12.0, 13.0, 14.0, 15.0, 16.0,
    17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 25.1, 50.0
])

CP_CURVE = np.array([
    0.0000, 0.1965, 0.3635, 0.4231, 0.4471, 0.4532, 0.4534, 0.4534,
    0.4535, 0.4534, 0.4534, 0.4533, 0.4532, 0.4531, 0.4530, 0.4530,
    0.4524, 0.4515, 0.4514, 0.4513, 0.4512, 0.4509, 0.4507, 0.4504,
    0.4501, 0.4497, 0.4493, 0.4488, 0.4484, 0.4479, 0.4474, 0.4419,
    0.4305, 0.4194, 0.4088, 0.3985, 0.3885, 0.3789, 0.2980, 0.2386,
    0.1940, 0.1598, 0.1333, 0.1123, 0.0954, 0.0818, 0.0707, 0.0615,
    0.0538, 0.0474, 0.0419, 0.0000, 0.0000
])

CT_CURVE = np.array([
    0.0000, 1.1320, 0.9995, 0.9177, 0.8608, 0.8154, 0.8116, 0.8079,
    0.8044, 0.8010, 0.7977, 0.7945, 0.7915, 0.7886, 0.7872, 0.7871,
    0.7858, 0.7838, 0.7836, 0.7833, 0.7812, 0.7773, 0.7735, 0.7697,
    0.7660, 0.7623, 0.7588, 0.7552, 0.7518, 0.7484, 0.7451, 0.7178,
    0.6722, 0.6383, 0.6102, 0.5855, 0.5632, 0.5429, 0.3993, 0.3105,
    0.2486, 0.2035, 0.1696, 0.1435, 0.1229, 0.1065, 0.0930, 0.0816,
    0.0722, 0.0644, 0.0578, 0.0000, 0.0000
])

def get_cp(u):
    """Interpolate Cp at wind speed u."""
    return float(np.interp(u, WIND_SPEEDS, CP_CURVE))

def get_ct(u):
    """Interpolate Ct at wind speed u."""
    return float(np.interp(u, WIND_SPEEDS, CT_CURVE))

def power_from_speed(u, rho=1.225, D=126.0):
    """Compute turbine power from effective inflow speed using real Cp curve."""
    cp = get_cp(u)
    A  = np.pi * (D/2)**2
    P  = 0.5 * rho * A * cp * u**3
    return max(0.0, P)

def power_MW(u):
    return power_from_speed(u) / 1e6

# =============================================================================
# LAYOUT — 3 TURBINE ALIGNED ROW, 7D SPACING
# Standard layout from Niayifar & Porte-Agel (2016)
# =============================================================================

D          = 126.0
HUB_HEIGHT = 90.0
RHO        = 1.225
TI         = 0.08

turbines_3t = [
    (0.0,        HUB_HEIGHT, 0.0),
    (0.0,        HUB_HEIGHT, 7*D),
    (0.0,        HUB_HEIGHT, 14*D),
]

# Wind speeds to test
wind_speeds = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

# =============================================================================
# MODEL WITH REAL Cp/Ct — WRAPPER FUNCTION
# =============================================================================

def run_model_real_curves(model_type, U_inf):
    """
    Run engineering model with real Cp/Ct at each turbine position.
    Returns list of powers in MW for each turbine.
    """
    ct_local = get_ct(U_inf)
    wv = [0.0, U_inf]

    # Initialise models with real Ct at this wind speed
    if model_type == 'jensen':
        model = JensenWakeModel(
            diameter=D, ct=ct_local, air_density=RHO, cp=get_cp(U_inf),
            cut_in=3.0, cut_out=25.0, kw=0.03, ambient_ti=TI
        )
        _, vels, _ = model.wind_speeds_full(turbines_3t, U_inf, wv)

    elif model_type == 'gaussian':
        model = GaussianWakeModel(
            diameter=D, ct=ct_local, air_density=RHO, cp=get_cp(U_inf),
            cut_in=3.0, cut_out=25.0, ambient_ti=TI, eps=0.22
        )
        _, vels, _ = model.wind_speeds_full(turbines_3t, U_inf, wv)

    elif model_type == 'larsen':
        model = LarsenWakeModel(
            diameter=D, ct=ct_local, air_density=RHO, cp=get_cp(U_inf),
            ambient_ti=TI, cut_in_speed=3.0, cut_out_speed=25.0
        )
        _, vels, _ = model.wind_speeds_full(turbines_3t, wv)

    # Compute power using real Cp at local velocity
    powers = [power_MW(v) for v in vels]
    return powers

# =============================================================================
# RUN FLORIS WIND SPEED SWEEP
# =============================================================================

def run_floris_speed(U_inf, velocity_model='gauss'):
    """Run FLORIS at a single wind speed with the requested velocity deficit
    model. 'jensen' requires the jimenez deflection model (same pairing used
    in gen_all_figures.py and models/extended_models.py run_floris())."""
    try:
        import floris
        floris_path = os.path.dirname(floris.__file__)
        with open(floris_path + '/default_inputs.yaml') as f:
            config = yaml.safe_load(f)
        config['farm']['layout_x'] = [t[2] for t in turbines_3t]
        config['farm']['layout_y'] = [t[0] for t in turbines_3t]
        config['farm']['turbine_type'] = ['nrel_5MW']
        config['flow_field']['wind_speeds'] = [U_inf]
        config['flow_field']['wind_directions'] = [270.0]
        config['flow_field']['turbulence_intensities'] = [TI]
        config['flow_field']['air_density'] = RHO
        config['wake']['model_strings']['velocity_model'] = velocity_model
        if velocity_model == 'jensen':
            config['wake']['model_strings']['deflection_model'] = 'jimenez'
        tmp_yaml = tmp_path(f'floris_speed_{velocity_model}.yaml')
        with open(tmp_yaml, 'w') as f:
            yaml.dump(config, f)
        from floris import FlorisModel
        fm = FlorisModel(tmp_yaml)
        fm.run()
        return [p/1e6 for p in fm.get_turbine_powers().flatten()]
    except Exception as e:
        print(f"  FLORIS ({velocity_model}) error at {U_inf} m/s: {e}")
        return None

# =============================================================================
# CFD REFERENCE DATA
# =============================================================================

cfd_data = {
    6:  {'T1': 0.943,  'T2': 0.676,  'T3': 0.563},
    8:  {'T1': 2.215,  'T2': 1.590,  'T3': 1.314},
    10: {'T1': 3.806,  'T2': 1.904,  'T3': None},
    12: {'T1': 7.243,  'T2': 5.231,  'T3': 4.325},
    14: {'T1': 11.216, 'T2': 8.145,  'T3': 6.742},
}
cfd_speeds = sorted(cfd_data.keys())

# =============================================================================
# RUN SWEEP
# =============================================================================

results = {
    'LOTUSim-Jensen':   {'T1': [], 'T2': [], 'T3': []},
    'LOTUSim-Gaussian': {'T1': [], 'T2': [], 'T3': []},
    'LOTUSim-Larsen':   {'T1': [], 'T2': [], 'T3': []},
    'LOTUSim-Blended':  {'T1': [], 'T2': [], 'T3': []},
    'FLORIS':            {'T1': [], 'T2': [], 'T3': []},  # velocity_model = gauss
    'FLORIS-Jensen':     {'T1': [], 'T2': [], 'T3': []},
}

print("Running wind speed sweep with real Cp/Ct curves...")
print(f"{'U (m/s)':>9} {'Cp':>8} {'Ct':>8}")
print("-"*28)
for U in wind_speeds:
    print(f"{U:>9.1f} {get_cp(U):>8.4f} {get_ct(U):>8.4f}")

print("\nRunning models...")
for U in wind_speeds:
    for name, mtype in [('LOTUSim-Jensen','jensen'), ('LOTUSim-Gaussian','gaussian'), ('LOTUSim-Larsen','larsen')]:
        p = run_model_real_curves(mtype, U)
        results[name]['T1'].append(p[0])
        results[name]['T2'].append(p[1])
        results[name]['T3'].append(p[2])
    # LOTUSim-Blended power
    from models.extended_models import BlendedWakeModel
    _ct  = get_ct(U)
    _cp  = get_cp(U)
    _bm  = BlendedWakeModel(diameter=D, ct=_ct, air_density=RHO,
                            cp=_cp, ambient_ti=TI,
                            cut_in_speed=3.0, cut_out_speed=25.0)
    _ups = []
    _bpow = []
    for _xt, _yt, _zt in turbines_3t:
        _uhub = U if not _ups else _bm.farm_velocity_at_point(U, _zt, _xt, _ups)
        _bpow.append(power_MW(_uhub))
        _ups.append((_xt, _yt, _zt))
    results["LOTUSim-Blended"]["T1"].append(_bpow[0])
    results["LOTUSim-Blended"]["T2"].append(_bpow[1])
    results["LOTUSim-Blended"]["T3"].append(_bpow[2])

    for fl_key, vmodel in [('FLORIS', 'gauss'), ('FLORIS-Jensen', 'jensen')]:
        fl = run_floris_speed(U, velocity_model=vmodel)
        if fl:
            results[fl_key]['T1'].append(fl[0])
            results[fl_key]['T2'].append(fl[1])
            results[fl_key]['T3'].append(fl[2])
        else:
            results[fl_key]['T1'].append(0)
            results[fl_key]['T2'].append(0)
            results[fl_key]['T3'].append(0)

    print(f"  {U} m/s done")

# =============================================================================
# PRINT RESULTS TABLE — ABSOLUTE POWER
# =============================================================================

print("\n" + "="*80)
print("WIND SPEED SWEEP — Real Cp/Ct — NREL 5MW, 3 Turbines, 7D, Aligned")
print("="*80)

for turb, label in [('T1','T1 (upstream)'), ('T2','T2 (7D)'), ('T3','T3 (14D)')]:
    print(f"\n{label}:")
    print(f"{'U (m/s)':>9} {'LOTUSim-J':>10} {'LOTUSim-G':>10} {'LOTUSim-L':>10} "
          f"{'FLORIS':>10} {'CFD':>10}")
    print("-"*62)
    for i, U in enumerate(wind_speeds):
        cfd_val = cfd_data[U][turb] if U in cfd_data and cfd_data[U][turb] is not None else None
        cfd_str = f"{cfd_val:>10.3f}" if cfd_val is not None else f"{chr(8212):>10}"
        print(f"{U:>9.1f} "
              f"{results['LOTUSim-Jensen'][turb][i]:>10.3f} "
              f"{results['LOTUSim-Gaussian'][turb][i]:>10.3f} "
              f"{results['LOTUSim-Larsen'][turb][i]:>10.3f} "
              f"{results['FLORIS'][turb][i]:>10.3f} "
              f"{cfd_str}")

# =============================================================================
# PRINT RESULTS TABLE — NORMALISED POWER
# =============================================================================

print("\n" + "="*80)
print("NORMALISED POWER (P/P_T1) — Standard literature format")
print("="*80)

for turb, label in [('T2','T2/T1 (7D wake)'), ('T3','T3/T1 (14D wake)')]:
    print(f"\n{label}:")
    print(f"{'U (m/s)':>9} {'LOTUSim-J':>10} {'LOTUSim-G':>10} {'LOTUSim-L':>10} "
          f"{'FLORIS':>10} {'CFD':>10}")
    print("-"*62)
    for i, U in enumerate(wind_speeds):
        t1_j = results['LOTUSim-Jensen']['T1'][i]
        t1_g = results['LOTUSim-Gaussian']['T1'][i]
        t1_l = results['LOTUSim-Larsen']['T1'][i]
        t1_f = results['FLORIS']['T1'][i]
        j_norm = results['LOTUSim-Jensen'][turb][i]  / t1_j if t1_j > 0 else 0
        g_norm = results['LOTUSim-Gaussian'][turb][i] / t1_g if t1_g > 0 else 0
        l_norm = results['LOTUSim-Larsen'][turb][i]  / t1_l if t1_l > 0 else 0
        f_norm = results['FLORIS'][turb][i]  / t1_f if t1_f > 0 else 0
        if U in cfd_data:
            c_norm = cfd_data[U][turb] / cfd_data[U]["T1"] if cfd_data[U][turb] is not None else None
            cfd_str = f"{c_norm:>10.3f}" if c_norm is not None else f"{chr(8212):>10}"
        else:
            cfd_str = f"{'—':>10}"
        print(f"{U:>9.1f} {j_norm:>10.3f} {g_norm:>10.3f} "
              f"{l_norm:>10.3f} {f_norm:>10.3f} {cfd_str}")

# =============================================================================
# RMSE vs CFD (normalised, at CFD wind speeds only)
# =============================================================================

RATED_SPEED = 11.4  # NREL 5MW rated wind speed

MODEL_ORDER = ['LOTUSim-Jensen', 'LOTUSim-Gaussian', 'LOTUSim-Larsen',
               'LOTUSim-Blended', 'FLORIS', 'FLORIS-Jensen']

def normalised_rmse(name, below_rated_only=False):
    """Normalised P/P_T1 RMSE vs CFD over T2 and T3. If below_rated_only,
    above-rated CFD speeds (U > 11.4 m/s) are excluded — matching the
    exclusion the eval.tex prose describes. The T3@10 m/s CFD point is
    always excluded (it is None in cfd_data: actuator-line breakdown)."""
    errors = []
    for U in cfd_speeds:
        if below_rated_only and U > RATED_SPEED:
            continue
        i = wind_speeds.index(U) if U in wind_speeds else None
        if i is None:
            continue
        t1_m = results[name]['T1'][i]
        for turb in ['T2', 'T3']:
            if t1_m > 0 and cfd_data[U][turb] is not None:
                m_norm = results[name][turb][i] / t1_m
                c_norm = cfd_data[U][turb] / cfd_data[U]["T1"]
                errors.append((m_norm - c_norm) ** 2)
    return (sum(errors) / len(errors)) ** 0.5 if errors else None

print("\n--- RMSE vs CFD (normalised P/P_T1, T2 and T3) ---")
print(f"{'Model':<20}{'RMSE (all)':>14}{'RMSE (<=rated)':>16}")
print("-" * 50)
for name in MODEL_ORDER:
    r_all = normalised_rmse(name, below_rated_only=False)
    r_br  = normalised_rmse(name, below_rated_only=True)
    print(f"{name:<20}{r_all:>14.4f}{r_br:>16.4f}")

# =============================================================================
# FIGURE 1 — Absolute power vs wind speed (3 panels)
# =============================================================================

fig, axes = plt.subplots(1, 3, figsize=(18, 7))
fig.patch.set_facecolor('white')

# Colour/marker conventions match gen_all_figures.py STYLES so every
# figure uses the same visual identity for each model.
colours = {
    'LOTUSim-Jensen':   ('tomato',         'v', '--', 1.4),
    'LOTUSim-Gaussian': ('#DAA000',        'D', '--', 1.4),
    'LOTUSim-Larsen':   ('mediumseagreen', 'P', '--', 1.4),
    'LOTUSim-Blended':  ('darkviolet',     'h', '-',  1.6),
    'FLORIS':           ('royalblue',      's', '-',  1.4),
    'FLORIS-Jensen':    ('cornflowerblue', '^', '-',  1.4),
}

# Legend labels — distinguish the two FLORIS velocity models.
DISPLAY = {'FLORIS': 'FLORIS Gaussian', 'FLORIS-Jensen': 'FLORIS Jensen'}

for ax, turb, title in [
    (axes[0], 'T1', 'T1 — Upstream (no wake)'),
    (axes[1], 'T2', 'T2 — 7D downstream'),
    (axes[2], 'T3', 'T3 — 14D downstream'),
]:
    ax.set_facecolor('#FAFAFA')

    # Plot model lines
    for name, (col, marker, ls, lw) in colours.items():
        ax.plot(wind_speeds, results[name][turb],
                marker=marker, color=col, linewidth=lw+0.5,
                linestyle=ls, markersize=11, label=DISPLAY.get(name, name))

    cfd_vals = [cfd_data[U][turb] for U in cfd_speeds if cfd_data[U][turb] is not None]
    cfd_val_speeds = [U for U in cfd_speeds if cfd_data[U][turb] is not None]
    ax.scatter(cfd_val_speeds, cfd_vals, color="black", marker="*",
               s=200, zorder=6, label='CFD reference')

    # Rated power line
    ax.axhline(5.0, color='grey', linestyle=':', linewidth=1.0,
               alpha=0.6, label='Rated (5 MW)')

    ax.set_xlabel('Wind speed (m/s)', fontsize=15)
    ax.set_ylabel('Power (MW)', fontsize=15)
    ax.set_title(title, fontsize=15, fontweight='bold')
    ax.legend(fontsize=12, framealpha=0.95)
    ax.grid(True, alpha=0.25)
    ax.set_xticks(wind_speeds)

fig.suptitle(
    'Protocol P2-2 — Wind Speed Sweep\n'
    'NREL 5MW, Layout A, Real Cp/Ct curves vs CFD',
    fontsize=16, fontweight='bold'
)
fig.tight_layout(rect=[0, 0, 1, 0.94])
os.makedirs(FIG_DIR, exist_ok=True)
_out1 = os.path.join(FIG_DIR, 'wind_speed_sweep.png')
fig.savefig(_out1, dpi=600, bbox_inches='tight')
print("\nFigure 1 saved to:", _out1)

# =============================================================================
# FIGURE 2 — Normalised power P/P_T1 vs wind speed (literature format)
# =============================================================================

fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
fig2.patch.set_facecolor('white')

for ax, turb, title in [
    (ax1, 'T2', 'T2/T1 — 7D downstream'),
    (ax2, 'T3', 'T3/T1 — 14D downstream'),
]:
    ax.set_facecolor('#FAFAFA')

    # Plot model lines
    for name, (col, marker, ls, lw) in colours.items():
        t1_vals  = results[name]['T1']
        norm_vals = [
            results[name][turb][i] / t1_vals[i] if t1_vals[i] > 0 else 0
            for i in range(len(wind_speeds))
        ]
        ax.plot(wind_speeds, norm_vals,
                marker=marker, color=col, linewidth=lw+0.5,
                linestyle=ls, markersize=11, label=DISPLAY.get(name, name))

    # Plot CFD normalised reference points — skip None values
    cfd_norm_speeds = [U for U in cfd_speeds if cfd_data[U][turb] is not None]
    cfd_norm = [cfd_data[U][turb] / cfd_data[U]["T1"] for U in cfd_norm_speeds]
    ax.scatter(cfd_norm_speeds, cfd_norm, color="black", marker="*",
               s=200, zorder=6, label='CFD reference')

    # Rated wind speed marker
    ax.axvline(11.4, color='grey', linestyle=':', linewidth=1.0,
               alpha=0.6, label='Rated speed (11.4 m/s)')

    ax.set_xlabel('Wind speed (m/s)', fontsize=15)
    ax.set_ylabel('P / P_T1', fontsize=15)
    ax.set_title(title, fontsize=15, fontweight='bold')
    ax.legend(fontsize=12, framealpha=0.95)
    ax.grid(True, alpha=0.25)
    ax.set_xticks(wind_speeds)
    ax.set_ylim(0, 1.1)
    ax.axhline(1.0, color='k', linestyle=':', linewidth=0.8, alpha=0.4)

fig2.suptitle(
    'Normalised Wake Power P/P_T1 — NREL 5MW, 3 Turbines, 7D Spacing\n'
    'Real Cp/Ct curves vs CFD — Standard literature format\n'
    '(Niayifar & Porte-Agel 2016, Bastankhah & Porte-Agel 2014)',
    fontsize=16, fontweight='bold'
)
fig2.tight_layout(rect=[0, 0, 1, 0.92])
os.makedirs(SUPP_DIR, exist_ok=True)
_out2 = os.path.join(SUPP_DIR, 'wind_speed_sweep_normalised_allmodels.png')
fig2.savefig(_out2, dpi=600, bbox_inches='tight')
print("Figure 2 saved to:", _out2)
if __name__ == "__main__":
    plt.show()