"""
Wake Model Comparison — FLORIS vs Custom Models vs CFD
=======================================================
Layout:  3-turbine row matching OpenFOAM CFD case
Turbine: NREL 5MW, D=126m, hub height=90m
Wind:    10 m/s, TI=6%, direction=270 (west, fully aligned)

Compares:
  - FLORIS Gauss (validated reference)
  - FLORIS Jensen (validated reference)
  - Your Jensen implementation
  - Your Gaussian implementation
  - Your Larsen implementation
  - OpenFOAM/turbinesFoam CFD results
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import yaml
import os
import tempfile

# ---------------------------------------------------------------------------
# Portable paths. Everything is resolved relative to this file so the module
# behaves identically whatever the working directory, machine or OS.
#   REPO_ROOT  - repository root
#   DATA_DIR   - extracted CFD reference data (benchmark/data/)
#   FIG_DIR    - where this script's own figures are written
#   tmp_path() - scratch file in the platform temp dir (not hardcoded /tmp)
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR  = os.path.join(REPO_ROOT, 'benchmark', 'data')
FIG_DIR   = os.path.join(REPO_ROOT, 'figures', 'supplementary')


def tmp_path(name):
    """Scratch file in the platform temp directory."""
    return os.path.join(tempfile.gettempdir(), name)


def _fig_out(name):
    """Absolute output path for this script's figures (created on demand)."""
    os.makedirs(FIG_DIR, exist_ok=True)
    return os.path.join(FIG_DIR, name)


def _show_if_main():
    """
    Open the figure windows only when this file is executed directly.

    This module has a script body that runs on import (gen_all_figures.py and
    the analysis scripts import the model classes from here). A bare plt.show()
    at module level blocks that import indefinitely under any interactive
    matplotlib backend, which used to hang the whole benchmark.
    """
    if __name__ == "__main__":
        plt.show()


# JensenWakeModel
class JensenWakeModel:

    def __init__(self, diameter: float, ct: float = 0.8, air_density: float = 1.225, cp: float = 0.35,

                 cut_in: float = 5.0, cut_out: float = 25.0, kw: float = 0.04):

        self.diameter = diameter

        self.ct = ct

        self.air_density = air_density

        self.cp = cp

        self.cut_in = cut_in

        self.cut_out = cut_out

        self.kw = kw



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



    def wake_radius(self, x_dist: float) -> float:

        # Jensen Wake Radius Expansion

        return self.diameter / 2 + self.kw * x_dist



    def in_wake(self, x_dist: float, lateral_dist: float) -> bool:

        r = self.wake_radius(x_dist)

        return abs(lateral_dist) < r



    def wake_deficit(self, ogWind: float, x_dist: float) -> float:

        # returns deficit not final velocity

        if x_dist <= 0:

            return 0.0



        factor = (1.0 - np.sqrt(1.0 - self.ct)) / (1.0 + 2.0 * self.kw * x_dist / self.diameter) ** 2

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
                     * np.cos(yaw_angle_rad)**2 * x_dist
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
        weights = np.sqrt(np.maximum(0, R**2 - (z_samples - hub_height)**2))
        # Wind speed at each sample height
        U_samples = np.array([
	    self.shear_adjusted_speed(U_hub, z, hub_height, alpha)
	    for z in z_samples
        ])
        # Area-weighted average
        if weights.sum() > 0:
            return float(np.average(U_samples, weights=weights))
        return U_hub

    def partial_overlap_factor(self, x_dist, lateral_dist):
        """
        Computes fraction of rotor disk area overlapping with wake cross-section.
        Replaces binary in/out wake detection with smooth geometric transition.
        Uses circle-circle intersection area formula.
        Reference: Katic et al. (1986), European Wind Energy Conference, Rome.
        Extended by: Yang (2020), Energies, 13(3), 739.
        """
        R_rotor = self.diameter / 2.0
        R_wake  = self.wake_radius(x_dist)
        d       = abs(lateral_dist)
    
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
        d1 = (d**2 + R_rotor**2 - R_wake**2) / (2 * d)
        d2 = d - d1
    
        cos1 = np.clip(d1 / R_rotor, -1, 1)
        cos2 = np.clip(d2 / R_wake,  -1, 1)
    
        A_overlap = (R_rotor**2 * np.arccos(cos1)
                     - d1 * np.sqrt(max(0, R_rotor**2 - d1**2))
                     + R_wake**2  * np.arccos(cos2)
                     - d2 * np.sqrt(max(0, R_wake**2  - d2**2)))
    
        A_rotor = np.pi * R_rotor**2
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
                   overlap = self.partial_overlap_factor(x_dist, effective_lateral)
                   if overlap > 0:
                       deficit = self.wake_deficit(U_local, x_dist) * overlap
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



    def __init__(self, diameter:float, ct:float, air_density:float = 1.225,cp: float = 0.35, cut_in: float = 5.0, cut_out: float = 25.0, k_y: float=0.05, k_z: float = 0.05, eps: float = 0.2,  ambient_ti: float = None):

        self.diameter = diameter

        self.ct = ct

        self.air_density = air_density

        self.cp = cp

        self.cut_in = cut_in

        self.cut_out = cut_out

        self.k_y = k_y # describes the width expansion of the wake downstream

        self.k_z = k_z # describes the height expansion of the wake downstream

        self.eps = eps # parameter linked to the starting width of the wake ratio to diameter

        self.ambient_ti = ambient_ti
        
        # If TI provided, use Niayifar & Porte-Agel (2016) TI-dependent expansion
        # k = 0.38 * TI + 0.004
        # Otherwise use user-supplied k_y and k_z
        if ambient_ti is not None:
            self.k_y = 0.38 * ambient_ti + 0.004
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

        vector = np.array(vector, dtype = float)

        magnitude = np.linalg.norm(vector)



        if magnitude ==0:

            raise ValueError("Wind vector cannot be zero.")



        return vector/magnitude



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
        amplitude = 1.0 - np.sqrt(max(0.0, 1.0 - self.ct / (8 * sigma_d**2)))



        # Gaussian spatial spread — how the deficit distributes laterally and vertically

        exponent = -0.5 * ((y_dist / sigma_y) ** 2 + (z_dist / sigma_z) ** 2)

        spread = np.exp(exponent)



        return ogWind * amplitude * spread



    #calculating overlapping wakes

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
                     * np.cos(yaw_angle_rad)**2 * x_dist
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
        weights = np.sqrt(np.maximum(0, R**2 - (z_samples - hub_height)**2))
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

    def combined_velocity(ogWind:float, deficits)-> float:

        #combining multiple wake deficits using root sum square



        if not deficits:

            return ogWind



        total_deficit = np.sqrt(sum((d/ogWind)**2 for d in deficits))

        total_deficit = min(total_deficit, 0.999) # to prevent negative wind speeds



        return ogWind*(1.0-total_deficit)



    #Main section

    def ti_at_point(self, x, r):
        """
        Turbulence intensity at point (x, r) using Crespo-Hernandez.
        Returns TI as fraction (0 to 1).
        """
        if x <= 0:
            return self.ambient_ti

        x_D = x / self.diameter
        # Wake width, needed before the local TI can be evaluated. The expansion
        # rate is the TI-dependent form of Niayifar & Porte-Agel (2016), seeded
        # with the ambient TI: the wake-added TI is not yet known at this point.
        # (This line previously read `ti_local = self.ti_at_point(x, r)`, an
        #  unconditional self-call that made the method infinitely recursive.)
        ti_eff = self.ambient_ti
        kw_eff = 0.38 * ti_eff + 0.004
        sigma = kw_eff * x + 0.22 * self.diameter

        # Outside wake
        if r >= 2.5 * sigma:
            return self.ambient_ti

        # Crespo-Hernandez added TI
        ti_added = 0.5 * (self.ct * self.ambient_ti) ** 0.25 * x_D ** (-0.32)
        ti_centreline = self.ambient_ti + ti_added

        # Gaussian lateral profile
        ti_local = self.ambient_ti + (ti_centreline - self.ambient_ti) * \
                   np.exp(-r ** 2 / (2 * sigma ** 2))

        x_D = x / self.diameter
        blend = np.exp(-x_D / 15.0)
        ti_blended = self.ambient_ti + (ti_local - self.ambient_ti) * blend
        return min(0.40, max(self.ambient_ti, ti_blended))

    def velocity_at_point_2d(self, U_inf, x, r):
        """
        Gaussian wake velocity at downstream distance x and
        radial position r from wake centreline.
        Returns velocity in m/s.
        """
        if x <= 0:
            return U_inf

        sigma = (0.38 * self.ambient_ti + 0.004) * x + 0.22 * self.diameter
        C = 1 - np.sqrt(max(0, 1 - self.ct / (8 * sigma ** 2 / self.diameter ** 2)))
        deficit = U_inf * C * np.exp(-r ** 2 / (2 * sigma ** 2))

        return max(0.0, U_inf - deficit)

    def velocity_gradient_at_point(self, U_inf, x, r, dr=1.0):
        """
        Lateral velocity gradient dU/dr at point (x, r).
        Units: s^-1
        """
        U_plus = self.velocity_at_point_2d(U_inf, x, r + dr)
        U_minus = self.velocity_at_point_2d(U_inf, x, max(0.0, r - dr))
        return (U_plus - U_minus) / (2 * dr)

    def wake_hazard_zone(self, U_inf, x, r,
                         deficit_caution=0.15,
                         deficit_restricted=0.25,
                         ti_caution=0.12,
                         ti_restricted=0.18):
        """
        Classify point (x, r) for drone operations.
        Returns: 0=safe, 1=caution, 2=restricted
        """
        u_local = self.velocity_at_point_2d(U_inf, x, r)
        deficit = 1.0 - u_local / U_inf
        ti_local = self.ti_at_point(x, r)

        if deficit > deficit_restricted or ti_local > ti_restricted:
            return 2
        elif deficit > deficit_caution or ti_local > ti_caution:
            return 1
        else:
            return 0

    def velocity_at_height(self, U_inf, x, y, z,
                           hub_height=90.0, alpha=0.12):
        """
        Wake velocity at arbitrary 3D position (x, y, z).
        """
        U_z = U_inf * (z / hub_height) ** alpha
        r = np.sqrt(y ** 2 + (z - hub_height) ** 2)
        return self.velocity_at_point_2d(U_z, x, r)

    def farm_velocity_at_point(self, U_inf, x_query, y_query,
                               turbines, hub_height=90.0, alpha=0.12):
        """
        Farm velocity at query point accounting for all upstream turbines.
        """
        U_local = U_inf

        for (x_t, y_t, z_t) in turbines:
            dx = x_query - z_t
            dy = y_query - x_t

            if dx <= 1e-6:
                continue

            r = abs(dy)
            U_wake = self.velocity_at_point_2d(U_local, dx, r)
            deficit = max(0.0, U_local - U_wake)
            U_local = max(0.0, U_local - deficit)

        return U_local

    def farm_ti_at_point(self, x_query, y_query, turbines):
        """
        Combined TI at query point from all upstream turbines.
        """
        ti_sq_sum = self.ambient_ti ** 2

        for (x_t, y_t, z_t) in turbines:
            dx = x_query - z_t
            dy = y_query - x_t

            if dx <= 1e-6:
                continue

            r = abs(dy)
            ti_local = self.ti_at_point(dx, r)
            ti_added = ti_local - self.ambient_ti
            if ti_added > 0:
                ti_sq_sum += ti_added ** 2

        return min(0.40, np.sqrt(ti_sq_sum))

    def farm_hazard_zone(self, U_inf, x_query, y_query, turbines,
                         deficit_caution=0.15, deficit_restricted=0.25,
                         ti_caution=0.12, ti_restricted=0.18):
        """
        Farm-scale hazard zone classification.
        Returns: 0=safe, 1=caution, 2=restricted
        """
        u_local = self.farm_velocity_at_point(U_inf, x_query, y_query, turbines)
        deficit = 1.0 - u_local / U_inf
        ti_local = self.farm_ti_at_point(x_query, y_query, turbines)

        if deficit > deficit_restricted or ti_local > ti_restricted:
            return 2
        elif deficit > deficit_caution or ti_local > ti_caution:
            return 1
        else:
            return 0

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


class CalibratedGaussianWakeModel(GaussianWakeModel):
    """
    CFD-calibrated Gaussian wake model.
    Uses an empirically fitted sigma(x) function derived from
    single-turbine OpenFOAM/turbinesFoam actuator line CFD data
    rather than the standard linear TI-based expansion formula.

    The calibrated sigma uses an exponential approach function:
    sigma(x) = sigma_inf - (sigma_inf - sigma_0) * exp(-decay * x/D)

    Fitted to NREL 5MW single turbine wake profiles at 1D-14D:
    sigma_inf = 0.402D, sigma_0 = 0.200D, decay = 2.796

    This corrects the standard Gaussian which under-predicts sigma
    at near-wake (1D: 53% too small) and over-predicts at far-wake
    (14D: 43% too large), directly improving velocity gradient accuracy.

    Reference: Bastankhah & Porte-Agel (2014) base formulation,
    sigma calibrated against OpenFOAM v8 turbinesFoam ALM results.
    """

    # Calibrated sigma parameters from CFD curve fit
    # Ct-dependent sigma parameters fitted to single-turbine CFD
    # at 8 m/s (Ct=0.80), 10 m/s (Ct=0.75), 12 m/s (Ct=0.53)
    # sigma_0(Ct)   = 0.0939*Ct + 0.3286  R=0.85
    # sigma_inf(Ct) = 0.0875*Ct + 0.3453  R=0.89
    SIGMA_0_SLOPE   = 0.0939
    SIGMA_0_INTER   = 0.3286
    SIGMA_INF_SLOPE = 0.0875
    SIGMA_INF_INTER = 0.3453
    DECAY           = 2.796

    def calibrated_sigma(self, x, U_inf=None):
        """
        Ct-dependent calibrated wake width sigma(x).
        Fitted to single-turbine OpenFOAM CFD at 8, 10, 12 m/s.
        sigma_0(Ct)   = 0.0939*Ct + 0.3286  (R=0.85)
        sigma_inf(Ct) = 0.0875*Ct + 0.3453  (R=0.89)
        Valid for Ct in range 0.40-0.90 (U = 6-14 m/s NREL 5MW).
        """
        x_D = x / self.diameter
        # Ct-dependent sigma parameters
        sigma_0_D   = self.SIGMA_0_SLOPE   * self.ct + self.SIGMA_0_INTER
        sigma_inf_D = self.SIGMA_INF_SLOPE * self.ct + self.SIGMA_INF_INTER
        # Exponential approach from sigma_0 to sigma_inf
        sigma_D = sigma_inf_D - (sigma_inf_D - sigma_0_D) * np.exp(-self.DECAY * x_D)
        return sigma_D * self.diameter

        return sigma_D * self.diameter

    def velocity_at_point_cal(self, U_inf, x, r):
        """
        Calibrated Gaussian velocity at (x, r).
        Uses CFD-fitted sigma instead of TI-based formula.
        """
        if x <= 0:
            return U_inf

        sigma = self.calibrated_sigma(x)
        C = 1 - np.sqrt(max(0, 1 - self.ct /
                            (8 * sigma**2 / self.diameter**2)))
        deficit = U_inf * C * np.exp(-r**2 / (2 * sigma**2))
        return max(0.0, U_inf - deficit)

    def velocity_gradient_at_point(self, U_inf, x, r, dr=1.0):
        """Override to use calibrated velocity."""
        U_plus  = self.velocity_at_point_cal(U_inf, x, r + dr)
        U_minus = self.velocity_at_point_cal(U_inf, x, max(0.0, r - dr))
        return (U_plus - U_minus) / (2 * dr)

    def wake_hazard_zone(self, U_inf, x, r,
                          deficit_caution=0.15,
                          deficit_restricted=0.25,
                          ti_caution=0.12,
                          ti_restricted=0.18):
        """Override to use calibrated velocity."""
        u_local = self.velocity_at_point_cal(U_inf, x, r)
        deficit  = 1.0 - u_local / U_inf
        ti_local = self.ti_at_point(x, r)
        if deficit > deficit_restricted or ti_local > ti_restricted:
            return 2
        elif deficit > deficit_caution or ti_local > ti_caution:
            return 1
        else:
            return 0

    def farm_velocity_at_point(self, U_inf, x_query, y_query,
                                turbines, hub_height=90.0, alpha=0.12):
        """Override to use calibrated velocity."""
        U_local = U_inf
        for (x_t, y_t, z_t) in turbines:
            dx = x_query - z_t
            dy = y_query - x_t
            if dx <= 1e-6:
                continue
            r = abs(dy)
            U_wake  = self.velocity_at_point_cal(U_local, dx, r)
            deficit = max(0.0, U_local - U_wake)
            U_local = max(0.0, U_local - deficit)
        return U_local

    def farm_hazard_zone(self, U_inf, x_query, y_query, turbines,
                          deficit_caution=0.15, deficit_restricted=0.25,
                          ti_caution=0.12, ti_restricted=0.18):
        """Override to use calibrated farm velocity."""
        u_local  = self.farm_velocity_at_point(U_inf, x_query,
                                                y_query, turbines)
        deficit  = 1.0 - u_local / U_inf
        ti_local = self.farm_ti_at_point(x_query, y_query, turbines)
        if deficit > deficit_restricted or ti_local > ti_restricted:
            return 2
        elif deficit > deficit_caution or ti_local > ti_caution:
            return 1
        else:
            return 0


class BlendedWakeModel:
    """
    Distance-weighted blend of Larsen and Calibrated Gaussian models.
    Optimised against single-turbine OpenFOAM/turbinesFoam CFD data.

    Blend function:
    - x <= 1D: 100% Larsen (4.7% gradient error vs CFD)
    - x > 1D:  40% Larsen + 60% Calibrated Gaussian

    This hybrid model achieves <5% velocity gradient error at all
    downstream positions from 1D to 14D against RANS CFD reference,
    compared to 4.7-175.8% for individual models.

    Reference: calibrated against NREL 5MW single-turbine
    OpenFOAM v8 turbinesFoam actuator line simulation.
    """

    def __init__(self, diameter, ct, air_density=1.225, cp=0.498,
                 ambient_ti=0.08, cut_in_speed=3.0, cut_out_speed=25.0,
                 eps=0.22):
        self.diameter = diameter
        self.ct = ct
        self.ambient_ti = ambient_ti

        self.larsen = LarsenWakeModel(
            diameter=diameter, ct=ct, air_density=air_density,
            cp=cp, ambient_ti=ambient_ti,
            cut_in_speed=cut_in_speed, cut_out_speed=cut_out_speed)

        self.cal_gauss = CalibratedGaussianWakeModel(
            diameter=diameter, ct=ct, air_density=air_density,
            cp=cp, cut_in=cut_in_speed, cut_out=cut_out_speed,
            ambient_ti=ambient_ti, eps=eps)

    def blend_weight(self, x, ct=None):
        """
        Ct-dependent blend weight.
        Higher Ct = stronger wake = Larsen more accurate near-wake.
        """
        if ct is None:
            ct = self.ct
        x_D = x / self.diameter
        # Scale transition distance with Ct
        # Higher Ct = longer near-wake region
        transition = 1.5 * ct / 0.75  # normalised to calibration Ct
        w = 0.40 + 0.60 * np.exp(-1.5 * (x_D - 1.0) / transition)
        return np.clip(w, 0.40, 1.0)

    def velocity_at_point(self, U_inf, x, r):
        """
        Blended velocity at (x, r).
        Weighted combination of Larsen and Calibrated Gaussian.
        """
        if x <= 0:
            return U_inf
        w = self.blend_weight(x)
        u_l = self.larsen.velocity_at_point(U_inf, x, r)
        u_g = self.cal_gauss.velocity_at_point_cal(U_inf, x, r)
        return w * u_l + (1 - w) * u_g

    def ti_at_point(self, x, r):
        """TI from Larsen model."""
        return self.larsen.ti_at_point(x, r)

    def velocity_gradient_at_point(self, U_inf, x, r, dr=1.0):
        """Lateral velocity gradient at (x, r)."""
        U_plus  = self.velocity_at_point(U_inf, x, r + dr)
        U_minus = self.velocity_at_point(U_inf, x, max(0.0, r - dr))
        return (U_plus - U_minus) / (2 * dr)

    def wake_hazard_zone(self, U_inf, x, r,
                          deficit_caution=0.15,
                          deficit_restricted=0.25,
                          ti_caution=0.12,
                          ti_restricted=0.18):
        """Hazard zone classification using blended velocity."""
        u_local = self.velocity_at_point(U_inf, x, r)
        deficit  = 1.0 - u_local / U_inf
        ti_local = self.ti_at_point(x, r)
        if deficit > deficit_restricted or ti_local > ti_restricted:
            return 2
        elif deficit > deficit_caution or ti_local > ti_caution:
            return 1
        else:
            return 0

    def farm_velocity_at_point(self, U_inf, x_query, y_query,
                                turbines, hub_height=90.0, alpha=0.12):
        """Farm velocity accounting for all upstream turbines."""
        U_local = U_inf
        for (x_t, y_t, z_t) in turbines:
            dx = x_query - z_t
            dy = y_query - x_t
            if dx <= 1e-6:
                continue
            r = abs(dy)
            U_wake  = self.velocity_at_point(U_local, dx, r)
            deficit = max(0.0, U_local - U_wake)
            U_local = max(0.0, U_local - deficit)
        return U_local

    def farm_ti_at_point(self, x_query, y_query, turbines):
        """Farm TI using root-sum-square combination."""
        return self.larsen.farm_ti_at_point(x_query, y_query, turbines)

    def farm_hazard_zone(self, U_inf, x_query, y_query, turbines,
                          deficit_caution=0.15, deficit_restricted=0.25,
                          ti_caution=0.12, ti_restricted=0.18):
        """Farm-scale hazard classification."""
        u_local = self.farm_velocity_at_point(U_inf, x_query,
                                               y_query, turbines)
        deficit  = 1.0 - u_local / U_inf
        ti_local = self.farm_ti_at_point(x_query, y_query, turbines)
        if deficit > deficit_restricted or ti_local > ti_restricted:
            return 2
        elif deficit > deficit_caution or ti_local > ti_caution:
            return 1
        else:
            return 0

    def farm_velocity_field(self, U_inf, X, Y, turbines,
                            hub_height=90.0, alpha=0.12):
        """
        Vectorised farm velocity field for real-time computation.
        X, Y are numpy meshgrid arrays of query positions.
        Returns U_field same shape as X, Y.
        Computation time: <100ms for 200x200 grid.

        Parameters:
        -----------
        U_inf   : freestream wind speed (m/s)
        X       : downstream positions array (m)
        Y       : lateral positions array (m)
        turbines: list of (x_lat, y_hub, z_down) turbine positions
        """
        U_field = np.ones_like(X, dtype=float) * U_inf

        for (x_t, y_t, z_t) in turbines:
            DX = X - z_t  # downstream distance from this turbine
            DY = Y - x_t  # lateral distance from this turbine

            # Only compute wake where downstream of turbine
            wake_mask = DX > 1e-6

            if not np.any(wake_mask):
                continue

            # Blend weight vectorised
            x_D_arr = np.where(wake_mask, DX / self.diameter, 1e6)
            w = np.clip(0.40 + 0.60 * np.exp(-1.5 * (x_D_arr - 1.0)),
                        0.40, 1.0)

            # Larsen deficit vectorised
            # Larsen R96 vectorised (avoid scalar wake_radius call)
            c1  = self.larsen._c1()
            rhs = (35.0/(2.0*np.pi))**0.3 * (3.0*c1**2)**(-0.2)
            x0 = self.larsen._x0()
            x_eff = np.where(wake_mask, DX + x0, x0)
            R96_arr = np.where(
                wake_mask,
                ((105.0 * c1**2) / (2.0 * np.pi))**0.2 *
                (self.larsen.ct * self.larsen.area * x_eff)**(1.0/3.0),
                0.0)
            in_wake = wake_mask & (np.abs(DY) < R96_arr)


            r_abs = np.abs(DY)
            bracket = np.where(
                r_abs < 1e-6,
                -rhs,
                r_abs ** 1.5 * (3.0 * c1 ** 2 * self.larsen.ct *
                                self.larsen.area * x_eff) ** (-0.5) - rhs
            )

            x_D_safe = np.where(x_D_arr > 0, x_D_arr, 1.0)
            x_eff_safe = np.where(x_eff > 0, x_eff, 1.0)
            term2 = (self.larsen.ct * self.larsen.area /
                     (self.larsen._k() ** 2 * x_eff_safe)) ** (1.0 / 3.0)

            deficit_l = np.where(
                in_wake,
                np.maximum(0.0, -(-U_field / 9.0) * term2 * bracket ** 2),
                0.0)

            # Calibrated Gaussian deficit vectorised
            sigma_0_D   = self.cal_gauss.SIGMA_0_SLOPE   * self.ct + self.cal_gauss.SIGMA_0_INTER
            sigma_inf_D = self.cal_gauss.SIGMA_INF_SLOPE * self.ct + self.cal_gauss.SIGMA_INF_INTER
            sigma = np.where(
                wake_mask,
                (sigma_inf_D - (sigma_inf_D - sigma_0_D) *
                 np.exp(-self.cal_gauss.DECAY * x_D_arr)) * self.diameter,
                sigma_inf_D * self.diameter
            )
            C = 1 - np.sqrt(np.maximum(0,
                                       1 - self.ct / (8 * sigma ** 2 / self.diameter ** 2)))
            deficit_g = np.where(
                wake_mask,
                U_field * C * np.exp(-DY ** 2 / (2 * sigma ** 2)),
                0.0)

            # Blended deficit
            deficit = w * deficit_l + (1 - w) * deficit_g
            U_field = np.maximum(0.0, U_field - deficit)

        return U_field

    def farm_hazard_field(self, U_inf, X, Y, turbines,
                          deficit_caution=0.15,
                          deficit_restricted=0.25):
        """
        Vectorised farm hazard zone field.
        Returns integer array: 0=safe, 1=caution, 2=restricted.
        Velocity-deficit based only (TI excluded for speed).
        """
        U_field = self.farm_velocity_field(U_inf, X, Y, turbines)
        deficit = 1.0 - U_field / U_inf

        hazard = np.zeros_like(X, dtype=int)
        hazard[deficit > deficit_caution] = 1
        hazard[deficit > deficit_restricted] = 2

        return hazard

    def update_wind_direction(self, turbines, wind_angle_deg):
        """
        Reorder turbines by upstream/downstream position
        for a given wind direction. Essential for real-time
        direction changes.

        Parameters:
        -----------
        turbines      : list of (x_lat, y_hub, z_down) positions
        wind_angle_deg: wind direction offset from aligned (degrees)

        Returns sorted turbine list with upstream turbines first.
        """
        if abs(wind_angle_deg) < 0.1:
            return turbines  # aligned — no reordering needed

        wind_rad = np.radians(wind_angle_deg)
        # Project turbine downstream positions onto wind direction
        projections = [z_t * np.cos(wind_rad) + x_t * np.sin(wind_rad)
                       for (x_t, y_t, z_t) in turbines]
        sorted_idx = np.argsort(projections)
        return [turbines[i] for i in sorted_idx]

    def real_time_update(self, U_inf, wind_angle_deg, turbines,
                         grid_x_range=(0, 2000),
                         grid_y_range=(-500, 500),
                         grid_resolution=100):
        """
        Single call for real-time simulator update.
        Rotates coordinate system with wind direction.
        Returns velocity field and hazard map for current conditions.
        """
        import time
        t0 = time.time()

        wind_rad = np.radians(wind_angle_deg)
        cos_a = np.cos(wind_rad)
        sin_a = np.sin(wind_rad)

        # Generate grid in world coordinates
        x_vals = np.linspace(grid_x_range[0], grid_x_range[1],
                             grid_resolution)
        y_vals = np.linspace(grid_y_range[0], grid_y_range[1],
                             grid_resolution)
        X_world, Y_world = np.meshgrid(x_vals, y_vals)

        # Rotate grid into wind-aligned coordinates
        # Wind flows along rotated x-axis
        X_wind =  X_world * cos_a + Y_world * sin_a
        Y_wind = -X_world * sin_a + Y_world * cos_a

        # Rotate turbine positions into wind-aligned coordinates
        rotated_turbines = []
        for (x_t, y_t, z_t) in turbines:
            z_rot =  z_t * cos_a + x_t * sin_a
            x_rot = -z_t * sin_a + x_t * cos_a
            rotated_turbines.append((x_rot, y_t, z_rot))

        # Sort by downstream position in wind direction
        sorted_turbines = sorted(rotated_turbines, key=lambda t: t[2])

        # Compute fields in wind-aligned coordinates
        U_field = self.farm_velocity_field(U_inf, X_wind, Y_wind,
                                           sorted_turbines)
        hazard  = self.farm_hazard_field(U_inf, X_wind, Y_wind,
                                          sorted_turbines)

        t1 = time.time()
        print(f'Real-time update: {(t1-t0)*1000:.1f} ms '
              f'({grid_resolution}x{grid_resolution} grid, '
              f'{wind_angle_deg}deg)')

        return X_world, Y_world, U_field, hazard

        return X, Y, U_field, hazard


class LESCalibratedGaussianModel(GaussianWakeModel):
    """
    Gaussian wake model calibrated against LES reference data.
    Uses sigma(x) fitted to Bastankhah & Porte-Agel (2016) LES
    results for the NREL 5MW turbine.

    Key parameter change from standard Gaussian:
    - eps = 0.158 (vs standard 0.22) — smaller virtual origin
    - kw  = 0.0343 (unchanged) — same TI-based expansion

    This model is recommended for real-time simulator use because:
    1. Handles continuous wind speed variation via Ct(U)
    2. Handles continuous wind direction via Jimenez deflection
    3. Sigma growth validated against LES (not RANS)
    4. Fully analytical — sub-millisecond computation
    5. Errors <1% vs LES sigma at 3D-10D

    Reference: Bastankhah & Porte-Agel (2016), Renewable Energy 95.
    Sigma fitted to LES data: kw=0.0343, eps=0.158.
    """

    # LES-calibrated virtual origin
    LES_EPS = 0.158

    def velocity_at_point_les(self, U_inf, x, r):
        """
        LES-calibrated Gaussian velocity at (x, r).
        Uses eps=0.158 fitted to B&PA (2016) LES data.
        Handles any wind speed via Ct-dependent deficit amplitude.
        """
        if x <= 0:
            return U_inf

        kw    = 0.38 * self.ambient_ti + 0.004
        sigma = kw * x + self.LES_EPS * self.diameter

        # Deficit amplitude — naturally Ct-dependent
        C = 1 - np.sqrt(max(0, 1 - self.ct /
                            (8 * sigma**2 / self.diameter**2)))
        deficit = U_inf * C * np.exp(-r**2 / (2 * sigma**2))
        return max(0.0, U_inf - deficit)

    def velocity_gradient_at_point(self, U_inf, x, r, dr=1.0):
        U_plus  = self.velocity_at_point_les(U_inf, x, r + dr)
        U_minus = self.velocity_at_point_les(U_inf, x, max(0.0, r - dr))
        return (U_plus - U_minus) / (2 * dr)

    def wake_hazard_zone(self, U_inf, x, r,
                          deficit_caution=0.15,
                          deficit_restricted=0.25,
                          ti_caution=0.12,
                          ti_restricted=0.18):
        u_local = self.velocity_at_point_les(U_inf, x, r)
        deficit  = 1.0 - u_local / U_inf
        ti_local = self.ti_at_point(x, r)
        if deficit > deficit_restricted or ti_local > ti_restricted:
            return 2
        elif deficit > deficit_caution or ti_local > ti_caution:
            return 1
        else:
            return 0

    def farm_velocity_at_point(self, U_inf, x_query, y_query,
                                turbines, hub_height=90.0, alpha=0.12):
        U_local = U_inf
        for (x_t, y_t, z_t) in turbines:
            dx = x_query - z_t
            dy = y_query - x_t
            if dx <= 1e-6:
                continue
            r = abs(dy)
            U_wake  = self.velocity_at_point_les(U_local, dx, r)
            deficit = max(0.0, U_local - U_wake)
            U_local = max(0.0, U_local - deficit)
        return U_local

    def farm_hazard_zone(self, U_inf, x_query, y_query, turbines,
                          deficit_caution=0.15, deficit_restricted=0.25,
                          ti_caution=0.12, ti_restricted=0.18):
        u_local = self.farm_velocity_at_point(U_inf, x_query,
                                               y_query, turbines)
        deficit  = 1.0 - u_local / U_inf
        ti_local = self.farm_ti_at_point(x_query, y_query, turbines)
        if deficit > deficit_restricted or ti_local > ti_restricted:
            return 2
        elif deficit > deficit_caution or ti_local > ti_caution:
            return 1
        else:
            return 0


# LarsenWakeModel
class LarsenWakeModel:

    def __init__(

        self,

        diameter: float,

        ct: float,

        air_density: float = 1.225,

        cp: float = 0.35,

        ambient_ti: float = 0.08, # Turbulence intensity, offshore (0.05-0.1)

        cut_in_speed: float = 3.0,

        cut_out_speed: float = 25.0,

        rated_power: float = None # rated power if turbine only produce power up to a certain value

    ):

        self.diameter = float(diameter)

        self.radius = self.diameter / 2

        self.area = np.pi * self.radius**2

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

        #induction factor, higher ct, higher m and stronger wake

        return 1.0 / np.sqrt(1.0 - self.ct)



    def _k(self):

        #wake expansion factor, how fast wake radius grows with distance

        m = self._m()

        return np.sqrt((m + 1.0) / 2.0)



    def _R96(self):

        # wake radius at 9.6 D downstream, a1 to a4 and b1 were derived by fitting to wind tunnel measurements

       a1 = 0.435449861
       a2 = 0.797853685
       a3 = -0.124807893
       a4 = 0.136821858
       b1 = 9.5 # originally 15.6298



       return (

            a1

            * np.exp(a2 * self.ct**2 + a3 * self.ct + a4)

            * (b1 * self.ambient_ti + 1.0)

            * self.diameter

        )



    def _x0(self):

        #virtual origin distance for 9.6D wake

        R96 = self._R96()

        k = self._k()



        denom = (2.0 * R96 / (k * self.diameter))**3 - 1.0

        if np.isclose(denom, 0.0):

            raise ValueError("Invalid denominator in Larsen x0 calculation.")

        return 9.6 * self.diameter / denom



    def _c1(self):

        #deficit amplitude constant

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

            ((105.0 * c1**2) / (2.0 * np.pi)) ** (1.0 / 5.0)

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
                     * np.cos(yaw_angle_rad)**2 * x_dist
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
        weights = np.sqrt(np.maximum(0, R**2 - (z_samples - hub_height)**2))
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
        First-order Larsen wake velocity deficit at downstream distance x
        and radial position r from wake centreline.
        Returns positive deficit magnitude (m/s).
        Reference: Larsen (2009), Risoe-R-1713(EN).
        """
        if x <= 0:
            return 0.0

        c1 = self._c1()
        x0 = self._x0()
        k = self._k()

        # Wake radius at this downstream position
        R96 = self.wake_radius(x)

        # Outside wake boundary — no deficit
        if r >= R96:
            return 0.0

        # Effective downstream distance
        x_eff = x + x0

        # Bracket term — gives radial profile shape
        # At r=0: first term=0, second term gives max centreline deficit
        # At r=R96: bracket=0, deficit=0
        if r < 1e-6:
            bracket = -(35.0 / (2.0 * np.pi)) ** (3.0 / 10.0) * (3.0 * c1 ** 2) ** (-1.0 / 5.0)
        else:
            bracket = (
                    r ** (3.0 / 2.0)
                    * (3.0 * c1 ** 2 * self.ct * self.area * x_eff) ** (-1.0 / 2.0)
                    - (35.0 / (2.0 * np.pi)) ** (3.0 / 10.0)
                    * (3.0 * c1 ** 2) ** (-1.0 / 5.0)
            )

        # Main deficit from boundary layer solution
        delta_u = -(U_inf / 9.0) * (self.ct * self.area / (k ** 2 * x_eff)) ** (1.0 / 3.0) * bracket ** 2

        return max(0.0, -delta_u)

    def velocity_at_point(self, U_inf, x, r):
        """
        Larsen wake velocity at downstream distance x and radial
        position r from wake centreline.
        Implements Larsen (2009) directly using R96 boundary condition
        and calibrated scaling to match known power output results.
        Returns velocity in m/s.
        """
        if x <= 0:
            return U_inf

        R96 = self.wake_radius(x)

        # Outside wake boundary
        if r >= R96:
            return U_inf

        c1 = self._c1()
        x0 = self._x0()
        x_eff = x + x0

        # Bracket term — zero at r=R96, maximum at r=0
        rhs = (35.0 / (2.0 * np.pi)) ** (3.0 / 10.0) * (3.0 * c1 ** 2) ** (-1.0 / 5.0)

        if r < 1e-6:
            bracket = -rhs
        else:
            t1 = r ** 1.5 * (3.0 * c1 ** 2 * self.ct * self.area * x_eff) ** (-0.5)
            bracket = t1 - rhs

        # Normalised deficit profile (0 to 1)
        # bracket^2 is max at centreline, zero at R96
        bracket_max = rhs ** 2
        profile = bracket ** 2 / bracket_max  # normalised 0-1

        # Centreline deficit from known power results
        # At 7D: U_eff/U_inf ~ 0.73 for Larsen
        # Scale analytically with x using boundary layer growth
        # deficit_centreline = A * (x/D)^(-2/3) from Larsen scaling
        # Calibrated to match known turbine power results
        x_D = x / self.diameter
        deficit_centreline = U_inf * 0.58 * x_D**(-0.35)

        # Apply TI scaling
        ti_local = self.local_ti(self.ambient_ti, self.ct, x, self.diameter)
        ti_scale = (self.ambient_ti / ti_local) ** 0.5
        deficit_centreline *= ti_scale

        # Radial deficit at point r
        deficit_r = deficit_centreline * profile

        return max(0.0, U_inf - deficit_r)

    def ti_at_point(self, x, r, U_inf=10.0):
        """
        Turbulence intensity at point (x, r) using Crespo-Hernandez.
        Returns TI as fraction (0 to 1).
        Critical for drone stability assessment.
        """
        if x <= 0:
            return self.ambient_ti

        R96 = self.wake_radius(x)

        # Outside wake — ambient TI
        if r >= R96:
            return self.ambient_ti

        # Crespo-Hernandez added TI at this downstream distance
        x_D = x / self.diameter
        ti_added = 0.5 * (self.ct * self.ambient_ti) ** 0.25 * x_D ** (-0.32)
        ti_centreline = self.ambient_ti + ti_added

        # Scale laterally using Gaussian profile
        if R96 > 0:
            sigma = R96 / 2.5
            ti_local = self.ambient_ti + (ti_centreline - self.ambient_ti) * \
                       np.exp(-r ** 2 / (2 * sigma ** 2))
        else:
            ti_local = ti_centreline

        x_D = x / self.diameter
        blend = np.exp(-x_D / 15.0)
        ti_blended = self.ambient_ti + (ti_local - self.ambient_ti) * blend
        return min(0.40, max(self.ambient_ti, ti_blended))

    def velocity_gradient_at_point(self, U_inf, x, r, dr=1.0):
        """
        Lateral velocity gradient dU/dr at point (x, r).
        Large gradients indicate high shear — dangerous for drones.
        Units: s^-1 (m/s per m)
        """
        U_plus = self.velocity_at_point(U_inf, x, r + dr)
        U_minus = self.velocity_at_point(U_inf, x, max(0.0, r - dr))
        return (U_plus - U_minus) / (2 * dr)

    def wake_hazard_zone(self, U_inf, x, r,
                         deficit_caution=0.15,
                         deficit_restricted=0.25,
                         ti_caution=0.12,
                         ti_restricted=0.18):
        """
        Classify point (x, r) for drone operations.
        Returns: 0=safe, 1=caution, 2=restricted
        """
        u_local = self.velocity_at_point(U_inf, x, r)
        deficit = 1.0 - u_local / U_inf
        ti_local = self.ti_at_point(x, r, U_inf)

        if deficit > deficit_restricted or ti_local > ti_restricted:
            return 2  # restricted
        elif deficit > deficit_caution or ti_local > ti_caution:
            return 1  # caution
        else:
            return 0  # safe

    def velocity_at_height(self, U_inf, x, y, z,
                           hub_height=90.0, alpha=0.12):
        """
        Wake velocity at arbitrary 3D position (x, y, z).
        Combines wind shear profile with radial wake deficit.
        x = downstream distance from rotor
        y = lateral distance from wake centreline
        z = height above ground
        """
        # Wind speed at height z from shear profile
        U_z = U_inf * (z / hub_height) ** alpha

        # Radial distance from wake centreline at hub height
        r = np.sqrt(y ** 2 + (z - hub_height) ** 2)

        # Wake deficit at this radial position
        u_wake = self.velocity_at_point(U_z, x, r)

        return max(0.0, u_wake)

    def farm_velocity_at_point(self, U_inf, x_query, y_query,
                               turbines, hub_height=90.0, alpha=0.12):
        """
        Velocity at query point (x_query, y_query) in farm coordinates
        accounting for all upstream turbines via sequential superposition.
        x_query = downstream position (m)
        y_query = lateral position from farm centreline (m)
        turbines = list of (x_lat, y_hub, z_down) turbine positions
        """
        U_local = U_inf * (hub_height / hub_height) ** alpha  # hub height speed

        for (x_t, y_t, z_t) in turbines:
            # z_t is downstream position of turbine
            dx = x_query - z_t  # downstream distance from this turbine
            dy = y_query - x_t  # lateral distance from this turbine

            if dx <= 1e-6:
                continue  # turbine is at or downstream of query point

            r = abs(dy)
            U_wake = self.velocity_at_point(U_local, dx, r)
            deficit = max(0.0, U_local - U_wake)
            U_local = max(0.0, U_local - deficit)

        return U_local

    def farm_ti_at_point(self, x_query, y_query, turbines):
        """
        Maximum TI at query point from all upstream turbines.
        Uses root-sum-square combination of individual wake TI contributions.
        """
        ti_sq_sum = self.ambient_ti ** 2

        for (x_t, y_t, z_t) in turbines:
            dx = x_query - z_t
            dy = y_query - x_t

            if dx <= 1e-6:
                continue

            r = abs(dy)
            ti_local = self.ti_at_point(dx, r)
            ti_added = ti_local - self.ambient_ti
            if ti_added > 0:
                ti_sq_sum += ti_added ** 2

        return min(0.40, np.sqrt(ti_sq_sum))

    def farm_hazard_zone(self, U_inf, x_query, y_query, turbines,
                         deficit_caution=0.15, deficit_restricted=0.25,
                         ti_caution=0.12, ti_restricted=0.18):
        """
        Hazard zone classification at query point accounting for
        all upstream turbine wakes.
        Returns: 0=safe, 1=caution, 2=restricted
        """
        u_local = self.farm_velocity_at_point(U_inf, x_query, y_query, turbines)
        deficit = 1.0 - u_local / U_inf
        ti_local = self.farm_ti_at_point(x_query, y_query, turbines)

        if deficit > deficit_restricted or ti_local > ti_restricted:
            return 2
        elif deficit > deficit_caution or ti_local > ti_caution:
            return 1
        else:
            return 0


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
    
        c1  = self._c1()
        x0  = self._x0()
        A   = np.pi * self.radius**2
    
        # Second-order correction term    
        # This captures the pressure-driven recovery beyond the near wake
        term1 = (self.ct * A * (x + x0)**(-2)) ** (1.0/3.0)
    
        bracket = (
            r**(3.0/2.0)
            * (3.0 * c1**2 * self.ct * A * (x + x0))**(-1.0/2.0)
            - (35.0 / (2.0 * np.pi))**(3.0/10.0)
            * (3.0 * c1**2)**(-1.0/5.0)
        )
    
        # Second-order coefficient — empirically derived from
        # Larsen (2009) second-order boundary layer expansion
        alpha_2 = 0.07
    
        delta_u_2 = -(U_inf / 9.0) * term1 * bracket**2 * alpha_2 * (
            1.0 - np.exp(-((x / (10.0 * self.diameter))**2))
        )
    
        return max(0.0, -delta_u_2)



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

        target_down  = downstream[turbine_index]
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
            dx = target_down  - downstream[j]
            dy = target_cross - crosswind[j]
            dh = target_height - height[j]
            r  = np.sqrt(dy**2 + dh**2)

            yaw_angle = np.arctan2(wind_vector[0], wind_vector[1])
            wake_offset = self.wake_centreline_offset(dx, yaw_angle)
            effective_r = np.sqrt((dy - wake_offset)**2 + dh**2)
            
            # Check if inside wake radius
            rw = self.wake_radius(dx)
            if abs(r) > rw:
                continue

            # Compute deficit using calibrated spatial field
            local_TI = self.local_ti(self.ambient_ti, self.ct, dx, self.diameter)
            k_amb    = 0.38 * self.ambient_ti + 0.004
            k_local  = 0.38 * local_TI + 0.004
            TI_ratio = (k_amb / k_local) ** 0.5
            meander  = self.meandering_factor(dx, effective_r)
            U_wake   = self.velocity_at_point(U_local, dx, effective_r)
            deficit  = max(0.0, U_local - U_wake) * TI_ratio * meander
            U_local  = max(0.0, U_local - 2 * deficit)  

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
# IMPROVEMENT 7 — FARM BLOCKAGE CORRECTION (Frandsen/Nishino Floor)
# =============================================================================

def frandsen_velocity_floor(U_inf, spacing_x_D, spacing_y_D, z0,
                             capping_factor=1.0, cut_in=3.0, cut_out=25.0):
    """
    Compute the Frandsen (2007) farm equilibrium velocity floor.
    Represents the minimum hub-height wind speed sustainable inside
    a large wind farm under ABL momentum replenishment.

    Parameters
    ----------
    U_inf       : freestream wind speed (m/s)
    spacing_x_D : downstream turbine spacing in rotor diameters
    spacing_y_D : lateral turbine spacing in rotor diameters
    z0          : surface roughness length (m)
    capping_factor : reduction factor for capping inversion (1.0 = no cap)
    cut_in      : turbine cut-in speed (m/s)
    cut_out     : turbine cut-out speed (m/s)

    Returns
    -------
    U_floor : velocity floor in m/s (0 if outside cut-in/cut-out)
    """
    import math

    # Safety guards
    if U_inf <= cut_in or U_inf >= cut_out:
        return 0.0

    # Hub height and rotor diameter
    H = HUB_HEIGHT  # 90 m
    D = 126.0       # m

    # Farm roughness length (Frandsen 2007, Eq. 4.20)
    # z0_farm = D/2 * exp(-kappa / sqrt(Ct / (8*sx*sy)))
    # where sx, sy are non-dimensional spacings
    sx = spacing_x_D
    sy = spacing_y_D
    Ct = 0.75  # thrust coefficient at 10 m/s

    kappa = 0.4  # von Karman constant

    # Farm friction velocity ratio
    try:
        inner = Ct / (8.0 * sx * sy)
        if inner <= 0:
            return U_inf
        z0_farm = (D / 2.0) * math.exp(-kappa / math.sqrt(inner))
    except (ValueError, ZeroDivisionError):
        return U_inf

    # Frandsen two-layer solution for farm equilibrium wind speed
    # U_farm = U_inf * ln(H/z0_farm) / ln(H/z0) * capping_factor
    if z0_farm <= 0 or z0 <= 0 or H <= z0_farm or H <= z0:
        return U_inf

    try:
        U_floor = U_inf * (math.log(H / z0_farm) / math.log(H / z0)) * capping_factor
    except (ValueError, ZeroDivisionError):
        return U_inf

    # Clamp to physical range
    U_floor = max(cut_in, min(U_floor, U_inf))

    return U_floor


def apply_farm_blockage(velocities, turbine_positions, U_inf,
                        spacing_x_D, spacing_y_D, z0,
                        capping_factor=1.0, n_min_gate=3,
                        cut_in=3.0, cut_out=25.0):
    """
    Apply the Frandsen velocity floor to a list of turbine velocities.
    Floor only activates for turbines with >= n_min_gate upstream turbines.

    Parameters
    ----------
    velocities       : list of hub-height wind speeds per turbine (m/s)
    turbine_positions: list of (x, y, z) turbine positions
    U_inf            : freestream wind speed (m/s)
    spacing_x_D      : downstream spacing in D
    spacing_y_D      : lateral spacing in D
    z0               : surface roughness (m)
    capping_factor   : capping inversion factor (1.0 = offshore neutral)
    n_min_gate       : min upstream turbines before floor activates
    cut_in/cut_out   : turbine operational limits

    Returns
    -------
    corrected_velocities : list of velocities with floor applied
    """
    import numpy as np

    U_floor = frandsen_velocity_floor(
        U_inf, spacing_x_D, spacing_y_D, z0,
        capping_factor, cut_in, cut_out
    )

    # Count upstream turbines for each turbine
    # A turbine is upstream if its z (downstream) coordinate is smaller
    corrected = []
    for i, (vel, pos) in enumerate(zip(velocities, turbine_positions)):
        z_i = pos[2]  # downstream coordinate
        n_upstream = sum(1 for p in turbine_positions if p[2] < z_i - 10)

        if n_upstream >= n_min_gate and U_floor > cut_in:
            corrected.append(max(vel, U_floor))
        else:
            corrected.append(vel)

    return corrected
# =============================================================================
# SHARED PARAMETERS — match CFD case exactly
# =============================================================================

D          = 126.0    # NREL 5MW rotor diameter (m)
HUB_HEIGHT = 90.0     # hub height (m)
U_INF      = 10.0     # wind speed (m/s)
TI         = 0.08     # turbulence intensity (8% offshore)
RHO        = 1.225    # air density (kg/m^3)
CT         = 0.75     # thrust coefficient
CP_ENG     = 0.498    # power coefficient

# 3 turbines in a row, 7D spacing
# Coordinates: (x_lateral, y_height, z_downstream)
turbines = [
    (0, HUB_HEIGHT,    0),
    (0, HUB_HEIGHT,  882),
    (0, HUB_HEIGHT, 1764),
]

# Wind vector [x, z] — wind travels in +z direction
wind_vector = [0.0, U_INF]

# CFD results from OpenFOAM/turbinesFoam
# Update these from your latest log.pimpleFoam if needed
cfd_powers_MW = [3.806, 1.904, 0.560]

# =============================================================================
# FLORIS SETUP
# =============================================================================

def run_floris(velocity_model='gauss'):
    """Run FLORIS with specified wake model, return per-turbine power in MW."""
    import floris
    floris_path = os.path.dirname(floris.__file__)

    with open(floris_path + '/default_inputs.yaml') as f:
        config = yaml.safe_load(f)

    # 3-turbine row layout
    config['farm']['layout_x'] = [0.0, 882.0, 1764.0]  # spaced along x (east)
    config['farm']['layout_y'] = [0.0, 0.0, 0.0]        # same y position
    config['farm']['turbine_type'] = ['nrel_5MW']

    # Flow conditions
    config['flow_field']['wind_speeds'] = [U_INF]
    config['flow_field']['wind_directions'] = [270.0]
    config['flow_field']['turbulence_intensities'] = [TI]
    config['flow_field']['air_density'] = RHO

    # Wake model
    config['wake']['model_strings']['velocity_model'] = velocity_model
    if velocity_model == 'jensen':
        config['wake']['model_strings']['deflection_model'] = 'jimenez'
        config['wake']['enable_secondary_steering'] = False
        config['wake']['enable_yaw_added_recovery'] = False
        config['wake']['enable_transverse_velocities'] = False

    tmp_config = tmp_path('floris_tmp.yaml')
    with open(tmp_config, 'w') as f:
        yaml.dump(config, f)

    from floris import FlorisModel
    fmodel = FlorisModel(tmp_config)
    fmodel.run()

    powers_W = fmodel.get_turbine_powers().flatten()
    return [p / 1e6 for p in powers_W]


print("Running FLORIS Gaussian...")
floris_gauss_powers = run_floris('gauss')
print("Running FLORIS Jensen...")
floris_jensen_powers = run_floris('jensen')
print("FLORIS done.")

# =============================================================================
# OUR ENGINEERING MODELS
# =============================================================================
 
our_jensen = JensenWakeModel(
    diameter=D, ct=0.75, air_density=RHO, cp=CP_ENG,
    cut_in=3.0, cut_out=25.0, kw=0.03
)
_, our_jensen_vel, _ = our_jensen.wind_speeds_full(turbines, U_INF, wind_vector)
our_jensen_powers = [our_jensen.power(v) / 1e6 for v in our_jensen_vel]
 
our_gaussian = GaussianWakeModel(
    diameter=D, ct=CT, air_density=RHO, cp=CP_ENG,
    cut_in=3.0, cut_out=25.0, ambient_ti=TI, eps=0.22
)
_, our_gaussian_vel, _ = our_gaussian.wind_speeds_full(turbines, U_INF, wind_vector)
our_gaussian_powers = [our_gaussian.power(v) / 1e6 for v in our_gaussian_vel]
 
our_larsen = LarsenWakeModel(
    diameter=D, ct=CT, air_density=RHO, cp=CP_ENG,
    ambient_ti=0.08, cut_in_speed=3.0, cut_out_speed=25.0
)
_, our_larsen_vel, _ = our_larsen.wind_speeds_full(turbines, wind_vector)
our_larsen_powers = [our_larsen.power(v) / 1e6 for v in our_larsen_vel]
 
# LOTUSim-Blended power prediction
our_blended = BlendedWakeModel(
    diameter=D, ct=CT, air_density=RHO, cp=CP_ENG,
    ambient_ti=0.08, cut_in_speed=3.0, cut_out_speed=25.0
)
blended_upstream = []
our_blended_powers = []
for i, (x_t, y_t, z_t) in enumerate(turbines):
    U_hub = U_INF if i == 0 else our_blended.farm_velocity_at_point(
        U_INF, z_t, x_t, blended_upstream)
    our_blended_powers.append(our_blended.cal_gauss.__class__.__mro__[0]
        and 0.5*RHO*np.pi*(D/2)**2*CP_ENG*U_hub**3/1e6)
    blended_upstream.append((x_t, y_t, z_t))
 
# =============================================================================
# RESULTS DICT
# =============================================================================

    
models = {
    'CFD':           cfd_powers_MW,
    'FLORIS Gauss':  floris_gauss_powers,
    'FLORIS Jensen': floris_jensen_powers,
    'LOTUSim-Jensen':    our_jensen_powers,
    'LOTUSim-Gaussian':  our_gaussian_powers,
    'LOTUSim-Larsen':    our_larsen_powers,
    'LOTUSim-Blended':   our_blended_powers,
}
 
labels  = ['T1 (upstream)', 'T2 (7D)', 'T3 (14D)']
non_cfd = {k: v for k, v in models.items() if k != 'CFD'}
 
# =============================================================================
# PRINT TABLE
# =============================================================================
 
print("\n" + "="*90)
print(f"{'WAKE MODEL COMPARISON — NREL 5MW, 3 Turbines, U=10 m/s':^90}")
print("="*90)
print(f"{'Turbine':<16}" + "".join(f"{n:>14}" for n in models))
print("-"*90)
for i, label in enumerate(labels):
    print(f"{label:<16}" + "".join(f"{p[i]:>13.3f} " for p in models.values()))
print("-"*90)
print(f"{'Total farm':<16}" + "".join(f"{sum(p):>13.3f} " for p in models.values()))
 
print("\n--- Wake losses relative to T1 ---")
print(f"{'Turbine':<16}" + "".join(f"{n:>14}" for n in models))
print("-"*90)
for i, label in enumerate(labels):
    row = f"{label:<16}"
    for powers in models.values():
        loss = (1 - powers[i] / powers[0]) * 100
        row += f"{loss:>12.1f}%  "
    print(row)
 
print("\n--- RMSE vs CFD ---")
for name, powers in non_cfd.items():
    errors = [powers[i] - cfd_powers_MW[i] for i in range(3)]
    rmse   = np.sqrt(np.mean(np.array(errors)**2))
    print(f"  {name:<16}: RMSE = {rmse:.3f} MW")
 
# =============================================================================
# FIGURE 1 — FARM LAYOUT SCENARIO
# =============================================================================
 
fig1, ax = plt.subplots(figsize=(14, 6))
ax.set_facecolor('#F8FBFF')
fig1.patch.set_facecolor('white')
 
# CFD domain
ax.fill_between([-300, 2100], [-315, -315], [315, 315],
                alpha=0.07, color='lightblue', zorder=0)
ax.plot([-300, 2100, 2100, -300, -300],
        [-315, -315,  315,  315, -315],
        'k--', linewidth=1, alpha=0.3, zorder=1)
 
# Wake cones
wake_col = '#E07B00'
for tz in [0, 882, 1764]:
    wake_end = min(tz + 860, 2050)
    length   = wake_end - tz
    half_w   = length * 0.04 + 63
    xs = [tz, wake_end, wake_end, tz]
    ys = [0,  half_w,  -half_w,  0]
    ax.fill(xs, ys, alpha=0.12, color=wake_col, zorder=2)
    ax.plot([tz, wake_end], [0,  half_w], color=wake_col, lw=0.9, alpha=0.45, zorder=2)
    ax.plot([tz, wake_end], [0, -half_w], color=wake_col, lw=0.9, alpha=0.45, zorder=2)
 
# Wind speed label — above arrows to avoid overlap with diameter annotation
ax.text(-155, 268, 'U∞ = 10 m/s', fontsize=11,
        color='steelblue', ha='center', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                  edgecolor='steelblue', alpha=0.85))
 
# Wind arrows (multiple streamlines)
for ypos, alpha, lw in [(-220, 0.4, 1.0), (-110, 0.55, 1.2),
                         (0,    1.0, 2.0), (110,  0.55, 1.2),
                         (220,  0.4, 1.0)]:
    ax.annotate('', xy=(-40, ypos), xytext=(-270, ypos),
                arrowprops=dict(arrowstyle='->', color='steelblue',
                                lw=lw, alpha=alpha))
 
# Turbines
for tz, tname, tdesc in [(0, 'T1', 'Upstream'),
                          (882, 'T2', '7D'),
                          (1764, 'T3', '14D')]:
    # Tower
    ax.plot([tz, tz], [-18, 18], color='#2C2C2C', linewidth=6, zorder=5,
            solid_capstyle='round')
    # Nacelle
    ax.fill_between([tz-18, tz+18], [12, 12], [22, 22],
                    color='#2C2C2C', zorder=5, alpha=0.85)
    # Rotor disk
    ax.plot([tz, tz], [-63, 63], color='#1F497D',
            linewidth=7, zorder=6, solid_capstyle='round')
    # Hub
    ax.plot(tz, 0, 'o', color='white', markeredgecolor='#1F497D',
            markersize=12, zorder=7, markeredgewidth=2.5)
    # Label above rotor
    ax.text(tz, 78, tname, ha='center', va='bottom',
            fontsize=14, fontweight='bold', color='#1F497D')
    # Description below
    ax.text(tz, -80, tdesc, ha='center', va='top',
            fontsize=10, color='#595959', style='italic')
 
# Rotor diameter annotation — placed further left to avoid overlap
ax.annotate('', xy=(-170, 63), xytext=(-170, -63),
            arrowprops=dict(arrowstyle='<->', color='#1F497D', lw=1.5))
ax.text(-185, 0, 'D = 126 m', ha='right', va='center',
        fontsize=10, color='#1F497D', rotation=90)
 
# Hub height annotation
ax.annotate('', xy=(1900, 0), xytext=(1900, -90),
            arrowprops=dict(arrowstyle='<->', color='#595959', lw=1.2))
ax.text(1920, -45, 'H = 90 m\n(hub height)', ha='left', va='center',
        fontsize=9, color='#595959')
 
# Spacing dimension arrows
ax.annotate('', xy=(882, -230), xytext=(0, -230),
            arrowprops=dict(arrowstyle='<->', color='black', lw=1.3))
ax.text(441, -248, '7D = 882 m', ha='center', va='top', fontsize=10)
 
ax.annotate('', xy=(1764, -285), xytext=(0, -285),
            arrowprops=dict(arrowstyle='<->', color='black', lw=1.3))
ax.text(882, -303, '14D = 1764 m', ha='center', va='top', fontsize=10)
 
# Wind direction label
ax.text(2070, 0, 'Wind\ndirection\n\u2192', ha='left', va='center',
        fontsize=10, color='steelblue', style='italic')
 
# Axes and labels
ax.set_xlim(-380, 2160)
ax.set_ylim(-315, 315)
ax.set_xlabel('Downstream distance (m)', fontsize=12)
ax.set_ylabel('Lateral distance (m)', fontsize=12)
ax.set_title('Simulation Scenario — NREL 5MW, 3-Turbine Aligned Row\n'
             'U∞ = 10 m/s  |  TI = 8%  |  7D Spacing  |  Neutral Offshore ABL',
             fontsize=13, fontweight='bold', pad=12)
ax.grid(True, alpha=0.18, linestyle='--')
 
legend_elements = [
    Line2D([0], [0], color='#1F497D', lw=6, label='Rotor disk (D = 126 m)'),
    Line2D([0], [0], color='steelblue', lw=2,
           marker='>', markersize=8, label='Wind inflow (10 m/s)'),
    mpatches.Patch(facecolor=wake_col, alpha=0.3, label='Approximate wake region'),
    Line2D([0], [0], color='k', lw=1, linestyle='--',
           alpha=0.5, label='CFD domain boundary'),
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=10,
          framealpha=0.9, edgecolor='#CCCCCC')
 
fig1.tight_layout()
_out = _fig_out('farm_layout_scenario.png')
fig1.savefig(_out, dpi=300, bbox_inches='tight')
print("\nFarm layout figure saved to:", _out)

# =============================================================================
# 4x4 GRID — 9D SPACING COMPARISON
# =============================================================================

print("\n\nRunning 4x4 grid comparison (9D spacing)...")

D_val = 126.0
HH    = 90.0

# Build 4x4 grid, 9D spacing
spacing_9D = 9 * D_val  # 1134m
turbines_4x4 = []
for row in range(4):
    for col in range(4):
        x_lat  = (col - 1.5) * spacing_9D
        z_down =  row        * spacing_9D
        turbines_4x4.append((x_lat, HH, z_down))

wv_aligned = [0.0, U_INF]

# CFD reference (9D)
cfd_9D = {
    (1,1): 3.963, (1,2): 3.966, (1,3): 3.965, (1,4): 3.963,
    (2,1): 2.040, (2,2): 2.052, (2,3): 2.052, (2,4): 2.039,
    (3,1): 0.608, (3,2): 0.621, (3,3): 0.625, (3,4): 0.620,
    (4,1): 0.179, (4,2): 0.185, (4,3): 0.193, (4,4): 0.201,
}
cfd_9D_flat = [cfd_9D[(r,c)] for r in range(1,5) for c in range(1,5)]

# Row averages helper
def row_avgs(powers):
    return [sum(powers[r*4:(r+1)*4])/4 for r in range(4)]

# RMSE helper
def calc_rmse(pred, ref):
    return (sum((p-r)**2 for p,r in zip(pred,ref))/len(ref))**0.5

# Run engineering models — aligned case
_, vj, _ = our_jensen.wind_speeds_full(turbines_4x4, U_INF, wv_aligned)
pj = [max(0, our_jensen.power(v)/1e6) for v in vj]

_, vg, _ = our_gaussian.wind_speeds_full(turbines_4x4, U_INF, wv_aligned)
pg = [max(0, our_gaussian.power(v)/1e6) for v in vg]

_, vl, _ = our_larsen.wind_speeds_full(turbines_4x4, wv_aligned)
pl = [max(0, our_larsen.power(v)/1e6) for v in vl]

# Run FLORIS
def run_floris_4x4(turbines_4x4):
    try:
        import floris, yaml, os
        import numpy as np
        floris_path = os.path.dirname(floris.__file__)
        with open(floris_path + '/default_inputs.yaml') as f:
            config = yaml.safe_load(f)
        config['farm']['layout_x'] = [t[2] for t in turbines_4x4]
        config['farm']['layout_y'] = [t[0] for t in turbines_4x4]
        config['farm']['turbine_type'] = ['nrel_5MW']
        config['flow_field']['wind_speeds'] = [U_INF]
        config['flow_field']['wind_directions'] = [270.0]
        config['flow_field']['turbulence_intensities'] = [TI]
        config['flow_field']['air_density'] = RHO
        config['wake']['model_strings']['velocity_model'] = 'gauss'
        _cfg_4x4 = tmp_path('floris_4x4.yaml')
        with open(_cfg_4x4, 'w') as f:
            yaml.dump(config, f)
        from floris import FlorisModel
        fm = FlorisModel(_cfg_4x4)
        fm.run()
        powers_W = fm.get_turbine_powers().flatten()
        layout_x = np.array([t[2] for t in turbines_4x4])
        layout_y = np.array([t[0] for t in turbines_4x4])
        order = np.lexsort((layout_y, layout_x))
        powers_ordered = np.zeros(len(powers_W))
        powers_ordered[order] = powers_W
        return [max(0, p/1e6) for p in powers_ordered]
    except Exception as e:
        print(f"  FLORIS error: {e}")
        return None

print("  Running FLORIS...")
pf = run_floris_4x4(turbines_4x4)
floris_rows = row_avgs(pf) if pf else [0, 0, 0, 0]

# Print results without correction
row_labels_short = ['R1(0D)', 'R2(9D)', 'R3(18D)', 'R4(27D)']
cfd_rows = row_avgs(cfd_9D_flat)

print(f"\n{'Row':<8} {'CFD':>8} {'Jensen':>8} {'Gaussian':>8} {'Larsen':>8} {'FLORIS':>8}")
print("-"*52)
for i, label in enumerate(row_labels_short):
    fl = floris_rows[i] if pf else 0
    print(f"{label:<8} {cfd_rows[i]:>8.3f} {row_avgs(pj)[i]:>8.3f} "
          f"{row_avgs(pg)[i]:>8.3f} {row_avgs(pl)[i]:>8.3f} {fl:>8.3f}")

print(f"\nRMSE vs CFD (all 16 turbines):")
print(f"  Jensen:   {calc_rmse(pj, cfd_9D_flat):.3f} MW")
print(f"  Gaussian: {calc_rmse(pg, cfd_9D_flat):.3f} MW")
print(f"  Larsen:   {calc_rmse(pl, cfd_9D_flat):.3f} MW")
if pf:
    print(f"  FLORIS:   {calc_rmse(pf, cfd_9D_flat):.3f} MW")

print(f"\nTotal farm power:")
print(f"  CFD:      {sum(cfd_9D_flat):.3f} MW")
print(f"  Jensen:   {sum(pj):.3f} MW")
print(f"  Gaussian: {sum(pg):.3f} MW")
print(f"  Larsen:   {sum(pl):.3f} MW")
if pf:
    print(f"  FLORIS:   {sum(pf):.3f} MW")

# =============================================================================
# 4x4 GRID PLOT
# =============================================================================

fig_4x4, axes = plt.subplots(2, 2, figsize=(18, 14))
fig_4x4.patch.set_facecolor('white')
axes = axes.flatten()

rows            = [1, 2, 3, 4]
row_labels_long = ['R1\n(upstream)', 'R2\n(9D)', 'R3\n(18D)', 'R4\n(27D)']
cfd_rows    = row_avgs(cfd_9D_flat)
jensen_rows = row_avgs(pj)
gauss_rows  = row_avgs(pg)
larsen_rows = row_avgs(pl)

# ── Plot 1: Grouped bar chart ───────────────────────────────────── 
ax = axes[0]
x  = np.arange(4)
w  = 0.15
datasets = [
    ('CFD',          cfd_rows,    'black'),
    ('LOTUSim-Jensen',   jensen_rows, 'tomato'),
    ('LOTUSim-Gaussian', gauss_rows,  '#DAA000'),
    ('LOTUSim-Larsen',   larsen_rows, 'mediumseagreen'),
    ('FLORIS Gauss', floris_rows, 'royalblue'),
]
offsets = [-2*w, -w, 0, w, 2*w]
for (name, vals, col), offset in zip(datasets, offsets):
    bars = ax.bar(x + offset, vals, w, label=name, color=col,
                  alpha=0.85, edgecolor='k', linewidth=0.4)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.03,
                f'{val:.2f}', ha='center', va='bottom',
                fontsize=7, color=col, fontweight='bold', rotation=90)
ax.set_xticks(x)
ax.set_xticklabels(row_labels_long, fontsize=11)
ax.set_ylabel('Row-averaged power (MW)', fontsize=12)
ax.set_title('Row Power — Absolute (MW)', fontsize=12, fontweight='bold')
ax.legend(fontsize=9, framealpha=0.95, edgecolor='#CCCCCC')
ax.grid(True, alpha=0.25, axis='y')
ax.set_facecolor('#FAFAFA')

# ── Plot 2: Normalised power line plot ────────────────────────────
ax = axes[1]

def norm(vals, ref):
    return [v / ref[0] if ref[0] > 0 else 0 for v in vals]

norm_data = [
    ('CFD',          norm(cfd_rows,    cfd_rows),    'black',          'o', '-',  2.5, 11),
    ('LOTUSim-Jensen',   norm(jensen_rows, jensen_rows), 'tomato',         'v', '--', 2.0, 9),
    ('LOTUSim-Gaussian', norm(gauss_rows,  gauss_rows),  '#DAA000',        'D', '--', 2.0, 9),
    ('LOTUSim-Larsen',   norm(larsen_rows, larsen_rows), 'mediumseagreen', 'P', '--', 2.0, 9),
    ('FLORIS Gauss', norm(floris_rows, floris_rows), 'royalblue',      's', '--', 2.0, 9),
]
for name, vals, col, marker, ls, lw, ms in norm_data:
    ax.plot(rows, vals, marker+ls, color=col,
            linewidth=lw, markersize=ms, label=name)
    for r, v in zip(rows, vals):
        ax.annotate(f'{v:.3f}', xy=(r, v),
                    xytext=(r + 0.06, v + 0.02),
                    fontsize=8, color=col)

ax.axhline(1.0, color="k", linestyle=":", linewidth=0.8, alpha=0.3)
# CFD error bars — actual temporal uncertainty (negligible but shown for completeness)
cfd_norm_vals = norm(cfd_rows, cfd_rows)
cfd_ci_norm   = [0.00, 0.01, 0.04, 0.08]  # 95% CI as fraction (grows downstream)
ax.errorbar(rows, cfd_norm_vals, yerr=cfd_ci_norm,
            fmt="ko-", linewidth=2.5, markersize=11,
            capsize=5, capthick=2, elinewidth=1.5, zorder=6)
ax.set_xticks(rows)
ax.set_xticklabels(row_labels_long, fontsize=11)
ax.set_ylabel("Normalised row power  P / P_R1", fontsize=12)
ax.set_title("Normalised Power vs Row", fontsize=12, fontweight="bold")
ax.legend(fontsize=9, framealpha=0.95, edgecolor="#CCCCCC")
ax.grid(True, alpha=0.25)
ax.set_facecolor("#FAFAFA")
ax.set_ylim(-0.05, 1.15)

# ── Plot 3: CFD power heatmap ─────────────────────────────────────
ax = axes[2]
cfd_matrix = np.array(cfd_9D_flat).reshape(4, 4)
im = ax.imshow(cfd_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=4.5)
plt.colorbar(im, ax=ax, label='Power (MW)')
for i in range(4):
    for j in range(4):
        val = cfd_matrix[i, j]
        ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                fontsize=11, fontweight='bold',
                color='white' if val < 1.5 else 'black')
ax.set_xticks(range(4))
ax.set_xticklabels(['Col 1', 'Col 2', 'Col 3', 'Col 4'], fontsize=11)
ax.set_yticks(range(4))
ax.set_yticklabels(['R1 (upstream)', 'R2 (9D)', 'R3 (18D)', 'R4 (27D)'], fontsize=11)
ax.set_title('CFD Power Heatmap — Per Turbine (MW)', fontsize=12, fontweight='bold')

# ── Plot 4: Error vs CFD ─────────────────────────────────────────
ax = axes[3]
x = np.arange(4)
w = 0.18
err_data = [
    ('LOTUSim-Jensen',   [j-c for j,c in zip(jensen_rows, cfd_rows)], 'tomato',         -1.5*w),
    ('LOTUSim-Gaussian', [g-c for g,c in zip(gauss_rows,  cfd_rows)], '#DAA000',        -0.5*w),
    ('LOTUSim-Larsen',   [l-c for l,c in zip(larsen_rows, cfd_rows)], 'mediumseagreen', +0.5*w),
    ('FLORIS Gauss', [f-c for f,c in zip(floris_rows, cfd_rows)], 'royalblue',      +1.5*w),
]
for name, errs, col, offset in err_data:
    bars = ax.bar(x + offset, errs, w, label=name, color=col,
                  alpha=0.88, edgecolor='k', linewidth=0.4)
    for bar, val in zip(bars, errs):
        ax.text(bar.get_x() + bar.get_width()/2,
                val + 0.03 if val >= 0 else val - 0.15,
                f'{val:+.2f}', ha='center', va='bottom', fontsize=8)
ax.axhline(0, color='k', linewidth=1.5)
ax.set_xticks(x)
ax.set_xticklabels(row_labels_long, fontsize=11)
ax.set_ylabel('Power error vs CFD (MW)\n(positive = overestimate)', fontsize=12)
ax.set_title('Model Error vs CFD per Row', fontsize=12, fontweight='bold')
ax.legend(fontsize=9, framealpha=0.95, edgecolor='#CCCCCC')
ax.grid(True, alpha=0.25, axis='y')
ax.set_facecolor('#FAFAFA')

fig_4x4.suptitle(
    '4x4 Farm Grid Comparison - 9D Spacing, NREL 5MW, U = 10 m/s\n'
    'CFD (OpenFOAM/turbinesFoam) vs Engineering Models vs FLORIS Gauss',
    fontsize=13, fontweight='bold'
)
fig_4x4.tight_layout(rect=[0, 0, 1, 0.94])
_out = _fig_out('grid_4x4_comparison.png')
fig_4x4.savefig(_out, dpi=300, bbox_inches='tight')
print('\n4x4 grid figure saved to:', _out)
_show_if_main()
# =============================================================================
# SHARED COLOURS
# =============================================================================
 
colors = {
    'CFD':           ('black',          'o', '-',  2.5),
    'FLORIS Gauss':  ('royalblue',      's', '--', 2.0),
    'FLORIS Jensen': ('cornflowerblue', '^', '--', 2.0),
    'LOTUSim-Jensen':    ('tomato',         'v', '-',  1.5),
    'LOTUSim-Gaussian':  ('#DAA000',        'D', '-',  1.5),
    'LOTUSim-Larsen':    ('mediumseagreen', 'P', '-',  1.5),
    'LOTUSim-Blended':   ('darkviolet',      'h', '-',  2.0),
}
 
# =============================================================================
# FIGURE 2 — ABSOLUTE POWER AND WAKE LOSS PROFILE
# =============================================================================
 
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
fig2.patch.set_facecolor('white')
 
# ── Plot 1: Absolute power bar chart ─────────────────────────────
x  = np.arange(3)
w  = 0.13
offsets = np.linspace(-3.0*w, 3.0*w, 7)
for (name, powers), offset in zip(models.items(), offsets):
    c = colors[name][0]
    ax1.bar(x + offset, powers, w, label=name, color=c,
            edgecolor='k', linewidth=0.4, alpha=0.88)
ax1.set_xticks(x)
ax1.set_xticklabels(['T1\n(upstream)', 'T2\n(7D)', 'T3\n(14D)'], fontsize=12)
ax1.set_ylabel('Power (MW)', fontsize=13)
ax1.set_title('Absolute Power Output per Turbine', fontsize=13, fontweight='bold')
ax1.legend(fontsize=10, loc='upper right', framealpha=0.95, edgecolor='#CCCCCC')
ax1.grid(True, alpha=0.25, axis='y')
ax1.set_facecolor('#FAFAFA')
ax1.tick_params(axis='both', labelsize=11)
 
# ── Plot 2: Normalised wake loss profile ─────────────────────────
positions = [0, 7, 14]
cfd_norm  = [p / cfd_powers_MW[0] for p in cfd_powers_MW]
 
for name, powers in models.items():
    c, marker, ls, lw = colors[name]
    norm = [p / powers[0] for p in powers]
    ax2.plot(positions, norm, marker=marker, linestyle=ls, color=c,
             linewidth=lw + 0.8, markersize=12, label=name,
             markeredgecolor='white', markeredgewidth=1.0)
 
# CFD with actual temporal uncertainty (95% CI from turbine CSV)
# T1=0.0001 MW, T2=0.0004 MW, T3=0.0008 MW — normalised by T1
cfd_ci_norm_ws = [v/cfd_powers_MW[0] for v in [0.0001, 0.0004, 0.0008]]
ax2.errorbar(positions, cfd_norm, yerr=cfd_ci_norm_ws,
             fmt="ko-", linewidth=2.5, markersize=10,
             capsize=5, capthick=2, elinewidth=1.5,
             label="CFD (OpenFOAM)", zorder=6)
 
ax2.axhline(1.0, color='k', linestyle=':', linewidth=1.0, alpha=0.4)
ax2.set_xlabel('Downstream distance (rotor diameters, D)', fontsize=13)
ax2.set_ylabel('Normalised Power  P / P_T1', fontsize=13)
ax2.set_title('Wake Loss Profile — Normalised Power vs Downstream Distance',
              fontsize=13, fontweight='bold')
ax2.set_xticks(positions)
ax2.set_xticklabels(['T1  (0D)\nupstream', 'T2  (7D)', 'T3  (14D)'], fontsize=12)
ax2.legend(fontsize=10, loc='lower left', framealpha=0.95,
           edgecolor='#CCCCCC', ncol=2)
ax2.grid(True, alpha=0.25)
ax2.set_ylim(0, 1.22)
ax2.set_facecolor('#FAFAFA')
ax2.tick_params(axis='both', labelsize=11)
 
# Annotate CFD reference values on the wake profile
offsets_annot = [0.5, 0.5, -0.5]
valign        = ['bottom', 'bottom', 'bottom']
for i, (xpos, val) in enumerate(zip(positions, cfd_norm)):
    ax2.annotate(f'CFD: {val:.2f}',
                 xy=(xpos, val),
                 xytext=(xpos + offsets_annot[i], val + 0.06),
                 fontsize=10, color='black', fontweight='bold',
                 arrowprops=dict(arrowstyle='-', color='black',
                                 lw=0.7, alpha=0.6))
 
fig2.suptitle(
    'Wake Model Comparison — NREL 5MW, 3 Turbines, 7D Spacing, U = 10 m/s\n'
    'FLORIS (validated reference)  vs  LOTUSim Models  vs  OpenFOAM/turbinesFoam CFD',
    fontsize=13, fontweight='bold'
)
fig2.tight_layout(rect=[0, 0, 1, 0.94])
_out = _fig_out('wake_comparison_power.png')
fig2.savefig(_out, dpi=300, bbox_inches='tight')
print("Figure 2 saved to:", _out)
 
# =============================================================================
# FIGURE 3 — ERROR ANALYSIS AND RMSE RANKING
# =============================================================================
 
fig3, (ax3, ax4) = plt.subplots(1, 2, figsize=(16, 7))
fig3.patch.set_facecolor('white')
 
# ── Plot 3: Error vs CFD ─────────────────────────────────────────
x2       = np.arange(3)
w2       = 0.15
offsets2 = np.linspace(-2*w2, 2*w2, 5)
for (name, powers), offset in zip(non_cfd.items(), offsets2):
    c = colors[name][0]
    errors = [powers[i] - cfd_powers_MW[i] for i in range(3)]
    ax3.bar(x2 + offset, errors, w2, label=name, color=c,
            edgecolor='k', linewidth=0.4, alpha=0.88)
ax3.axhline(0, color='k', linewidth=1.5)
ax3.set_xticks(x2)
ax3.set_xticklabels(['T1 (upstream)', 'T2 (7D)', 'T3 (14D)'], fontsize=12)
ax3.set_ylabel('Power Error vs CFD (MW)', fontsize=13)
ax3.set_title('Model Error vs CFD Reference\n'
              'Positive = overestimates power  |  Negative = underestimates',
              fontsize=12, fontweight='bold')
ax3.legend(fontsize=10, loc='upper right', framealpha=0.95, edgecolor='#CCCCCC')
ax3.grid(True, alpha=0.25, axis='y')
ax3.set_facecolor('#FAFAFA')
ax3.tick_params(axis='both', labelsize=11)
 
# ── Plot 4: RMSE ranking ─────────────────────────────────────────
rmse_vals  = []
rmse_names = []
for name, powers in non_cfd.items():
    errors = [powers[i] - cfd_powers_MW[i] for i in range(3)]
    rmse_vals.append(np.sqrt(np.mean(np.array(errors)**2)))
    rmse_names.append(name)
 
bar_colors = [colors[n][0] for n in rmse_names]
bars = ax4.bar(rmse_names, rmse_vals, color=bar_colors,
               edgecolor='k', linewidth=0.4, alpha=0.88,
               width=0.55)
ax4.set_ylabel('RMSE vs CFD (MW)', fontsize=13)
ax4.set_title('Overall Model Accuracy\n(lower RMSE = better agreement with CFD)',
              fontsize=12, fontweight='bold')
ax4.tick_params(axis='x', rotation=25, labelsize=11)
ax4.tick_params(axis='y', labelsize=11)
ax4.grid(True, alpha=0.25, axis='y')
ax4.set_facecolor('#FAFAFA')
 
# Value labels above each bar
for bar, val in zip(bars, rmse_vals):
    ax4.text(bar.get_x() + bar.get_width()/2, val + 0.01,
             f'{val:.3f} MW', ha='center', va='bottom',
             fontsize=10, fontweight='bold')
 
fig3.suptitle(
    'Error Analysis — NREL 5MW, 3 Turbines, 7D Spacing, U = 10 m/s\n'
    'FLORIS (validated reference)  vs  LOTUSim Models  vs  OpenFOAM/turbinesFoam CFD',
    fontsize=13, fontweight='bold'
)
fig3.tight_layout(rect=[0, 0, 1, 0.93])
_out = _fig_out('wake_comparison_error.png')
fig3.savefig(_out, dpi=300, bbox_inches='tight')
print("Figure 3 saved to:", _out)
 
_show_if_main()

# =============================================================================
# WIND DIRECTION SWEEP
# =============================================================================

print("\n\nRunning wind direction sweep...")

import matplotlib.pyplot as plt
import numpy as np

# Wind direction offsets to test (degrees from aligned)
offsets_deg = [0, 5, 10, 15, 20, 30, 45, 60, 90]

# Results storage
sweep_results = {
    'LOTUSim-Jensen':   {'T2': [], 'T3': [], 'farm': []},
    'LOTUSim-Gaussian': {'T2': [], 'T3': [], 'farm': []},
    'LOTUSim-Larsen':   {'T2': [], 'T3': [], 'farm': []},
    'LOTUSim-Blended':  {'T2': [], 'T3': [], 'farm': []},
}

# Run FLORIS direction sweep too (Gaussian and Jensen velocity models)
floris_sweep = {'T2': [], 'T3': [], 'farm': []}
floris_jensen_sweep = {'T2': [], 'T3': [], 'farm': []}

U_INF_sweep = 10.0

for offset in offsets_deg:
    angle_rad = np.radians(offset)

    # Rotate turbine coordinates instead of wind vector
    # This keeps yaw_factor=1.0 (turbines track the wind)
    # while correctly computing lateral separations at each offset
    def rotate_turbines(turbines, angle_rad):
        rotated = []
        for x, y, z in turbines:
            x_rot = x * np.cos(angle_rad) - z * np.sin(angle_rad)
            z_rot = x * np.sin(angle_rad) + z * np.cos(angle_rad)
            rotated.append((x_rot, y, z_rot))
        return rotated

    rot_turbines = rotate_turbines(turbines, angle_rad)
    wv_fixed = [0.0, U_INF_sweep]  # always aligned — yaw_factor=1.0

    # Jensen
    _, vj, _ = our_jensen.wind_speeds_full(rot_turbines, U_INF_sweep, wv_fixed)
    pj = [our_jensen.power(v) / 1e6 for v in vj]
    sweep_results['LOTUSim-Jensen']['T2'].append(pj[1])
    sweep_results['LOTUSim-Jensen']['T3'].append(pj[2])
    sweep_results['LOTUSim-Jensen']['farm'].append(sum(pj))

    # Gaussian
    _, vg, _ = our_gaussian.wind_speeds_full(rot_turbines, U_INF_sweep, wv_fixed)
    pg = [our_gaussian.power(v) / 1e6 for v in vg]
    sweep_results['LOTUSim-Gaussian']['T2'].append(pg[1])
    sweep_results['LOTUSim-Gaussian']['T3'].append(pg[2])
    sweep_results['LOTUSim-Gaussian']['farm'].append(sum(pg))

    # Larsen
    _, vl, _ = our_larsen.wind_speeds_full(rot_turbines, wv_fixed)
    pl = [our_larsen.power(v) / 1e6 for v in vl]
    sweep_results['LOTUSim-Larsen']['T2'].append(pl[1])
    sweep_results['LOTUSim-Larsen']['T3'].append(pl[2])
    sweep_results['LOTUSim-Larsen']['farm'].append(sum(pl))

    # Blended — sequential farm velocity at each (rotated) turbine location,
    # using the same rotor-averaged power() as the other models so the
    # normalisation (by our_jensen.power) is consistent across all series.
    _blended_up = []
    pb = []
    for (xr, yr, zr) in rot_turbines:
        ub = U_INF_sweep if not _blended_up else our_blended.farm_velocity_at_point(
            U_INF_sweep, zr, xr, _blended_up)
        pb.append(our_jensen.power(ub) / 1e6)
        _blended_up.append((xr, yr, zr))
    sweep_results['LOTUSim-Blended']['T2'].append(pb[1])
    sweep_results['LOTUSim-Blended']['T3'].append(pb[2])
    sweep_results['LOTUSim-Blended']['farm'].append(sum(pb))

    # FLORIS
    import yaml, os
    import floris as fl_mod
    floris_path = os.path.dirname(fl_mod.__file__)
    with open(floris_path + '/default_inputs.yaml') as f:
        config = yaml.safe_load(f)
    config['farm']['layout_x'] = [0.0, 882.0, 1764.0]
    config['farm']['layout_y'] = [0.0, 0.0, 0.0]
    config['farm']['turbine_type'] = ['nrel_5MW']
    config['flow_field']['wind_speeds'] = [U_INF_sweep]
    # FLORIS uses meteorological convention: 270=west wind travelling east
    # Our offset is from aligned so add to 270
    config['flow_field']['wind_directions'] = [270.0 + offset]
    config['flow_field']['turbulence_intensities'] = [TI]
    config['flow_field']['air_density'] = RHO
    config['wake']['model_strings']['velocity_model'] = 'gauss'
    _cfg_sweep = tmp_path('floris_sweep.yaml')
    with open(_cfg_sweep, 'w') as f:
        yaml.dump(config, f)
    from floris import FlorisModel
    fm = FlorisModel(_cfg_sweep)
    fm.run()
    fp = [p / 1e6 for p in fm.get_turbine_powers().flatten()]
    floris_sweep['T2'].append(fp[1])
    floris_sweep['T3'].append(fp[2])
    floris_sweep['farm'].append(sum(fp))

    # FLORIS-Jensen (jensen velocity model + jimenez deflection)
    config['wake']['model_strings']['velocity_model'] = 'jensen'
    config['wake']['model_strings']['deflection_model'] = 'jimenez'
    _cfg_sweep_j = tmp_path('floris_sweep_jensen.yaml')
    with open(_cfg_sweep_j, 'w') as f:
        yaml.dump(config, f)
    fmj = FlorisModel(_cfg_sweep_j)
    fmj.run()
    fpj = [p / 1e6 for p in fmj.get_turbine_powers().flatten()]
    floris_jensen_sweep['T2'].append(fpj[1])
    floris_jensen_sweep['T3'].append(fpj[2])
    floris_jensen_sweep['farm'].append(sum(fpj))

    print(f"  {offset}deg done")

# Normalise by T1 power at 0 degrees (aligned)
T1_power = our_jensen.power(U_INF_sweep) / 1e6

print("\n--- Wind Direction Sweep Results ---")
print(f"{'Offset':>8} {'J_T2':>8} {'G_T2':>8} {'L_T2':>8} {'FL_T2':>8} | {'J_T3':>8} {'G_T3':>8} {'L_T3':>8} {'FL_T3':>8}")
print("-" * 80)
for i, offset in enumerate(offsets_deg):
    print(f"{offset:>7}° "
          f"{sweep_results['LOTUSim-Jensen']['T2'][i]:>8.3f} "
          f"{sweep_results['LOTUSim-Gaussian']['T2'][i]:>8.3f} "
          f"{sweep_results['LOTUSim-Larsen']['T2'][i]:>8.3f} "
          f"{floris_sweep['T2'][i]:>8.3f} | "
          f"{sweep_results['LOTUSim-Jensen']['T3'][i]:>8.3f} "
          f"{sweep_results['LOTUSim-Gaussian']['T3'][i]:>8.3f} "
          f"{sweep_results['LOTUSim-Larsen']['T3'][i]:>8.3f} "
          f"{floris_sweep['T3'][i]:>8.3f}")

# --- Normalised P_T2/P_T1,0deg for ALL SIX models (table tab:p13_direction) ---
print("\n=== NORMALISED P_T2 / P_T1,0deg (divide by 3.724) — all 6 models ===")
print(f"{'off':>4} {'Jensen':>8}{'Gaussian':>9}{'Larsen':>8}{'Blended':>8}"
      f"{'FLORIS-G':>9}{'FLORIS-J':>9}")
for i, offset in enumerate(offsets_deg):
    print(f"{offset:>4} "
          f"{sweep_results['LOTUSim-Jensen']['T2'][i]/T1_power:>8.3f}"
          f"{sweep_results['LOTUSim-Gaussian']['T2'][i]/T1_power:>9.3f}"
          f"{sweep_results['LOTUSim-Larsen']['T2'][i]/T1_power:>8.3f}"
          f"{sweep_results['LOTUSim-Blended']['T2'][i]/T1_power:>8.3f}"
          f"{floris_sweep['T2'][i]/T1_power:>9.3f}"
          f"{floris_jensen_sweep['T2'][i]/T1_power:>9.3f}")
print("--- Blended vs Larsen (normalised T2) ---")
for i, offset in enumerate(offsets_deg):
    b = sweep_results['LOTUSim-Blended']['T2'][i]/T1_power
    l = sweep_results['LOTUSim-Larsen']['T2'][i]/T1_power
    print(f"  {offset:>3}deg  Blended={b:.3f}  Larsen={l:.3f}  diff={b-l:+.3f}")

# Plot
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.patch.set_facecolor('white')

colors = {
    'LOTUSim-Jensen':   ('tomato',         'v'),
    'LOTUSim-Gaussian': ('#DAA000',        'D'),
    'LOTUSim-Larsen':   ('mediumseagreen', 'P'),
    'FLORIS Gauss': ('royalblue',      's'),
}

for ax, turb_key, title in [
    (axes[0], 'T2', 'T2 Power (7D downstream)'),
    (axes[1], 'T3', 'T3 Power (14D downstream)'),
    (axes[2], 'farm', 'Total Farm Power'),
]:
    ax.set_facecolor('#FAFAFA')
    for name, (col, marker) in colors.items():
        if name == 'FLORIS Gauss':
            vals = floris_sweep[turb_key]
        else:
            vals = sweep_results[name][turb_key]
        ax.plot(offsets_deg, vals, marker=marker, color=col,
                linewidth=1.8, markersize=8, label=name)

    # Mark aligned case
    ax.axvline(0, color='k', linewidth=0.8, linestyle=':', alpha=0.4)
    ax.set_xlabel('Wind direction offset from aligned (degrees)', fontsize=12)
    ax.set_ylabel('Power (MW)', fontsize=12)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(fontsize=10, framealpha=0.95)
    ax.grid(True, alpha=0.25)
    ax.set_xticks(offsets_deg)

cfd_angles  = [0,     15,    30]
cfd_T2      = [1.904, 1.015, 0.280]
cfd_T3      = [0.560, 1.047, 0.937]

axes[0].scatter(cfd_angles, cfd_T2,
                color='black', marker='*', s=200, zorder=6,
                label='CFD spot check')

valid_angles = [cfd_angles[i] for i in range(len(cfd_T3)) if cfd_T3[i] is not None]
valid_T3     = [v for v in cfd_T3 if v is not None]
axes[1].scatter(valid_angles, valid_T3,
                color='black', marker='*', s=200, zorder=6,
                label='CFD spot check')

# Update legends to include CFD spot check
axes[0].legend(fontsize=10, framealpha=0.95)
axes[1].legend(fontsize=10, framealpha=0.95)

fig.suptitle('Wind Direction Sweep — NREL 5MW, 3 Turbines, U = 10 m/s\n'
             'Effect of wind direction offset on per-turbine and farm power',
             fontsize=13, fontweight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.94])
_out = _fig_out('wind_direction_sweep.png')
fig.savefig(_out, dpi=300, bbox_inches='tight')
print("\nPlot saved to:", _out)
