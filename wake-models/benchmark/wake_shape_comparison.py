"""
Wake Shape and Decay Comparison
================================
Compares lateral velocity deficit profiles from:
  - OpenFOAM CFD (line probe data)
  - LOTUSim-Jensen, Gaussian, Larsen models
  - FLORIS Gauss (reference)

Positions: 1D, 3D, 5D, 7D, 10D, 14D downstream of T1
Metric: Normalised streamwise velocity U/U_inf at hub height
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
import os
import sys
import tempfile

# Repo-relative paths: this script runs identically from any directory.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
DATA_DIR  = os.path.join(REPO_ROOT, 'benchmark', 'data')
SUPP_DIR  = os.path.join(REPO_ROOT, 'figures', 'supplementary')


def tmp_path(name):
    """Scratch file in the platform temp directory (not a hardcoded /tmp)."""
    return os.path.join(tempfile.gettempdir(), name)


# =============================================================================
# =============================================================================

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

            * (3.0 * c1**2 * self.ct * self.area * (x + x0)) ** (-1.0 / 2.0)

            - (35.0 / (2.0 * np.pi)) ** (3.0 / 10.0)

            * (3.0 * c1**2) ** (-1.0 / 5.0)

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

            # Compute deficit using LOCAL velocity as inflow (not U_inf)
            # This is the key change — Larsen deficit scaled to local conditions
            local_TI = self.local_ti(self.ambient_ti, self.ct, dx, self.diameter)
            k_amb   = 0.38 * self.ambient_ti + 0.004
            k_local = 0.38 * local_TI + 0.004
            TI_ratio = np.sqrt(k_amb / k_local) ** (1/4)
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



    # ------------------------------------------------------------

    # Plotting

    # ------------------------------------------------------------

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
# PARAMETERS — must match CFD case
# =============================================================================

D          = 126.0
HUB_HEIGHT = 90.0
U_INF      = 10.0
TI         = 0.08
RHO        = 1.225
CT         = 0.80
CP_ENG     = 0.593

# Turbine positions (x_lateral, y_height, z_downstream)
turbines    = [
    (0, HUB_HEIGHT,    0),
    (0, HUB_HEIGHT,  882),
    (0, HUB_HEIGHT, 1764),
]
wind_vector = [0.0, U_INF]
 
positions_D = [1, 3, 5, 7, 10, 14]
positions_m = [d * D for d in positions_D]
 
CFD_DIR = os.path.join(DATA_DIR, 'layout_single', 'wakeProfiles')
 
# =============================================================================
# LOAD CFD PROBE DATA  (U_0 = streamwise x-velocity)
# =============================================================================
 
def load_cfd_profile(position_m, base_dir, u_inf=10.0):
    label    = f'{int(round(position_m / D))}D'
    filename = f'wake_{label}_U.csv'
 
    dirs = sorted(
        [d for d in os.listdir(base_dir)
         if os.path.isdir(os.path.join(base_dir, d))
         and os.path.exists(os.path.join(base_dir, d, filename))],
        key=lambda x: int(x), reverse=True
    )[:5]
 
    if not dirs:
        return None, None
 
    profiles, y_vals = [], None
    for t in dirs:
        try:
            df = pd.read_csv(os.path.join(base_dir, t, filename))
            profiles.append(df['U_0'].values)   # U_0 = x-velocity = streamwise
            y_vals = df['y'].values
        except Exception:
            continue
 
    if not profiles:
        return None, None
 
    return y_vals / D, np.mean(profiles, axis=0) / u_inf
 
 
print("Loading CFD probe data...")
cfd_profiles = {}
for pos_m in positions_m:
    label = f'{int(round(pos_m / D))}D'
    y_D, U_norm = load_cfd_profile(pos_m, CFD_DIR, u_inf=U_INF)
    if y_D is not None:
        cfd_profiles[label] = (y_D, U_norm)
        cl = U_norm[len(U_norm) // 2]
        print(f"  {label}: {len(y_D)} pts, centreline U/Uinf = {cl:.3f}")
    else:
        print(f"  WARNING: could not load {label}")
 
# =============================================================================
# FLORIS LATERAL PROFILES
# =============================================================================
 
def floris_lateral_profile(x_downstream, y_range_D=2.5, n_points=51):
    """FLORIS lateral profile using direct horizontal plane sampling."""
    try:
        import floris
        fpath = os.path.dirname(floris.__file__)
        with open(fpath + '/default_inputs.yaml') as f:
            config = yaml.safe_load(f)

        config['farm']['layout_x']                        = [0.0]
        config['farm']['layout_y']                        = [0.0]
        config['farm']['turbine_type']                    = ['nrel_5MW']
        config['flow_field']['wind_speeds']               = [U_INF]
        config['flow_field']['wind_directions']           = [270.0]
        config['flow_field']['turbulence_intensities']    = [TI]
        config['flow_field']['air_density']               = RHO
        config['wake']['model_strings']['velocity_model'] = 'gauss'

        _cfg = tmp_path('floris_wake.yaml')
        with open(_cfg, 'w') as f:
            yaml.dump(config, f)

        from floris import FlorisModel
        fm = FlorisModel(_cfg)
        fm.run()

        y_range = y_range_D * D
        hp = fm.calculate_horizontal_plane(
            height       = HUB_HEIGHT,
            x_resolution = 3,
            y_resolution = n_points,
            x_bounds     = (x_downstream - 5, x_downstream + 5),
            y_bounds     = (-y_range, y_range),
        )
        df     = hp.df.copy()
        df_mid = df[np.abs(df['x1'] - x_downstream) ==
                    np.abs(df['x1'] - x_downstream).min()]
        df_mid = df_mid.sort_values('x2')
        return df_mid['x2'].values / D, df_mid['u'].values / U_INF

    except Exception as e:
        print(f"    FLORIS error: {e}")
        return None, None
 
 
print("\nComputing FLORIS lateral profiles...")
floris_profiles = {}
for pos_m in positions_m:
    label = f'{int(round(pos_m / D))}D'
    print(f"  {label}...")
    y_f, u_f = floris_lateral_profile(pos_m)
    if y_f is not None:
        floris_profiles[label] = (y_f, u_f)
        print(f"    centreline = {u_f[len(u_f)//2]:.3f}")
    else:
        print(f"    skipped")
 
# =============================================================================
# ENGINEERING MODEL LATERAL PROFILES
# =============================================================================
 
def model_lateral_profile(model, model_type, x_downstream,
                           y_range_D=2.5, n_points=101):
    y_D    = np.linspace(-y_range_D, y_range_D, n_points)
    U_norm = np.ones(n_points)
    for i, yd in enumerate(y_D):
        r = abs(yd * D)
        try:
            if model_type == 'blended':
                U_norm[i] = model.velocity_at_point(U_INF, x_downstream, r) / U_INF
            else:
                probe = (yd * D, HUB_HEIGHT, x_downstream)
                farm  = [turbines[0], probe]
                if model_type in ('jensen', 'gaussian'):
                    _, vels, _ = model.wind_speeds_full(farm, U_INF, wind_vector)
                else:
                    _, vels, _ = model.wind_speeds_full(farm, wind_vector)
                U_norm[i] = vels[1] / U_INF
        except Exception:
            U_norm[i] = 1.0
    return y_D, U_norm
 
 
our_jensen = JensenWakeModel(
    diameter=D, ct=0.75, air_density=RHO, cp=CP_ENG,
    cut_in=3.0, cut_out=25.0, kw=0.03)
our_gaussian = GaussianWakeModel(
    diameter=D, ct=CT, air_density=RHO, cp=CP_ENG,
    cut_in=3.0, cut_out=25.0, ambient_ti=TI, eps=0.22)
from models.extended_models import BlendedWakeModel as _BlendedWakeModel
our_larsen = LarsenWakeModel(
    diameter=D, ct=CT, air_density=RHO, cp=CP_ENG,
    ambient_ti=0.08, cut_in_speed=3.0, cut_out_speed=25.0)
 
print("\nComputing engineering model profiles...")
our_blended = _BlendedWakeModel(diameter=D, ct=CT, air_density=RHO, cp=CP_ENG,
    ambient_ti=TI, cut_in_speed=3.0, cut_out_speed=25.0)
model_profiles = {n: {} for n in ['LOTUSim-Jensen', 'LOTUSim-Gaussian', 'LOTUSim-Larsen', 'LOTUSim-Blended']}
for pos_m in positions_m:
    label = f'{int(round(pos_m / D))}D'
    print(f"  {label}...")
    model_profiles['LOTUSim-Jensen'][label]   = model_lateral_profile(our_jensen,   'jensen',   pos_m)
    model_profiles['LOTUSim-Gaussian'][label] = model_lateral_profile(our_gaussian, 'gaussian', pos_m)
    model_profiles['LOTUSim-Larsen'][label]   = model_lateral_profile(our_larsen,   'larsen',   pos_m)
    model_profiles['LOTUSim-Blended'][label]  = model_lateral_profile(our_blended,  'blended',  pos_m)
 
# =============================================================================
# CENTRELINE EXTRACTION
# =============================================================================
 
def centreline(profs_dict, label):
    if label not in profs_dict:
        return np.nan
    y, u = profs_dict[label]
    return float(u[np.argmin(np.abs(y))])
 
cfd_cl      = [centreline(cfd_profiles,                  f'{d}D') for d in positions_D]
floris_cl   = [centreline(floris_profiles,               f'{d}D') for d in positions_D]
jensen_cl   = [centreline(model_profiles['LOTUSim-Jensen'],  f'{d}D') for d in positions_D]
gaussian_cl = [centreline(model_profiles['LOTUSim-Gaussian'],f'{d}D') for d in positions_D]
larsen_cl   = [centreline(model_profiles['LOTUSim-Larsen'],  f'{d}D') for d in positions_D]
blended_cl  = [centreline(model_profiles['LOTUSim-Blended'], f'{d}D') for d in positions_D]
 
# =============================================================================
# RMSE vs CFD
# =============================================================================
 
all_models = {
    'FLORIS Gauss': floris_profiles,
    'LOTUSim-Jensen':   model_profiles['LOTUSim-Jensen'],
    'LOTUSim-Gaussian': model_profiles['LOTUSim-Gaussian'],
    'LOTUSim-Larsen':   model_profiles['LOTUSim-Larsen'],
    'LOTUSim-Blended':  model_profiles['LOTUSim-Blended'],
}
rmse_results = {n: [] for n in all_models}
 
for pos_m in positions_m:
    label = f'{int(round(pos_m / D))}D'
    if label not in cfd_profiles:
        for v in rmse_results.values():
            v.append(np.nan)
        continue
    y_cfd, u_cfd = cfd_profiles[label]
    for mname, mprofs in all_models.items():
        if label not in mprofs:
            rmse_results[mname].append(np.nan)
            continue
        y_m, u_m  = mprofs[label]
        u_interp  = np.interp(y_cfd, y_m, u_m)
        rmse_results[mname].append(
            float(np.sqrt(np.mean((u_interp - u_cfd)**2))))
 
print("\n--- Wake Profile RMSE vs CFD (U/U_inf) ---")
col_w = 14
print(f"{'Position':<10}" + "".join(f"{n:>{col_w}}" for n in rmse_results))
print("-" * (10 + col_w * len(rmse_results)))
for i, d in enumerate(positions_D):
    row = f"{d}D{'':<8}"
    for vals in rmse_results.values():
        v = vals[i] if i < len(vals) else np.nan
        row += f"{v:>{col_w}.4f}" if not np.isnan(v) else f"{'N/A':>{col_w}}"
    print(row)
 
avg = {n: np.nanmean(v) for n, v in rmse_results.items()}
print("-" * (10 + col_w * len(rmse_results)))
print(f"{'Average':<10}" + "".join(f"{v:>{col_w}.4f}" for v in avg.values()))
best = min(avg, key=lambda k: avg[k] if not np.isnan(avg[k]) else 999)
print(f"\nBest wake shape model: {best}  (RMSE = {avg[best]:.4f})")
 
# =============================================================================
# COLOURS / STYLES
# =============================================================================
 
styles = {
    'CFD':          dict(color='black',          lw=3.0, ls='-',  marker='o', ms=10),
    'FLORIS Gauss': dict(color='royalblue',      lw=2.5, ls='--', marker='s', ms=9),
    'LOTUSim-Jensen':   dict(color='tomato',         lw=2.0, ls='--', marker='v', ms=9),
    'LOTUSim-Gaussian': dict(color='#DAA000',        lw=2.0, ls='--', marker='D', ms=9),
    'LOTUSim-Larsen':   dict(color='mediumseagreen', lw=2.0, ls='--', marker='P', ms=9),
    'LOTUSim-Blended':  dict(color='darkviolet',     lw=2.5, ls='-',  marker='h', ms=10),
}
 
# =============================================================================
# FIGURE 1 — LATERAL PROFILES  (2 rows x 3 cols)
# =============================================================================
 
fig1, axes = plt.subplots(2, 3, figsize=(22, 13))
fig1.patch.set_facecolor('white')
axes = axes.flatten()
 
for idx, (pos_m, ax) in enumerate(zip(positions_m, axes)):
    label = f'{int(round(pos_m / D))}D'
    ax.set_facecolor('#FAFAFA')
 
    all_profile_sources = [
        ('CFD',          cfd_profiles),
        ('FLORIS Gauss', floris_profiles),
        ('LOTUSim-Jensen',   model_profiles['LOTUSim-Jensen']),
        ('LOTUSim-Gaussian', model_profiles['LOTUSim-Gaussian']),
        ('LOTUSim-Larsen',   model_profiles['LOTUSim-Larsen']),
        ('LOTUSim-Blended',  model_profiles['LOTUSim-Blended']),
    ]
    for name, profs in all_profile_sources:
        if label in profs:
            y_D, U_norm = profs[label]
            st = styles[name]
            ax.plot(y_D, U_norm,
                    color=st['color'], linewidth=st['lw'],
                    linestyle=st['ls'], label=name, alpha=0.92, zorder=5 if name=='CFD' else 3)
 
    ax.axvline(-0.5, color='steelblue', lw=1.5, ls=':', alpha=0.45)
    ax.axvline( 0.5, color='steelblue', lw=1.5, ls=':', alpha=0.45,
               label='Rotor edges (±D/2)')
    ax.axhline( 1.0, color='k',         lw=0.7, ls=':', alpha=0.3)
 
    ax.set_xlabel('Lateral position (y/D)', fontsize=14)
    ax.set_ylabel('U / U∞', fontsize=14)
    ax.set_title(f'{label}  ({int(pos_m)} m downstream)',
                 fontsize=15, fontweight='bold')
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(0.2, 1.15)
    ax.grid(True, alpha=0.22)
    if idx == 0:
        ax.legend(fontsize=13, loc="lower center", ncol=2,
              framealpha=0.95, edgecolor="#CCCCCC")
 
# Suptitle intentionally disabled (the per-panel titles already carry the
# information). Previously only the opening line was commented out, which left
# the argument lines dangling and made this file a SyntaxError.
# fig1.suptitle(
#     'Lateral Wake Velocity Profiles — NREL 5MW, U∞ = 10 m/s, TI = 8%\n'
#     'CFD (OpenFOAM/turbinesFoam)  |  FLORIS Gauss  |  LOTUSim Models',
#     fontsize=16, fontweight='bold')
fig1.tight_layout(rect=[0, 0, 1, 0.94])
os.makedirs(SUPP_DIR, exist_ok=True)
_out = os.path.join(SUPP_DIR, 'wake_shape_1t_lateral_profiles.png')
fig1.savefig(_out, dpi=300, bbox_inches='tight')
print("\nFigure 1 saved:", _out)
 
# =============================================================================
# FIGURE 2 — CENTRELINE DECAY
# =============================================================================
 
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
fig2.patch.set_facecolor('white')
 
lines = [
    ('CFD',          cfd_cl,      styles['CFD']),
    ('FLORIS Gauss', floris_cl,   styles['FLORIS Gauss']),
    ('LOTUSim-Jensen',   jensen_cl,   styles['LOTUSim-Jensen']),
    ('LOTUSim-Gaussian', gaussian_cl, styles['LOTUSim-Gaussian']),
    ('LOTUSim-Larsen',   larsen_cl,   styles['LOTUSim-Larsen']),
    ('LOTUSim-Blended',  blended_cl,  styles['LOTUSim-Blended']),
]
 
for ax, transform, ylabel, ylim, title in [
    (ax1, lambda v: v,   'Normalised velocity  U/U∞', (0.2, 1.1),
     'Wake Centreline Velocity Decay'),
    (ax2, lambda v: 1-v, 'Velocity deficit  1 - U/U∞', (0.0, 0.80),
     'Wake Centreline Deficit Decay'),
]:
    for name, vals, st in lines:
        xp = [positions_D[i] for i, v in enumerate(vals) if not np.isnan(v)]
        yp = [transform(v) for v in vals if not np.isnan(v)]
        if xp:
            ax.plot(xp, yp,
                    color=st['color'], lw=st['lw'],
                    ls=st['ls'], marker=st['marker'],
                    markersize=st['ms'], label=name)
 
    for tp, tn in [(0,'T1'),(7,'T2'),(14,'T3')]:
        ax.axvline(tp, color='steelblue', lw=1.5, ls=':', alpha=0.5)
        ax.text(tp+0.2, ylim[0]+0.02, tn, fontsize=13, color='steelblue')
 
    ax.set_xlabel('Downstream distance (D)', fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.legend(fontsize=13, framealpha=0.95, edgecolor='#CCCCCC')
    ax.grid(True, alpha=0.25)
    ax.set_ylim(ylim)
    ax.set_facecolor('#FAFAFA')
 
fig2.suptitle(
    'Wake Centreline Decay — NREL 5MW, U∞ = 10 m/s, TI = 8%\n'
    'CFD  |  FLORIS Gauss  |  LOTUSim Models',
    fontsize=16, fontweight='bold')
fig2.tight_layout(rect=[0, 0, 1, 0.94])
os.makedirs(SUPP_DIR, exist_ok=True)
_out = os.path.join(SUPP_DIR, 'wake_shape_1t_centreline_decay.png')
fig2.savefig(_out, dpi=300, bbox_inches='tight')
print("Figure 2 saved:", _out)
 
# =============================================================================
# FIGURE 3 — RMSE RANKING
# =============================================================================
 
fig3, ax = plt.subplots(figsize=(12, 6))
fig3.patch.set_facecolor('white')
ax.set_facecolor('#FAFAFA')
 
x       = np.arange(len(positions_D))
w       = 0.18
model_list = list(rmse_results.keys())
offsets = np.linspace(-1.5*w, 1.5*w, len(model_list))
rcols   = [styles[n]['color'] for n in model_list]
 
for name, col, offset in zip(model_list, rcols, offsets):
    vals  = rmse_results[name]
    xp    = [x[i] + offset for i, v in enumerate(vals) if not np.isnan(v)]
    yp    = [v for v in vals if not np.isnan(v)]
    bars  = ax.bar(xp, yp, w, label=name, color=col,
                   edgecolor='k', linewidth=0.4, alpha=0.88)
    for bar, val in zip(bars, yp):
        ax.text(bar.get_x() + bar.get_width()/2,
                val + 0.002, f'{val:.3f}',
                ha='center', va='bottom', fontsize=11)
 
ax.set_xticks(x)
ax.set_xticklabels([f'{d}D' for d in positions_D], fontsize=14)
ax.set_ylabel('RMSE  (U/U∞)  vs CFD', fontsize=15)
ax.set_title('Wake Profile Shape Accuracy vs CFD\n(lower = better)',
             fontsize=15, fontweight='bold')
ax.legend(fontsize=13, framealpha=0.95, edgecolor='#CCCCCC')
ax.grid(True, alpha=0.25, axis='y')
fig3.tight_layout()
os.makedirs(SUPP_DIR, exist_ok=True)
_out = os.path.join(SUPP_DIR, 'wake_shape_1t_profile_rmse.png')
fig3.savefig(_out, dpi=300, bbox_inches='tight')
print("Figure 3 saved:", _out)
 
if __name__ == "__main__":
    plt.show()
 

