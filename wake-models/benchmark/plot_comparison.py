import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Repo-relative paths: this script runs identically from any directory.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR  = os.path.join(REPO_ROOT, 'benchmark', 'data')
SUPP_DIR  = os.path.join(REPO_ROOT, 'figures', 'supplementary')
TURBINE_DIR = os.path.join(DATA_DIR, 'layout_a', 'turbines', '0')


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



    def power(self, wind_speed: float) -> float:

        if wind_speed < self.cut_in or wind_speed > self.cut_out:

            return 0.0

        else:

            area = np.pi * (self.diameter / 2.0) ** 2.0

            return 0.5 * self.air_density * area * self.cp * wind_speed ** 3



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

            deficits = []



            for j in range(i):

                x_j, y_j, z_j = turbines_sorted[j]

                delta_xz = np.array([x_i - x_j, z_i - z_j])

                x_dist = np.dot(delta_xz, w)

                lateral_dist = np.dot(delta_xz, w_perp)



                if debug:

                    print(f"  T{j}→T{i}: x_dist={x_dist:.1f} m, lateral={lateral_dist:.1f} m, "

                          f"wake_r={self.wake_radius(x_dist):.1f} m")



                if x_dist > 1e-9 and self.in_wake(x_dist, lateral_dist):

                    deficit = self.wake_deficit(ogWind, x_dist)

                    if deficit > 1e-6:

                        deficits.append(deficit)



            # Apply yaw factor to effective velocity — zero for pure east/west wind

            v_eff = self.combined_velocity(ogWind, deficits) * yaw_factor

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
class GaussianWakeModel:



    def __init__(self, diameter:float, ct:float, air_density:float = 1.225,cp: float = 0.35, cut_in: float = 5.0, cut_out: float = 25.0, k_y: float=0.05, k_z: float = 0.05, eps: float = 0.2):

        self.diameter = diameter

        self.ct = ct

        self.air_density = air_density

        self.cp = cp

        self.cut_in = cut_in

        self.cut_out = cut_out

        self.k_y = k_y # describes the width expansion of the wake downstream

        self.k_z = k_z # describes the height expansion of the wake downstream

        self.eps = eps # parameter linked to the starting width of the wake ratio to diameter



    def power(self, wind_speed:float) -> float:



        # cut in and cut out speeds for the wind turbines

        if wind_speed < self.cut_in or wind_speed > self.cut_out:

            return 0.0

        else:

            area = np.pi * (self.diameter / 2.0) ** 2

            return 0.5 * self.air_density * area * (wind_speed ** 3) * self.cp





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



    def gaussian_wake_deficit_full(self, ogWind: float, x_dist: float, y_dist: float, z_dist: float) -> float:

        if x_dist <= 0:

            return 0.0



        sigma_y = self.k_y * x_dist + self.eps * self.diameter

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

            deficits = []



            for j in range(i):

                x_j, y_j, z_j = turbines_sorted[j]



                # x-z plane only, same as Jensen

                delta_xz = np.array([x_i - x_j, z_i - z_j])



                # Downstream distance along wind direction

                x_dist = np.dot(delta_xz, w)



                # Cross-wind (lateral) distance in x-z plane

                lateral_dist = np.dot(delta_xz, w_perp)



                # Vertical distance — y is height, kept separate

                vertical_dist = y_i - y_j  # always 0 for your turbines, correct to keep for generality



                if debug:

                    print(f"\nT{j}→T{i}: x_dist={x_dist:.2f}, lateral={lateral_dist:.2f}, vertical={vertical_dist:.2f}")



                if x_dist <= 1e-9:

                    continue



                deficit = self.gaussian_wake_deficit_full(

                    ogWind,

                    x_dist,

                    lateral_dist,  # horizontal cross-wind spread (sigma_y)

                    vertical_dist  # vertical spread (sigma_z), 0 when same height

                )



                if debug:

                    print(f"  deficit={deficit:.4f}")



                if deficit > 1e-6:

                    deficits.append(deficit)



            u_eff = self.combined_velocity(ogWind, deficits) * yaw_factor

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

        self.cut_in_speed = float(cut_in_speed)

        self.cut_out_speed = float(cut_out_speed)

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



    def power(self, wind_speed):

        if wind_speed < self.cut_in_speed or wind_speed >= self.cut_out_speed:

            return 0.0



        p = 0.5 * self.air_density * self.area * self.cp * wind_speed**3



        if self.rated_power is not None:

            p = min(p, self.rated_power)



        return p





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



    def velocity_at_point(self, x, r, U_inf):

        if x <= 0:

            return U_inf



        rw = self.wake_radius(x)

        if abs(r) > rw:

            return U_inf



        return max(0.0, U_inf - self.deficit(x, r, U_inf))



    def rotational_speed_rpm(self, wind_speed: float, tip_speed_ratio: float = 7.0) -> float:

        if wind_speed < self.cut_in_speed or wind_speed >= self.cut_out_speed:

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

        downstream, crosswind, height, U_inf, yaw_factor = self._project_coordinates(turbines, wind_vector)



        target_down = downstream[turbine_index]

        target_cross = crosswind[turbine_index]

        target_height = height[turbine_index]



        deficits = []



        for j in range(len(turbines)):

            if j == turbine_index:

                continue



            dx = target_down - downstream[j]

            dy = target_cross - crosswind[j]

            dheight = target_height - height[j]



            if dx <= 0:

                continue



            r = np.sqrt(dy ** 2 + dheight ** 2)



            local_velocity = self.velocity_at_point(dx, r, U_inf)

            local_deficit = max(0.0, U_inf - local_velocity)



            if local_deficit > 0:

                deficits.append(local_deficit)



        if deficits:

            combined_deficit = np.sqrt(np.sum(np.array(deficits) ** 2))

        else:

            combined_deficit = 0.0



        # Apply yaw factor — zero for pure east/west wind

        return max(0.0, U_inf - combined_deficit) * yaw_factor



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
# FARM SETUP - must match your OpenFOAM case
# =============================================================================

# NREL 5MW parameters
DIAMETER   = 126.0   # m  (rotor diameter)
CT         = 0.8     # thrust coefficient
CP         = 0.594    # power coefficient (engineering models use this)
RHO        = 1.225   # kg/m3
U_INF      = 10.0    # m/s  (matches freeStreamVelocity in fvOptions)
HUB_HEIGHT = 90.0    # m

# 3 turbines in a row, 7D spacing, wind blowing in +z direction
# Coordinates: (x_lateral, y_height, z_downstream)
D = DIAMETER
turbines = [
    (0,   HUB_HEIGHT,    0),      # Turbine 1 - upstream
    (0,   HUB_HEIGHT,  882),      # Turbine 2 - 7D downstream
    (0,   HUB_HEIGHT, 1764),      # Turbine 3 - 14D downstream
]

# Wind vector [x, z] — pure north (wind travels in +z direction)
wind_vector = [0.0, U_INF]

# =============================================================================
# RUN ENGINEERING MODELS
# =============================================================================

def get_powers_MW(model, turbines, wind_vector, model_type='larsen'):
    """Run a wake model and return per-turbine power in MW."""
    if model_type in ('jensen', 'gaussian'):
        _, velocities, _ = model.wind_speeds_full(turbines, U_INF, wind_vector)
    else:  # larsen
        _, velocities, _ = model.wind_speeds_full(turbines, wind_vector)
    powers_MW = [model.power(v) / 1e6 for v in velocities]
    return powers_MW, velocities
# Jensen
jensen_model = JensenWakeModel(
    diameter=DIAMETER, ct=CT, air_density=RHO, cp=CP,
    cut_in=3.0, cut_out=25.0,
    kw=0.025    # slower expansion = more wake loss downstream, originally 0.025
)
jensen_powers, jensen_velocities = get_powers_MW(jensen_model, turbines, wind_vector, model_type='jensen')
# Gaussian
gaussian_model = GaussianWakeModel(
    diameter=DIAMETER, ct=CT, air_density=RHO, cp=CP,
    cut_in=3.0, cut_out=25.0,
    k_y=0.03,   # reduced for offshore (less turbulent than onshore)
    k_z=0.03,
    eps=0.25
)
_, gaussian_velocities, _ = gaussian_model.wind_speeds_full(turbines, U_INF, wind_vector)
gaussian_powers = [gaussian_model.power(v) / 1e6 for v in gaussian_velocities]

# Larsen
larsen_model = LarsenWakeModel(
    diameter=DIAMETER, ct=0.8, air_density=RHO, cp=CP,
    ambient_ti=0.07, cut_in_speed=3.0, cut_out_speed=25.0
)
larsen_powers, larsen_velocities = get_powers_MW(larsen_model, turbines, wind_vector, model_type='larsen')

# =============================================================================
# LOAD CFD RESULTS
# =============================================================================

# Rotor area and available power
A = np.pi * (DIAMETER / 2) ** 2
P_avail = 0.5 * RHO * A * U_INF**3  # W

t1 = pd.read_csv(os.path.join(TURBINE_DIR, 'turbine1.csv'))
t2 = pd.read_csv(os.path.join(TURBINE_DIR, 'turbine2.csv'))
t3 = pd.read_csv(os.path.join(TURBINE_DIR, 'turbine3.csv'))

t1['power_MW'] = t1['cp'] * P_avail / 1e6
t2['power_MW'] = t2['cp'] * P_avail / 1e6
t3['power_MW'] = t3['cp'] * P_avail / 1e6

# Average over second half of simulation (after flow develops)
half = len(t1) // 2
cfd_powers = [
    t1['power_MW'].iloc[half:].mean(),
    t2['power_MW'].iloc[half:].mean(),
    t3['power_MW'].iloc[half:].mean(),
]

# =============================================================================
# PRINT COMPARISON TABLE
# =============================================================================

labels = ['Turbine 1 (upstream)', 'Turbine 2 (7D)', 'Turbine 3 (14D)']

print("\n" + "="*75)
print(f"{'WAKE MODEL COMPARISON — NREL 5MW, 3 Turbines, U=10 m/s':^75}")
print("="*75)
print(f"\n{'Turbine':<22} {'CFD (MW)':>10} {'Jensen (MW)':>12} {'Gaussian (MW)':>14} {'Larsen (MW)':>12}")
print("-"*75)
for i, label in enumerate(labels):
    print(f"{label:<22} {cfd_powers[i]:>10.3f} {jensen_powers[i]:>12.3f} "
          f"{gaussian_powers[i]:>14.3f} {larsen_powers[i]:>12.3f}")

print("-"*75)
print(f"{'Total farm power':<22} {sum(cfd_powers):>10.3f} {sum(jensen_powers):>12.3f} "
      f"{sum(gaussian_powers):>14.3f} {sum(larsen_powers):>12.3f}")

print("\n--- Wake losses relative to Turbine 1 ---")
print(f"\n{'Turbine':<22} {'CFD':>10} {'Jensen':>12} {'Gaussian':>14} {'Larsen':>12}")
print("-"*75)
for i, label in enumerate(labels):
    cfd_loss     = (1 - cfd_powers[i]     / cfd_powers[0])     * 100
    jensen_loss  = (1 - jensen_powers[i]  / jensen_powers[0])  * 100
    gauss_loss   = (1 - gaussian_powers[i]/ gaussian_powers[0])* 100
    larsen_loss  = (1 - larsen_powers[i]  / larsen_powers[0])  * 100
    print(f"{label:<22} {cfd_loss:>9.1f}% {jensen_loss:>11.1f}% "
          f"{gauss_loss:>13.1f}% {larsen_loss:>11.1f}%")

print("\n--- Error vs CFD (MW) ---")
print(f"\n{'Turbine':<22} {'Jensen err':>12} {'Gaussian err':>14} {'Larsen err':>12}")
print("-"*65)
for i, label in enumerate(labels):
    j_err = jensen_powers[i]  - cfd_powers[i]
    g_err = gaussian_powers[i]- cfd_powers[i]
    l_err = larsen_powers[i]  - cfd_powers[i]
    print(f"{label:<22} {j_err:>+11.3f}  {g_err:>+13.3f}  {l_err:>+11.3f}")

# =============================================================================
# PLOTS
# =============================================================================

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Wake Model Comparison vs CFD Reference\nNREL 5MW, 3 Turbines, 7D Spacing, U=10 m/s',
             fontsize=13, fontweight='bold')

turbine_labels = ['T1\n(upstream)', 'T2\n(7D)', 'T3\n(14D)']
x = np.arange(3)
w = 0.18
colors = {'CFD': 'steelblue', 'Jensen': 'tomato', 'Gaussian': 'gold', 'Larsen': 'mediumseagreen'}

# --- Plot 1: Absolute power ---
ax = axes[0]
ax.bar(x - 1.5*w, cfd_powers,      w, label='CFD',      color=colors['CFD'],      edgecolor='k', linewidth=0.5)
ax.bar(x - 0.5*w, jensen_powers,   w, label='Jensen',   color=colors['Jensen'],   edgecolor='k', linewidth=0.5)
ax.bar(x + 0.5*w, gaussian_powers, w, label='Gaussian', color=colors['Gaussian'], edgecolor='k', linewidth=0.5)
ax.bar(x + 1.5*w, larsen_powers,   w, label='Larsen',   color=colors['Larsen'],   edgecolor='k', linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels(turbine_labels)
ax.set_ylabel('Power (MW)')
ax.set_title('Absolute Power Output')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(0, max(cfd_powers)*1.2)

# --- Plot 2: Normalised power (P/P_T1) ---
ax = axes[1]
cfd_norm     = [p/cfd_powers[0]      for p in cfd_powers]
jensen_norm  = [p/jensen_powers[0]   for p in jensen_powers]
gauss_norm   = [p/gaussian_powers[0] for p in gaussian_powers]
larsen_norm  = [p/larsen_powers[0]   for p in larsen_powers]

ax.bar(x - 1.5*w, cfd_norm,     w, label='CFD',      color=colors['CFD'],      edgecolor='k', linewidth=0.5)
ax.bar(x - 0.5*w, jensen_norm,  w, label='Jensen',   color=colors['Jensen'],   edgecolor='k', linewidth=0.5)
ax.bar(x + 0.5*w, gauss_norm,   w, label='Gaussian', color=colors['Gaussian'], edgecolor='k', linewidth=0.5)
ax.bar(x + 1.5*w, larsen_norm,  w, label='Larsen',   color=colors['Larsen'],   edgecolor='k', linewidth=0.5)
ax.axhline(1.0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
ax.set_xticks(x)
ax.set_xticklabels(turbine_labels)
ax.set_ylabel('Normalised Power (P / P_T1)')
ax.set_title('Normalised Power (wake loss profile)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(0, 1.15)

# --- Plot 3: CFD Cp time series ---
ax = axes[2]
ax.plot(t1['time'], t1['cp'], label='T1 (upstream)', color='blue',   linewidth=1.2)
ax.plot(t2['time'], t2['cp'], label='T2 (7D)',       color='orange', linewidth=1.2)
ax.plot(t3['time'], t3['cp'], label='T3 (14D)',      color='green',  linewidth=1.2)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Power Coefficient (Cp)')
ax.set_title('CFD Cp Time Series')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
os.makedirs(SUPP_DIR, exist_ok=True)
_out = os.path.join(SUPP_DIR, 'wake_model_comparison.png')
plt.savefig(_out, dpi=150, bbox_inches='tight')
print('\nFigure saved:', _out)
if __name__ == "__main__":
    plt.show()
