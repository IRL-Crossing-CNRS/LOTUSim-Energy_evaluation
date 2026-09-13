"""
BlendedWakeModel
================
Distance-weighted, Ct-dependent blend of Larsen and a CFD-calibrated
Gaussian wake model for spatial wake field applications.

Reference: calibrated against NREL 5MW single-turbine OpenFOAM v8
           turbinesFoam actuator line simulation at 8, 10, 12 m/s.

Recommended use: spatial velocity field, wake visualisation, velocity
gradient computation, drone hazard zone assessment.

Key results from the benchmark:
    Gradient RMSE: <5% error at all positions 1D-14D (10 m/s, Ct=0.75)
    Multi-speed:   max errors 15.3% (8 m/s) and 19.4% (12 m/s)
    Hold-out test: sigma recovered within 3.9%
    Farm compounding: reduces gradient error from 65% to 14-15% at 7D-14D
"""

import numpy as np
from .larsen import LarsenWakeModel


class BlendedWakeModel:
    """
    Distance-weighted, Ct-dependent blend of Larsen and a CFD-calibrated
    Gaussian model. Recommended as the spatial/wake-field model: velocity
    gradient errors stay below 20% across the full operational wind speed
    range, where all other models exceed 50%.

    Blend weight (favours Larsen near-wake, Gaussian far-wake):
        x/D <= 1:  100% Larsen
        x/D >  1:  w(x) = clip(0.40 + 0.60*exp(-1.5*(x/D-1)/transition), 0.40, 1.0)
        transition = 1.5 * Ct / 0.75

    Calibrated Gaussian sigma parameters fitted to single-turbine CFD
    at 8, 10, 12 m/s (Ct = 0.80, 0.75, 0.53):
        sigma_0(Ct)   = 0.0939*Ct + 0.3286  (R=0.85)
        sigma_inf(Ct) = 0.0875*Ct + 0.3453  (R=0.89)
        sigma(x) = sigma_inf - (sigma_inf - sigma_0) * exp(-2.796 * x/D)
    """

    SIGMA_0_SLOPE   = 0.0939
    SIGMA_0_INTER   = 0.3286
    SIGMA_INF_SLOPE = 0.0875
    SIGMA_INF_INTER = 0.3453
    DECAY           = 2.796

    def __init__(self, diameter, ct, air_density=1.225, cp=0.498,
                 ambient_ti=0.08, cut_in_speed=3.0, cut_out_speed=25.0,
                 eps=0.22):
        self.diameter    = diameter
        self.ct          = ct
        self.air_density = air_density
        self.cp          = cp
        self.ambient_ti  = ambient_ti
        self.cut_in      = cut_in_speed
        self.cut_out     = cut_out_speed
        self.eps         = eps
        self.larsen = LarsenWakeModel(
            diameter=diameter, ct=ct, air_density=air_density, cp=cp,
            ambient_ti=ambient_ti, cut_in_speed=cut_in_speed,
            cut_out_speed=cut_out_speed)

    def calibrated_sigma(self, x):
        """Ct-dependent calibrated wake width sigma(x). Valid Ct 0.40-0.90."""
        x_D         = x / self.diameter
        sigma_0_D   = self.SIGMA_0_SLOPE   * self.ct + self.SIGMA_0_INTER
        sigma_inf_D = self.SIGMA_INF_SLOPE * self.ct + self.SIGMA_INF_INTER
        sigma_D     = sigma_inf_D - (sigma_inf_D - sigma_0_D) * \
                      np.exp(-self.DECAY * x_D)
        return sigma_D * self.diameter

    def gaussian_velocity_at_point(self, U_inf, x, r):
        """Calibrated Gaussian velocity component of the blend."""
        if x <= 0:
            return U_inf
        sigma   = self.calibrated_sigma(x)
        C       = 1 - np.sqrt(max(0, 1 - self.ct /
                  (8 * sigma ** 2 / self.diameter ** 2)))
        deficit = U_inf * C * np.exp(-r ** 2 / (2 * sigma ** 2))
        return max(0.0, U_inf - deficit)

    def blend_weight(self, x, ct=None):
        if ct is None:
            ct = self.ct
        x_D        = x / self.diameter
        transition = 1.5 * ct / 0.75
        w          = 0.40 + 0.60 * np.exp(-1.5 * (x_D - 1.0) / transition)
        return np.clip(w, 0.40, 1.0)

    def velocity_at_point(self, U_inf, x, r):
        """Blended velocity at (x, r): w*Larsen + (1-w)*CalibratedGaussian."""
        if x <= 0:
            return U_inf
        w   = self.blend_weight(x)
        u_l = self.larsen.velocity_at_point(U_inf, x, r)
        u_g = self.gaussian_velocity_at_point(U_inf, x, r)
        return w * u_l + (1 - w) * u_g

    def ti_at_point(self, x, r):
        return self.larsen.ti_at_point(x, r)

    def velocity_gradient_at_point(self, U_inf, x, r, dr=1.0):
        """Lateral velocity gradient dU/dr at (x, r). Units: s^-1."""
        U_plus  = self.velocity_at_point(U_inf, x, r + dr)
        U_minus = self.velocity_at_point(U_inf, x, max(0.0, r - dr))
        return (U_plus - U_minus) / (2 * dr)

    def wake_hazard_zone(self, U_inf, x, r,
                         deficit_caution=0.15, deficit_restricted=0.25,
                         ti_caution=0.12, ti_restricted=0.18):
        """Classify point for drone operations: 0=safe, 1=caution, 2=restricted."""
        u_local  = self.velocity_at_point(U_inf, x, r)
        deficit  = 1.0 - u_local / U_inf
        ti_local = self.ti_at_point(x, r)
        if deficit > deficit_restricted or ti_local > ti_restricted:
            return 2
        elif deficit > deficit_caution or ti_local > ti_caution:
            return 1
        return 0

    def farm_velocity_at_point(self, U_inf, x_query, y_query, turbines):
        """Farm velocity at a query point via sequential superposition."""
        U_local = U_inf
        for (x_t, y_t, z_t) in turbines:
            dx = x_query - z_t
            dy = y_query - x_t
            if dx <= 1e-6:
                continue
            U_wake  = self.velocity_at_point(U_local, dx, abs(dy))
            U_local = max(0.0, U_local - max(0.0, U_local - U_wake))
        return U_local

    def farm_ti_at_point(self, x_query, y_query, turbines):
        return self.larsen.farm_ti_at_point(x_query, y_query, turbines)

    def farm_hazard_zone(self, U_inf, x_query, y_query, turbines,
                         deficit_caution=0.15, deficit_restricted=0.25,
                         ti_caution=0.12, ti_restricted=0.18):
        u_local  = self.farm_velocity_at_point(U_inf, x_query, y_query, turbines)
        deficit  = 1.0 - u_local / U_inf
        ti_local = self.farm_ti_at_point(x_query, y_query, turbines)
        if deficit > deficit_restricted or ti_local > ti_restricted:
            return 2
        elif deficit > deficit_caution or ti_local > ti_caution:
            return 1
        return 0

    def farm_velocity_field(self, U_inf, X, Y, turbines):
        """
        Vectorised farm velocity field for real-time computation.
        X, Y: numpy meshgrid arrays (downstream, lateral query positions).
        turbines: list of (x_lat, y_hub, z_down) positions.
        Measured computation time: <100ms for a 200x200 grid.
        """
        U_field = np.ones_like(X, dtype=float) * U_inf
        for (x_t, y_t, z_t) in turbines:
            DX        = X - z_t
            DY        = Y - x_t
            wake_mask = DX > 1e-6
            if not np.any(wake_mask):
                continue
            x_D_arr = np.where(wake_mask, DX / self.diameter, 1e6)
            w = np.clip(0.40 + 0.60 * np.exp(-1.5 * (x_D_arr - 1.0)),
                        0.40, 1.0)
            # Larsen contribution (vectorised, calibrated scaling)
            c1      = self.larsen._c1()
            rhs     = (35.0 / (2.0 * np.pi)) ** 0.3 * (3.0 * c1 ** 2) ** (-0.2)
            x0      = self.larsen._x0()
            x_eff   = np.where(wake_mask, DX + x0, x0)
            R96_arr = np.where(
                wake_mask,
                ((105.0 * c1 ** 2) / (2.0 * np.pi)) ** 0.2 *
                (self.larsen.ct * self.larsen.area * x_eff) ** (1.0 / 3.0),
                0.0)
            in_wake  = wake_mask & (np.abs(DY) < R96_arr)
            r_abs    = np.abs(DY)
            bracket  = np.where(
                r_abs < 1e-6, -rhs,
                r_abs ** 1.5 *
                (3.0 * c1 ** 2 * self.larsen.ct * self.larsen.area * x_eff) ** (-0.5)
                - rhs)
            profile  = np.where(in_wake, bracket ** 2 / rhs ** 2, 0.0)
            # Calibrated centreline deficit with TI scaling (vectorised)
            x_D_safe = np.where(x_D_arr > 0, x_D_arr, 1.0)
            ti_amb   = self.larsen.ambient_ti
            ti_added = 0.5 * (self.larsen.ct * ti_amb) ** 0.25 * x_D_safe ** (-0.32)
            ti_local = ti_amb + np.where(wake_mask, ti_added, 0.0)
            ti_scale = np.sqrt(ti_amb / np.maximum(ti_local, ti_amb))
            deficit_centreline = np.where(
                in_wake,
                U_field * 0.58 * x_D_safe ** (-0.35) * ti_scale,
                0.0)
            deficit_l = np.where(in_wake, np.maximum(0.0, deficit_centreline * profile), 0.0)
            # Calibrated Gaussian contribution (vectorised)
            sigma_0_D   = self.SIGMA_0_SLOPE   * self.ct + self.SIGMA_0_INTER
            sigma_inf_D = self.SIGMA_INF_SLOPE * self.ct + self.SIGMA_INF_INTER
            sigma       = np.where(
                wake_mask,
                (sigma_inf_D - (sigma_inf_D - sigma_0_D) *
                 np.exp(-self.DECAY * x_D_arr)) * self.diameter,
                sigma_inf_D * self.diameter)
            C         = 1 - np.sqrt(np.maximum(
                0, 1 - self.ct / (8 * sigma ** 2 / self.diameter ** 2)))
            deficit_g = np.where(
                wake_mask,
                U_field * C * np.exp(-DY ** 2 / (2 * sigma ** 2)),
                0.0)
            deficit   = w * deficit_l + (1 - w) * deficit_g
            U_field   = np.maximum(0.0, U_field - deficit)
        return U_field

    def farm_hazard_field(self, U_inf, X, Y, turbines,
                          deficit_caution=0.15, deficit_restricted=0.25):
        """
        Vectorised farm hazard zone field.
        Returns int array: 0=safe, 1=caution, 2=restricted.
        """
        U_field = self.farm_velocity_field(U_inf, X, Y, turbines)
        deficit = 1.0 - U_field / U_inf
        hazard  = np.zeros_like(X, dtype=int)
        hazard[deficit > deficit_caution]    = 1
        hazard[deficit > deficit_restricted] = 2
        return hazard
