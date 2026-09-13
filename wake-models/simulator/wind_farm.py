from models.larsen import LarsenWakeModel
from models.blended import BlendedWakeModel


class WindFarmSimulator:
    """
    Main controller.

    - Larsen model: turbine velocities, power and RPM.
    - Blended model: wake shape / velocity field / hazard field.
    """

    def __init__(
        self,
        turbines,
        diameter=126.0,
        ct=0.8,
        air_density=1.225,
        cp=0.35,
        ambient_ti=0.08,
        cut_in_speed=3.0,
        cut_out_speed=25.0,
        rated_power=None,
    ):
        self.turbines = list(turbines)
        self.power_model = LarsenWakeModel(
            diameter=diameter,
            ct=ct,
            air_density=air_density,
            cp=cp,
            ambient_ti=ambient_ti,
            cut_in_speed=cut_in_speed,
            cut_out_speed=cut_out_speed,
            rated_power=rated_power,
        )
        self.wake_model = BlendedWakeModel(
            diameter=diameter,
            ct=ct,
            air_density=air_density,
            cp=cp,
            ambient_ti=ambient_ti,
            cut_in_speed=cut_in_speed,
            cut_out_speed=cut_out_speed,
        )

    def power_for_interval(self, wind_vector, debug=False):
        turbines_sorted, velocities, rpms = self.power_model.wind_speeds_full(
            self.turbines, wind_vector, debug=debug
        )
        powers = [self.power_model.power(v) for v in velocities]
        return {
            "wind_vector": wind_vector,
            "turbines": turbines_sorted,
            "velocities": velocities,
            "rpms": rpms,
            "Power_w": powers,
            "farm_power_w": sum(powers),
        }

    def energy_for_timeseries(self, wind_vectors, times):
        return self.power_model.multi_speed(self.turbines, wind_vectors, times)

    def wake_field(self, U_inf, X, Y):
        return self.wake_model.farm_velocity_field(U_inf, X, Y, self.turbines)

    def hazard_field(self, U_inf, X, Y):
        return self.wake_model.farm_hazard_field(U_inf, X, Y, self.turbines)
