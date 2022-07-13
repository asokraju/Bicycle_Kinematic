from gym_bicycle.envs.registration import register

register(
    id='bicycle-v0',
    entry_point='Bicycle_Kinematic.envs:BicycleKin',
)