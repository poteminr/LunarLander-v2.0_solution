from gym.envs.registration import register

register(
    id='grid_world-v0',
    entry_point='gym_foo.envs:FooEnv',
)