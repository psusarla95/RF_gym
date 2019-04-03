from gym.envs.registration import register

register(
    id='rf-v0',
    entry_point='gym_rf.envs:RFBeamEnv'
)


register(
    id='rf-v1',
    entry_point='gym_rf.envs:RFBeamEnv2'
)
