from pettingzoo.mpe import simple_speaker_listener_v4

env = simple_speaker_listener_v4.parallel_env(render_mode="human")
observations, infos = env.reset()

while env.agents:

    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    observations, rewards, terminations, truncations, infos = env.step(actions)

env.close()
