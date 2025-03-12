from pettingzoo.mpe import simple_speaker_listener_v4

# Here you need to write DQN agents to solve this simple_speaker_listener_v4 environment.
# Make sure to use separate q-networks and replay buffers for each agent to secure independent training and running.

env = simple_speaker_listener_v4.parallel_env(render_mode="human")
observations, infos = env.reset()

while env.agents:
    # this is where you would insert your policy
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}

    observations, rewards, terminations, truncations, infos = env.step(actions)
env.close()
