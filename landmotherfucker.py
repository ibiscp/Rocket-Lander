import gym

env = gym.make('RocketLander-v0')

for i_episode in range(100):
    s = env.reset()
    while True:
        env.render()
        a = env.action_space.sample()

        s_, r, done, _ = env.step(a)

        # print('\na')
        # print(a)
        # print('\ns_')
        # print(s_)
        # print('\nr')
        # print(r)
        # print('\ndone')
        # print(done)
        # print('\ninfo')
        # print(info)

        if done:
            break
        s = s_
