import gym
#env = environment
env = gym.make('CartPole-v0')
for i_episode in range(20):
	observation = env.reset()
	for t in range(100):
		env.render()
		print(observation)
		action = env.action_space.sample()
		observation, reward, done, info = env.step(action)
		print info
		if done:
			print("episode finished after {} timesteps.".format(t + 1))
			break
