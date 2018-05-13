# -*- coding: utf-8 -*-
import gym
import numpy as np

env = gym.make('FrozenLake-v0')

#Initialize table with all zeros.
Q = np.zeros([env.observation_space.n, env.action_space.n])
#Set learning parameters.
lr = .8
y = .95
num_episodes = 2000
#create list to contain total rewards and steps per episode
rList = []
for i in range(num_episodes):	
	#reset environment and get first new observation
	s = env.reset()
	rAll = 0
	d = False
	j = 0
	#the Q-Table learning algorithm
	while j < 99:
		j += 1
		#choose an action be greedily picking from Q table with noise
		a = np.argmax(Q[s,:] + np.random.randn(1, env.action_space.n)*(1./(i + 1)))
		#get new state and reward from environment
		sl, r, d, _ = env.step(a)
		#update Q-table with new knowledge 
		#Eq 1. update by the following formular: r + γ(max(Q(s’,a’))
		#Bellman equation, which states that the expected long-term reward for a given action is equal to the immediate reward from the current action combined with the expected reward from the best future action taken at the following state
		Q[s,a] = Q[s,a] + lr*(r + y*np.max(Q[sl,:]) - Q[s,a])
		rAll += r
		#the next state
		s = sl
		if d == True:
			break
		rList.append(rAll)
print "Score over time: " +  str(sum(rList)/num_episodes)
print "Final Q-Table Values: "
print Q
	
