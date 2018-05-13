# The key difference is that we can easily expand the Tensorflow network with added layers, activation functions, and different input types, whereas all that is impossible with a regular table.
import gym
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt

env = gym.make('FrozenLake-v0')

#implementing the network itself
tf.reset_default_graph()

#these lines establish the feed-forward part of the network used to choose actions
inputsl = tf.placeholder(shape=[1,16], dtype=tf.float32)
W = tf.Variable(tf.random_uniform([16,4], 0, 0.01))
Qout = tf.matmul(inputsl,W)
predict = tf.argmax(Qout,1)

#Below we obtain the loss by taking the sum of squares diffrence between the target and prediction Q vales
nextQ = tf.placeholder(shape=[1,4],dtype=tf.float32)
loss = tf.reduce_sum(tf.square(nextQ - Qout))
trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
updateModel = trainer.minimize(loss)

init = tf.initialize_all_variables()

#set learning parameters
y = .99
e = .1
num_episodes = 2000
#create lists to contain total rewards and steps per episode
#jList = []
rList = []
with tf.Session() as sess:
	sess.run(init)
	for i in range(num_episodes):
		#reset environment and get first new observation
	        s = env.reset()
		rAll = 0
		d = False
		j = 0
		#the Q-Network
		while j < 99:
			j += 1
			#choose an action by greedily (with e change of random action) from the Q-network
			a,allQ = sess.run([predict,Qout],feed_dict={inputsl:np.identity(16)[s:s+1]})
            		if np.random.rand(1) < e:
		                a[0] = env.action_space.sample()
			#get new state and reward from environment
			sl,r,d,_ = env.step(a[0])
			#obtain the Q' values by feeding the new state through our network
			Q1 = sess.run(Qout,feed_dict={inputsl:np.identity(16)[sl:sl+1]})
			#Obtain maxQ' and set our target value for chosen action.
			maxQ1 = np.max(Q1)
			targetQ = allQ
			targetQ[0,a[0]] = r + y*maxQ1
            #Train our network using target and predicted Q values.
			_,W1 = sess.run([updateModel,W],feed_dict={inputsl:np.identity(16)[s:s+1],nextQ:targetQ})
            		rAll += r
	                s = sl
            		if d == True:
                #Reduce chance of random action as we train the model.
            		    e = 1./((i/50) + 10)
            		    break
	        #jList.append(j)
        	rList.append(rAll)
print "Percent of succesful episodes: " + str(sum(rList)/num_episodes) + "%"	



