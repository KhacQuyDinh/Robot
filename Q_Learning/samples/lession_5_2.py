import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import gym
import matplotlib.pyplot as plt

try:
    xrange = xrange
except:
    xrange = range

env = gym.make('CartPole-v0')
gamma = 0.99

def discount_rewards(r):
	discounted_r = np.zeros_like(r)
	running_add = 0
	for t in reversed(xrange(0, r.size)):
		running_add = running_add * gamma + r[t]
		discounted_r[t] = running_add
	return discounted_r

class agent():
	def __init__(self, lr, s_size, a_size, h_size):
		#These lines established the feed-forward part of the network. The agent takes a state and produces an action.
		
		#input-layer = state-layer includes all states
		self.state_in= tf.placeholder(shape=[None,s_size],dtype=tf.float32)
		#hidden-layer includes a connected net to input layer and h_size
		hidden = slim.fully_connected(self.state_in,h_size,biases_initializer=None,activation_fn=tf.nn.relu)
		#output-layer = action-layer includes a connected net to hidden layer and a_zise
		self.output = slim.fully_connected(hidden,a_size,activation_fn=tf.nn.softmax,biases_initializer=None)
		#get the maximum element in row #2
		self.chosen_action = tf.argmax(self.output,1)

		#The next six lines establish the training proceedure. We feed the reward and chosen action into the network
		#to compute the loss, and use it to update the network.
	
		#1-dimensional array float32
		self.reward_holder = tf.placeholder(shape=[None],dtype=tf.float32)
		#1-dimensional array int32
		self.action_holder = tf.placeholder(shape=[None],dtype=tf.int32)
		
		self.indexes = tf.range(0, tf.shape(self.output)[0]) * tf.shape(self.output)[1] + self.action_holder
		self.responsible_outputs = tf.gather(tf.reshape(self.output, [-1]), self.indexes)

		#cal everage of all elements in the list
		self.loss = -tf.reduce_mean(tf.log(self.responsible_outputs)*self.reward_holder)
		
		#a list of vars
		tvars = tf.trainable_variables()
		print 'trainable variables: ' + str(tvars)
		self.gradient_holders = []
		for idx,var in enumerate(tvars):
		    placeholder = tf.placeholder(tf.float32,name=str(idx)+'_holder')
		    self.gradient_holders.append(placeholder)
		
		#A list of sum(dy/dx) for each x in xs.
		self.gradients = tf.gradients(self.loss,tvars)
		#Use Adam algorithm.
		optimizer = tf.train.AdamOptimizer(learning_rate=lr)
		self.update_batch = optimizer.apply_gradients(zip(self.gradient_holders,tvars))


#Clear the Tensorflow graph.
tf.reset_default_graph() 

#Load the agent.
myAgent = agent(lr=1e-2,s_size=4,a_size=2,h_size=8) 

#Set total number of episodes to train agent on.
total_episodes = 5000 
max_ep = 999
update_frequency = 5

init = tf.global_variables_initializer()

# Launch the tensorflow graph
with tf.Session() as sess:
    sess.run(init)
    i = 0
    total_reward = []
    total_length = []
        
    gradBuffer = sess.run(tf.trainable_variables())
    for ix,grad in enumerate(gradBuffer):
        gradBuffer[ix] = grad * 0
        
    while i < total_episodes:
        s = env.reset()
        running_reward = 0
        ep_history = []
        for j in range(max_ep):
            #Probabilistically pick an action given our network outputs.
            a_dist = sess.run(myAgent.output,feed_dict={myAgent.state_in:[s]})
            a = np.random.choice(a_dist[0],p=a_dist[0])
            a = np.argmax(a_dist == a)

	    #Get our reward for taking an action given a bandit.
            s1,r,d,_ = env.step(a) 
            ep_history.append([s,a,r,s1])
            s = s1
            running_reward += r
	    #When finished gradient descent.
            if d == True:
                #Update the network.
		
		#Will use ep_history for feed_dict.                
		ep_history = np.array(ep_history)
		#ep_history[:,2] relatives to rewards.
                ep_history[:,2] = discount_rewards(ep_history[:,2])

		#Will use feed_dict.
                feed_dict={myAgent.reward_holder:ep_history[:,2],
                        myAgent.action_holder:ep_history[:,1],
			myAgent.state_in:np.vstack(ep_history[:,0])}
               	
		#Use gradient descent to update the network.
		grads = sess.run(myAgent.gradients, feed_dict=feed_dict)
	
		#Store history of gradBuffer.
                for idx,grad in enumerate(grads):
                    gradBuffer[idx] += grad
		
		print "gradBuffer: " + str(gradBuffer)

                if (i % update_frequency == 0) and (i != 0):
                    feed_dict = dictionary = dict(zip(myAgent.gradient_holders, gradBuffer))		  
                    _ = sess.run(myAgent.update_batch, feed_dict=feed_dict)
                    for ix,grad in enumerate(gradBuffer):
                        gradBuffer[ix] = grad * 0
                
                total_reward.append(running_reward)
                total_length.append(j)
                break

        
            #Update our running tally of scores.
        if i % 100 == 0:
            print(np.mean(total_reward[:]))
        i += 1

