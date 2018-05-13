import tensorflow as tf
import numpy as np

#List out our bandits. Currently bandit 4 (index#3) is set to most often provide a positive reward.
bandits = [0.2, 0, -0.2, 2]
num_bandits = len(bandits)
def pullBandit(bandit):
	#get a random number.
	result = np.random.randn(1)
	if result > bandit:
		#return a positive reward
		return 1
	else:
		#return a negative reward
		return -1

tf.reset_default_graph()

#These two lines established the feed-forward part of the network. This does the actual choosing.
#create a variable with init value is an array with num_bandits column.
weights = tf.Variable(tf.ones([num_bandits]))
chosen_action = tf.argmax(weights, 0)

#The next six lines establish the training procedure. We feed the reward and chosen action into the network.
#To compute the loss, and use it to update the network.
#create an array [element] with element is float32
reward_holder = tf.placeholder(shape=[1], dtype=tf.float32)
#create an array [element] with element is int32
action_holder = tf.placeholder(shape=[1], dtype=tf.int32)
responsible_weight = tf.slice(weights, action_holder, [1])
loss = -(tf.log(responsible_weight) * reward_holder)
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.001)
update = optimizer.minimize(loss)

#Set total number of episodes to train agent on.
total_episodes = 1000 
#Set scoreboard for bandits to 0.
total_reward = np.zeros(num_bandits) 
#Set the chance of taking a random action.
e = 0.1

init = tf.initialize_all_variables()

#Launch the tensorflow graph.
with tf.Session() as sess:
	sess.run(init)
	i = 0
	while i < total_episodes:
		#choose either a random action or one from our network.
		if np.random.rand(1) < e:
			action = np.random.randint(num_bandits)
		else:
			action = sess.run(chosen_action)
		
		#Get our reward from picking one of the bandits.
		reward = pullBandit(bandits[action])
		
		#Update the network.
		_,resp,ww = sess.run([update,responsible_weight,weights], feed_dict={reward_holder:[reward],action_holder:[action]})

		#Update our running tally of scores.
		total_reward[action] += reward
		if i % 50 == 0:
            		print "Running reward for the " + str(num_bandits) + " bandits: " + str(total_reward)
		i+=1
print "The agent thinks bandit " + str(np.argmax(ww)+1) + " is the most promising...."
if np.argmax(ww) == np.argmax(-np.array(bandits)):
    print "...and it was right!"
else:
    print "...and it was wrong!"	
	
		


		
