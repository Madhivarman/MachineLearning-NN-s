import tensorflow as tf 
import numpy as np

class VAE():

	def __init__(
		self, 
		network_architecture, 
		transfer_fun = tf.nn.softplus,
		learning_rate=0.001, 
		batch_size=None
	):

		self.network_architecture = network_architecture
		self.transfer_fun = transfer_fun
		self.learning_rate = learning_rate
		self.batch_size = batch_size

		#graph input
		self.x = tf.placeholder(tf.float32, [None, network_architecture['n_input']])

		#encoder network
		self.__create_network()

		#loss function
		self.__loss_optimizer()

		#initializing all tensorflow variables
		init = tf.global_variables_initializer()

		#session
		self.sess = tf.InteractiveSession()
		self.sess.run(init)



	def xavier_init(
		self, 
		fan_in, 
		fan_out, 
		constant=1
	):

		low = -constant*np.sqrt(6.0/(fan_in + fan_out))
		high = constant*np.sqrt(6.0/(fan_in + fan_out))

		return tf.random_uniform((fan_in, fan_out),
			minval=low, maxval=high, dtype=tf.float32)


	def __create_network(self):
		#initialize network weights
		network_weights = self.__initialize_weights(**self.network_architecture)

		#here recognization network is encoder
		#which uses gaussion distribution to encode the image in
		#latent space

		self.z_mean, self.z_sigma = self.__encoder_network(
					network_weights['weights_encoder'],
					network_weights['biases_encoder']
		)

		#draw one sample z from gaussian distribution
		n_z = self.network_architecture["n_z"]
		eps = tf.random_normal((self.batch_size, n_z), 0, 1, dtype=tf.float32)

		#z = mu + sigma * epsilon
		self.z = tf.add(self.z_mean, tf.matmul(tf.sqrt(tf.exp(self.z_sigma)), eps))

		#here generator network is a decoder
		#where uses sample mean, sample std from encoder to a latent space
		#to decode a image

		self.x_reconstr_mean = self.__decoder_network(
										network_weights["weights_decoder"],
										network_weights["biases_decoder"]
		)




	def __initialize_weights(
		    self,
		    n_hidden_recog_1,
		    n_hidden_recog_2,
		    n_hidden_gener_1,
		    n_hidden_gener_2,
		    n_input,
		    n_z,
    ):

	    all_weights = dict()
	    all_weights['weights_encoder'] = {
	        'h1': tf.Variable(self.xavier_init(n_input, n_hidden_recog_1)),
	        'h2': tf.Variable(self.xavier_init(n_hidden_recog_1,
	                          n_hidden_recog_2)),
	        'out_mean': tf.Variable(self.xavier_init(n_hidden_recog_2, n_z)),
	        'out_log_sigma': tf.Variable(self.xavier_init(n_hidden_recog_2,
	                n_z)),
	        }
	    all_weights['biases_encoder'] = {
	        'b1': tf.Variable(tf.zeros([n_hidden_recog_1],
	                          dtype=tf.float32)),
	        'b2': tf.Variable(tf.zeros([n_hidden_recog_2],
	                          dtype=tf.float32)),
	        'out_mean': tf.Variable(tf.zeros([n_z], dtype=tf.float32)),
	        'out_log_sigma': tf.Variable(tf.zeros([n_z],
	                dtype=tf.float32)),
	        }
	    all_weights['weights_decoder'] = {
	        'h1': tf.Variable(self.xavier_init(n_z, n_hidden_gener_1)),
	        'h2': tf.Variable(self.xavier_init(n_hidden_gener_1,
	                          n_hidden_gener_2)),
	        'out_mean': tf.Variable(self.xavier_init(n_hidden_gener_2,
	                                n_input)),
	        'out_log_sigma': tf.Variable(self.xavier_init(n_hidden_gener_2,
	                n_input)),
	        }
	    all_weights['biases_decoder'] = {
	        'b1': tf.Variable(tf.zeros([n_hidden_gener_1],
	                          dtype=tf.float32)),
	        'b2': tf.Variable(tf.zeros([n_hidden_gener_2],
	                          dtype=tf.float32)),
	        'out_mean': tf.Variable(tf.zeros([n_input], dtype=tf.float32)),
	        'out_log_sigma': tf.Variable(tf.zeros([n_input],
	                dtype=tf.float32)),
	     	}

	    return all_weights

	"""
		A Simple Feed Forward Neural Network where last two layer produce
		mean and standard deviation for the input probability

	"""
	def __encoder_network(
		self,
		weights,
		biases
	):

		layer_1 = self.transfer_fun(tf.add(tf.matmul(self.x, weights['h1']),biases['b1']))
		layer_2 = self.transfer_fun(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2']))
		z_mean = tf.add(tf.matmul(layer_2, weights['out_mean']), biases['out_mean'])
		z_std = tf.add(tf.matmul(layer_2, weights['out_log_sigma']), biases['out_log_sigma'])

		return (z_mean, z_std)

	"""
		A Simple Feed Forward Neural Network where it takes latent space 
		vectors as an input and forward to do computation to produce output
		which is similar to the input. Last layer we use is sigmoid.

		the goal of the whole network is to produce the sigmoid value near
		to the value 1
	"""

	def __decoder_network(
		self,
		weights,
		biases
	):

		layer_1 = self.transfer_fun(tf.add(tf.matmul(self.z, weights['h1']), biases['b1']))
		layer_2 = self.transfer_fun(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2']))
		x_reconstr_mean = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['out_mean'], biases['out_mean'])))

		return x_reconstr_mean


	"""Optimization"""

	def __loss_optimizer(self):
		"""
			Two Loss

			1. reconstruction loss -> the negative log probability of the input
			under the reconstructed distribution induced by the decoder in the
			latent space. It can useful for reconstructing the input when the activation
			in latent is given

			2. KL Divergence between the distribution in latent space induced by the encoder
			on the data and some prior. this act as a regulizer
		"""

		#this formula is similar to GAN's
		reconstruction_loss = -tf.reduce_sum(self.x * tf.log(1e-10 + self.x_reconstr_mean)
							+ (1-self.x) * tf.log(1e-10 + 1 - self.x_reconstr_mean))

		latent_loss = -0.5 * tf.reduce_sum(1 + self.z_sigma - tf.square(self.z_mean) - tf.exp(self.z_sigma), 1)

		self.cost = tf.reduce_mean(reconstruction_loss + latent_loss)

		#adam optimizer
		self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)


	def partial_fit(self, X):
		"""Applying Mini Batch Here"""
		opt, cost = self.sess.run((self.optimizer, self.cost), feed_dict={self.x:X})
		return cost	

	def transform(self, X):

		return self.sess.run(self.z_mean, feed_dict={self.x: X})

	def generate(self, z_mu=None):

		"""
			Generate data by latent space. If z_mu is not None, data for
			this point in latent Space is generated. Otherwise z_mu is drawn
			from prior in latent space.
		"""
		if z_mu is None:
			z_mu = np.random.normal(size=self.network_architecture["n_z"])

		return self.sess.run(self.x_reconstr_mean, feed_dict={self.z: z_mu})


	def reconstruct(self, X):

		return self.sess.run(self.x_reconstr_mean, feed_dict={self.x: X })















