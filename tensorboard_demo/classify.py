"""
	In this  code section we going to do classifiction problem and visualize the whole
	graph in tensorboard. 

	for images:
		1. resize input
		2. conv layer outputs
		3. conv layers weight visualization

	for scalars:
		1.accuracy
		2.weight/loss
		3.cross entropy
		4.dropout

	for distributions and graphs:
		1.weights and biases
		2.activations

	for graph
		1.fully implemented convolutional network graph
"""

#import necessary libararies
import tensorflow as tf
import numpy as np
import shutil
from random import shuffle
import os
from tqdm import tqdm

class Preprocessing():

	def  convert_image_to_numpy_arrays(train_dir,test_dir):
		"""
			read the input image. convert into numpy array
		"""
		import cv2

		training_data,testing_data = [],[] #list to store all training data
		img_size = 28

		def label_img(image_name):
			get_label = image_name.split('.')[-3] #to get rid of ext file
			#if label is cat then return [1,0] else return [0,1]
			if get_label == 'cat': return [1,0]
			elif get_label == 'dog': return [0,1]

		for i in tqdm(os.listdir(train_dir)):
			label = label_img(i)
			image_path = os.path.join(train_dir,i)
			image_read = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
			image_read = cv2.resize(image_read,(img_size,img_size))
			#append to the list
			training_data.append([np.array(image_read),np.array(label)])

		shuffle(training_data)
		np.save("Train_Data.npy",training_data)
		print("Finished creating a Training Dataset..")

		for i in tqdm(os.listdir(test_dir)):
			image_path = os.path.join(test_dir,i)
			image_read = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
			image_read = cv2.resize(image_read,(img_size,img_size))
			#append to the list
			testing_data.append(np.array(image_read))

		shuffle(testing_data)
		np.save("Test_Data.npy",testing_data)
		print("Finished creating a Testing dataset...")

		return training_data,testing_data


class Neural_Network():

	def __init__(self):
		self.learning_rate = 1e-3
		self.display_rate = 100
		self.epochs = 500
		self.image_shape = [-1,50,50,1]
		self.output_dir = 'tb_output_dir'

	def network_architecture(self):
		print("Building an CNN Architecture")
		#set placeholders
		with tf.name_scope('input'):
			x = tf.placeholder(tf.float32,[None,2500],name='x-input')
			y_ = tf.placeholder(tf.float32,[None,2],name='y-input')
		#image resize
		with tf.name_scope('image reshaped'):
			x_reshaped = tf.reshape(x,self.image_shape)
			tf.summary.image('input',x_reshaped,2)
		#set dropout
		with tf.name_scope('Dropout'):
			keep_prob = tf.placeholder(tf.float32)
			tf.summary.scalar('dropout-keepup-probability',keep_prob)

		#convolutional_1 + max-pooling_1 layer starts from here
		"""
			Number of features after we have conv1 and maxpool layer is
			Conv_1 = 46 {input_size = 50 x 50, kernel_size=5x5, strides=1, filter=32, padding=same}. So 50-5/1 + 1
			Max_pool = 23 {kernel_size = 2x2, strides=2x2}. so 46 - 2/2+1  = 23
		"""
		with tf.name_scope('conv-1'):
			with tf.name_scope('weights'):
				W_conv1 = tf.Variable(tf.truncated_normal([5,5,1,32],stddev=0.1))
				with tf.name_scope('summaries'):
					mean = tf.reduce_mean(W_conv1)
					tf.summary.scalar('mean',mean)
					with tf.name_scope('stddev'):
						stddev = tf.sqrt(tf.reduce_mean(tf.square(W_conv1 - mean)))

					tf.summary.scalar('stddev',stddev)
					tf.summary.scalar('max',tf.reduce_mean(W_conv1))
					tf.summary.scalar('min',tf.reduce_mean(W_conv1))
					tf.summary.scalar('histogram',W_conv1)


			with tf.name_scope('biases'):
				b_conv1 = tf.Variable(tf.constant(0.1,shape=[32])) #constant value
				with tf.name_scope('summaries'):
					mean = tf.Variable(tf.truncated_normal([5,5,1,32],shape=[32]))
					tf.summary.scalar('mean',mean)
					with tf.name_scope('stddev'):
						stddev = tf.sqrt(tf.reduce_mean(tf.square(b_conv1 - mean)))

					tf.summary.scalar('stddev',stddev)
					tf.summary.scalar('max',tf.reduce_mean(b_conv1))
					tf.summary.scalar('min',tf.reduce_mean(b_conv1))
					tf.summary.scalar('histogram',b_conv1)

			with tf.name_scope('Wx_plus_b'):
				preactivated1 = tf.nn.conv2d(x_reshaped,W_conv1,strides=[1,1,1,1],
					padding='SAME') + b_conv1 #creating convolutional layer
				h_conv1 = tf.nn.relu(preactivated1) #activation layer
				tf.summary.histogram('pre_activated_1',preactivated1) #plotting histogram
				tf.summary.histogram('activations',h_conv1)


			#maxpool layer
			with tf.name_scope('max_pool_1'):
				h_pool1  = tf.nn.max_pool(h_conv1,
								ksize = [1,2,2,1],
								strides=[1,2,2,1],
								padding='SAME')

			#save output of conv layer to the tensorboard first 16 filters
			with tf.name_scope('Image_output_conv_1'):
				image = h_conv1[0:1,:,:,0:16]
				image = tf.transpose(image,perm=[3,1,2,0])
				tf.summary.image('Image_conv1_output',image)

		#save the visual representation of weights to tensorboard
		with tf.name_scope('Visualize_weights_conv1'):
			W_a = W_conv1 #[5,5,1,32]
			W_b = tf.split(W_a,32,3) #[32,5,5,1,1]
			rows = []

			for i in range(int(32/8)):
				x1 = i*8
				x2 = (i+1)*8
				row = tf.concat(W_b[x1:x2],0)
				rows.append(row)

			W_c = tf.concat(rows,1)
			c_shape = W_c.get_shape().as_list()
			W_d = tf.reshape(W_c,[c_shape[2],c_shape[0],c_shape[1],1])

			tf.summary.image("Visualize_kernels_conv1",W_d,1024)


	#convolutional_2 + max_pooling_2 

	with tf.name_scope('Convolutional_2'):
		with tf.name_scope('weights'):
			W_conv2 = tf.Variable(tf.truncated_normal([5,5,1,64],stddev=0.1))
			with tf.name_scope('Summaries'):
				mean = tf.reduce_mean(W_conv2)
					tf.summary.scalar('mean',mean)
					with tf.name_scope('stddev'):
						stddev = tf.sqrt(tf.reduce_mean(tf.square(W_conv2 - mean)))

					tf.summary.scalar('stddev',stddev)
					tf.summary.scalar('max',tf.reduce_mean(W_conv2))
					tf.summary.scalar('min',tf.reduce_mean(W_conv2))
					tf.summary.scalar('histogram',W_conv2)

		with tf.name_scope('biases'):
			b_conv2 = tf.Variable(tf.constant(0.1,shape=[64])) #constant value
				with tf.name_scope('summaries'):
					mean = tf.Variable(tf.truncated_normal([5,5,1,64],shape=[64]))
					tf.summary.scalar('mean',mean)
					with tf.name_scope('stddev'):
						stddev = tf.sqrt(tf.reduce_mean(tf.square(b_conv2 - mean)))

					tf.summary.scalar('stddev',stddev)
					tf.summary.scalar('max',tf.reduce_mean(b_conv2))
					tf.summary.scalar('min',tf.reduce_mean(b_conv2))
					tf.summary.scalar('histogram',b_conv2)


		with tf.name_scope('Wx_plus_b'):
			preactivated2 = tf.nn.conv2d(h_pool1,W_conv2,strides=[1,1,1,1],
					padding='SAME') + b_conv2 #creating convolutional layer
				h_conv2 = tf.nn.relu(preactivated2) #activation layer
				tf.summary.histogram('pre_activated_2',preactivated2) #plotting histogram
				tf.summary.histogram('activations',h_conv2)

		with tf.name_scope('dropout'):
			h_pool2  = tf.nn.max_pool(h_conv1,
								ksize = [1,2,2,1],
								strides=[1,2,2,1],
								padding='SAME')
		
		#image output
		with tf.name_scope('Image_output_conv2'):
			image = h_conv2[0:1,:,:,0:16]
				image = tf.transpose(image,perm=[3,1,2,0])
				tf.summary.image('Image_conv1_output',image)

	#visualization of second layer
	with tf.name_scope('Visualize_weights_conv2'):
			W_a = W_conv2 #[5,5,1,64]
			W_b = tf.split(W_a,64,3) #[64,5,5,1,1]
			rows = []

			for i in range(int(64/8)):
				x1 = i*8
				x2 = (i+1)*8
				row = tf.concat(W_b[x1:x2],0)
				rows.append(row)

			W_c = tf.concat(rows,1)
			c_shape = W_c.get_shape().as_list()
			W_d = tf.reshape(W_c,[c_shape[2],c_shape[0],c_shape[1],1])

			tf.summary.image("Visualize_kernels_conv2",W_d,1024)


def main():
	#load the dataset path
	TRAIN_DIR = 'F:/NN/kaggle_competition/classification/DC_classifier/train/train'
	TEST_DIR = 'F:/NN/kaggle_competition/classification/DC_classifier/test/test'

	#check if the model exists, throws an error if path doesn't valid
	if os.path.exists(TRAIN_DIR) and os.path.exists(TEST_DIR):
		print("Path to the dataset is valid")
	else:
		print("Error..!Please specify the correct path.")

	#convert the image into a numpy arrays
	#train, test = Preprocessing.convert_image_to_numpy_arrays(TRAIN_DIR,TEST_DIR)

	network = Neural_Network()
	network.network_architecture()

if __name__ == '__main__':
	main()

