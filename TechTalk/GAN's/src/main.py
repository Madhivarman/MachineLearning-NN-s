import os
import numpy as np
import matplotlib.pyplot as plt
import glob
import lib
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'


def get_noise(bs, noise_shape):
	"""
		input: create a random samples between 0,1 in the 
		shape (bs, noise_shape)
		args value: bs - number of images
					noise_shape - (1,1,100)
		output:
			return normal distribution
	"""
	return  np.random.normal(0, 1, size=(bs,)+noise_shape)


def generator_network(noise_shape):
	'''
		this function take noise_shape as an input and create the image
	'''
	kernel_init = 'glorot_uniform' #xavier initializer
	gen_input = lib.Input(shape= noise_shape)

	#network architecture
	#first layer
	generator = lib.Conv2DTranspose(filters=512, kernel_size=(4,4), strides=(1,1),
		padding='valid', data_format='channels_last', kernel_initializer=kernel_init)(gen_input)
	generator = lib.BatchNormalization(momentum=0.5)(generator)
	generator = lib.LeakyReLU(0.2)(generator)

	#second layer
	generator = lib.Conv2DTranspose(filters=256, kernel_size=(4,4), strides=(2,2),
		padding='same', data_format='channels_last', kernel_initializer=kernel_init)(generator)
	generator = lib.BatchNormalization(momentum=0.5)(generator)
	generator = lib.LeakyReLU(0.2)(generator)

	#third layer
	generator = lib.Conv2DTranspose(filters=128, kernel_size=(4,4), strides=(2,2),
		padding='same', data_format='channels_last', kernel_initializer=kernel_init)(generator)
	generator = lib.BatchNormalization(momentum=0.5)(generator)
	generator = lib.LeakyReLU(0.2)(generator)

	#fourth layer
	generator = lib.Conv2DTranspose(filters=64, kernel_size=(4,4), strides=(2,2),
		padding='same', data_format='channels_last', kernel_initializer=kernel_init)(generator)
	generator = lib.BatchNormalization(momentum=0.5)(generator)
	generator = lib.LeakyReLU(0.2)(generator)

	#conv layer
	generator = lib.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1),
		padding='same', data_format='channels_last', kernel_initializer=kernel_init)(generator)
	generator = lib.BatchNormalization(momentum=0.5)(generator)
	generator = lib.LeakyReLU(0.2)(generator)

	#fifth layer
	generator = lib.Conv2DTranspose(filters=3, kernel_size=(4,4), strides=(2,2),
		padding='same', data_format='channels_last', kernel_initializer=kernel_init)(generator)
	
	#tanh activation to get the final normalized image
	generator = lib.Activation('tanh')(generator)

	#optimizer
	generator_optimizer = lib.Adam(lr=0.00015, beta_1=0.5)
	generator_model = lib.Model(input=gen_input, output=generator)
	generator_model.compile(loss='binary_crossentropy', optimizer=generator_optimizer, metrics=['accuracy'])

	generator_model.summary()
	
	#to save the generator network architecture
	#lib.plot_model(generator, to_file='generator_architecture.png', show_shapes=True, show_layer_names=True)

	return generator_model

#get the image shape
def discriminator_network(image_shape=(64,64,3)):
	#drop out probability
	dropout_prob = 0.4
	kernel_init = 'glorot_uniform'
	dis_input = lib.Input(shape = image_shape)

	#conv layer 1
	discriminator = lib.Conv2D(filters=64, kernel_size=(4,4), strides=(2,2), padding='same',
		data_format='channels_last', kernel_initializer=kernel_init)(dis_input)

	discriminator = lib.LeakyReLU(0.2)(discriminator)

	#conv layer 2
	discriminator = lib.Conv2D(filters=128, kernel_size=(4,4), strides=(2,2), padding='same',
		data_format='channels_last', kernel_initializer=kernel_init)(discriminator)

	discriminator = lib.BatchNormalization(momentum=0.5)(discriminator)
	discriminator = lib.LeakyReLU(0.2)(discriminator)

	#conv layer 3
	discriminator = lib.Conv2D(filters=256, kernel_size=(4,4), strides=(2,2), padding='same',
		data_format='channels_last', kernel_initializer=kernel_init)(discriminator)

	discriminator = lib.BatchNormalization(momentum=0.5)(discriminator)
	discriminator = lib.LeakyReLU(0.2)(discriminator)

	#conv layer 4
	discriminator = lib.Conv2D(filters=512, kernel_size=(4,4), strides=(2,2), padding='same',
		data_format='channels_last', kernel_initializer=kernel_init)(discriminator)

	discriminator = lib.BatchNormalization(momentum=0.5)(discriminator)
	discriminator = lib.LeakyReLU(0.2)(discriminator)

	#flatten
	discriminator = lib.Flatten()(discriminator)

	#dense layer
	discriminator = lib.Dense(1)(discriminator)

	#sigmoid activation
	discriminator = lib.Activation('sigmoid')(discriminator)

	#optimizer and compiling model
	dis_opt = lib.Adam(lr=0.0002, beta_1=0.5)

	#creating a model
	discriminator_model = lib.Model(input=dis_input, output=discriminator)

	#optimizing the network
	discriminator_model.compile(loss='binary_crossentropy', optimizer=dis_opt, metrics=['accuracy'])

	#print the model summary
	print(discriminator_model.summary())


	return discriminator_model



#plot the images to see the sample dataset
def plot_sample_data(data):
	#set the figure size
	plt.figure(figsize=(10, 8))
	for i in range(20):
		    plt.subplot(4, 5, i+1)
		    plt.imshow(data[i])
		    plt.title(data[i].shape)
		    plt.xticks([])
		    plt.yticks([])
	plt.tight_layout()
	plt.show()



def main():
	records = "2d-data.npy"
	if os.path.isfile(records):
		#console message
		print("Data Dump is found, ready for training a network")
		data = np.load(records)
		print("*" * 20)

		print('TOTAL NUMBER OF BATCHES:{}'.format(len(data)))
		print("BATCH SIZE:{}".format(len(data[0])))
		print("IMAGE DIMENSION:{}".format(data[0].shape))
		print("*" * 20)

		#plot some samples
		#plot_sample_data(data)

		#set all the parameters required for creating a network
		batch_size = 14720
		noise_shape = (1, 1, 100) #initial noise needed to be add in the generator network
		image_shape = (64, 64, 3) #define the image shape

		#get the noise data
		noise = get_noise(batch_size,noise_shape)

		#call the generator architecture
		G = generator_network(noise_shape)

		#call the discriminator network
		D = discriminator_network(image_shape)




#main start
if __name__ == '__main__':
	main()