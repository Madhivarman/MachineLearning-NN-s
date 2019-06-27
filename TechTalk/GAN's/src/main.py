import os
import numpy as np
import matplotlib.pyplot as plt
import glob
import lib
import os
import matplotlib.gridspec as gridspec
import time 

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
	#print(discriminator_model.summary())


	return discriminator_model



#plot the images to see the sample dataset
def plot_sample_data(filenames):
	plt.figure(figsize=(10, 8))
	for i in range(5):
	    img = plt.imread(filenames[i], 0)
	    plt.subplot(4, 5, i+1)
	    plt.imshow(img)
	    plt.title(img.shape)
	    plt.xticks([])
	    plt.yticks([])
	plt.tight_layout()
	plt.show()


def norm_image(img):
	img = (img / 127.5) - 1
	return img

def denorm_image(img):
	"""
		A function to denormalize the image
	"""
	img = (img + 1) * 127.5
	return img.astype(np.uint8)


def sample_from_dataset(batch_size, image_shape, data_dir):

	sample_dim = (batch_size,) + image_shape
	sample = np.empty(sample_dim, dtype=np.float32)
	all_data_dirlist = list(glob.glob(data_dir))
	sample_imgs_path = np.random.choice(all_data_dirlist, batch_size)

	for index, img_filename in enumerate(sample_imgs_path):
		image = lib.Image.open(img_filename)
		image = image.resize(image_shape[:-1])
		image = image.convert('RGB')
		image = np.asarray(image)
		image = norm_image(image)
		sample[index,...] = image

		return sample


def save_img_batch(img_batch, img_save_dir):
	plt.figure(figsize=(16,16))
	gs1 = gridspec.GridSpec(4,4)
	gs1.update(wspace=0, hspace=0)
	rand_indices = np.random.choice(img_batch.shape[0], 16, replace=False)
	for i in range(16):
		ax1 = plt.subplot(gs1[i])
		ax1.set_aspect('equal')
		rand_index = rand_indices[i]
		image = img_batch[rand_index, :,:,:]
		fig = plt.imshow(denorm_image(image))
		plt.axis('off')
		fig.axes.get_xaxis().set_visible(False)
		fig.axes.get_yaxis().set_visible(False)

	plt.tight_layout()
	plt.savefig(img_save_dir, bbox_inches='tight', pad_inches=0)
	#plt.show()


def generate_images(generator, save_dir, batch_size, noise_shape):
    noise = get_noise(batch_size,noise_shape)
    fake_data_X = generator.predict(noise)
    print("Displaying generated images")
    plt.figure(figsize=(16,16))
    gs1 = gridspec.GridSpec(4, 4)
    gs1.update(wspace=0, hspace=0)
    rand_indices = np.random.choice(fake_data_X.shape[0],16,replace=False)
    for i in range(16):
        ax1 = plt.subplot(gs1[i])
        ax1.set_aspect('equal')
        rand_index = rand_indices[i]
        image = fake_data_X[rand_index, :,:,:]
        fig = plt.imshow(denorm_image(image))
        plt.axis('off')
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
    plt.tight_layout()
    plt.savefig(str(time.time())+"_GENERATEDimage.png",bbox_inches='tight',pad_inches=0)
    plt.show()


def plot_loss(avg_disc_real_loss, avg_disc_fake_loss, avg_GAN_loss, num_steps):

	disc_real_loss = np.array(avg_disc_real_loss)
	disc_fake_loss = np.array(avg_disc_fake_loss)
	GAN_loss = np.array(avg_GAN_loss)

	# Plot the losses vs training steps
	plt.figure(figsize=(16,8))
	plt.plot(range(0,num_steps), disc_real_loss, label="Discriminator Loss - Real")
	plt.plot(range(0,num_steps), disc_fake_loss, label="Discriminator Loss - Fake")
	plt.plot(range(0,num_steps), GAN_loss, label="Generator Loss")
	plt.xlabel('Steps')
	plt.ylabel('Loss')
	plt.legend()
	plt.title('GAN Loss')
	plt.grid(True)
	plt.savefig("loss.png")
	plt.show() 


def main():

	data_dir = "animeface-character-dataset/animeface-character-dataset/*/*/*.pn*"
	img_save_dir='images'
	num_steps = 2000
	noise_shape = (1, 1, 100)
	batch_size = 64
	image_shape = (64,64,3)

	filenames = glob.glob(data_dir)
	print("Num_Images: ",len(filenames))


	#get the noise data
	noise = get_noise(batch_size,noise_shape)

		#call the generator architecture
	G = generator_network(noise_shape)

		#call the discriminator network
	D = discriminator_network(image_shape)

		#generator and discriminator are then combined to create a Final GAN
	D.trainable = False

		#optimizer for GAN
	opt = lib.Adam(lr=0.00015, beta_1=0.5)

		#input to the generator
	gen_input = lib.Input(shape=noise_shape)

	GAN_input = G(gen_input)
	GAN_opt = D(GAN_input)

		#final gan
	GAN = lib.Model(input=gen_input, output=GAN_opt)
	GAN.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

		#to keep track of losses
	avg_disc_fake_loss = []
	avg_disc_real_loss = []
	avg_GAN_loss = []

	for step in range(num_steps):
			tot_step = step

			#sample a batch of normalized images from the dataset
			real_data_x = sample_from_dataset(batch_size, image_shape, data_dir=data_dir)
			#generate noise to send as input to the generatir
			noise = get_noise(batch_size, noise_shape)

			#use generator to create images
			fake_data_x = G.predict(noise)

			#save predicted images from the generator every 10th step
			if (tot_step % 100) == 0:
				step_num = str(tot_step).zfill(4)
				#save_img_batch(fake_data_x, "images/" +step_num+"_image.png")

			#creating the labels for real and fake data. 
			real_data_Y = np.ones(batch_size) - np.random.random_sample(batch_size)*0.2
			fake_data_y = np.random.random_sample(batch_size) * 0.2

			#train the discriminator using data and labels
			D.trainable = True
			G.trainable = False

			#training discriminator in real data separately for identifying real and fake one
			dis_metrics_real = D.train_on_batch(real_data_x, real_data_Y)

			#training discriminator separately on fake data
			dis_metrics_fake = D.train_on_batch(fake_data_x, fake_data_y)

			#save the metrics
			avg_disc_fake_loss.append(dis_metrics_fake[0])
			avg_disc_real_loss.append(dis_metrics_real[0])

			#train generator using a random vector of noise and its labels
			G.trainable = True
			D.trainable = False 

			GAN_X = get_noise(batch_size, noise_shape)
			GAN_Y = real_data_Y

			GAN_metrics = GAN.train_on_batch(GAN_X, GAN_Y)
			print("Step:{}, GAN Loss:{}".format(tot_step,GAN_metrics[0]))

			#log results store in the file
			text_file = open("training_log.txt", "a")
			text_file.write("Step:{}, Discriminator Real loss:{}, Fake loss:{}".format(tot_step, dis_metrics_real[0], dis_metrics_fake[0]))
			text_file.close()

			#save GAN loss to plot later
			avg_GAN_loss.append(GAN_metrics[0])

			#save model at every 500 steps
			if ((tot_step+1) %500) == 0:

				print("-" * 20)
				print("average discriminator fake loss:{}".format(np.mean(avg_disc_fake_loss)))
				print("average discriminator real loss:{}".format(np.mean(avg_disc_real_loss)))
				print("average gan loss:{}".format(np.mean(avg_GAN_loss)))
				print("-" * 20)

				D.trainable = False
				G.trainable = False

				fixed_noise_generate = G.predict(noise)
				step_num = str(tot_step).zfill(4)
				#save_img_batch(fixed_noise_generate, "imageGenerated/"+step_num+"fixed_image.png")
				G.save("modelLogs/"+str(tot_step)+"_GENERATOR_weights_and_arch.hdf5")
				D.save("modelLogs/"+str(tot_step)+"_DISCRIMINATORS_weights_and_arch.hdf5")


	plot_loss(avg_disc_real_loss, avg_disc_fake_loss, avg_GAN_loss, num_steps)
	generate_images(G, img_save_dir, batch_size, noise_shape)


		

#main start
if __name__ == '__main__':
	main()