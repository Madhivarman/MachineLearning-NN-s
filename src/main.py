import os
import numpy as np
from network import VAE
import matplotlib.pyplot as plt

class Data():
	"""Initial Declaration"""
	def __init__(self, dumpname):
		self.d = np.load(dumpname)
		self.batch_size = 8
		self.n_samples = None
		self.n_batches = None
		self.batch_start, self.batch_ends = 0, self.batch_size


	def return_fn(self):
		for r in range(self.n_batches):
			data = self.d[0][self.batch_start: self.batch_ends]
			label =  self.d[1][self.batch_start: self.batch_ends]
			self.batch_start = self.batch_ends
			self.batch_ends += self.batch_size
			yield data, label

	def data_as_batch(self, size, batch_name, data=None):

		#assert len(self.d[0]) == len(self.d[1]), "Length of Input and Label records should be same!"
		#self.n_samples = len(self.d[0])
		self.n_samples = len(data[0])

		#find total number of batches
		total_batches = int(self.n_samples/self.batch_size)
		self.n_batches = total_batches #setting total batches to the class object

		batch_data = self.return_fn()

		n_percent_data = int(self.n_samples * (size / 100))
		n_percent_batch = int(n_percent_data / self.batch_size)

		print("Number of Records in {n} Set: {rn}".format(n=batch_name, rn=n_percent_data))

		final_array = [], []

		for nb in range(n_percent_batch):
			feature, label = next(batch_data)
			final_array[0].append(feature)
			final_array[1].append(label)

		print("Data is Batched....")

		return final_array




def train(network_architecture, training_data, lr=0.001, 
	batch_size=8, training_epochs=75, display_step=5):

	vae = VAE(network_architecture, learning_rate=lr, batch_size=8)
	#start the network learning
	#training cycle
	for epochs in range(training_epochs):
		avg_cost = 0

		#iterate through batch range
		for batch_num in range(len(training_data[0])):
			#get features and labels
			features, labels = training_data[0][batch_num],  training_data[1][batch_num]

			#fit the training batch data
			cost = vae.partial_fit(features)

			#compute average loss
			avg_cost += cost / Data.n_samples * batch_size

			#display logs per epoch and steps
			if epochs % display_step == 0:
				print("Epochs:{}, cost={}".format(epochs+1, avg_cost))

	return vae #returning whole class object


def main():
	#check if numpy records were in the directory
	records = "train_dataset_dump.npy"
	if os.path.isfile(records):
		#carry the network architecture
		print("Numpy data for input records is Found!!!")
		data = Data(records)

		#check if data are correctly batched
		#no repeatable features
		#<!--Batch data is working correctly-->
		training_set = data.data_as_batch(90, "TRAIN", data.d) #Shape: [[features], [labels]]
		print("Length of the Training set for FEATURES={}, LABELS={}".format(len(training_set[0]), len(training_set[1])))

		#testing set
		testing_set = [data.d[0][data.batch_ends: ], data.d[1][data.batch_ends: ]]
		print("Number of records in Test Set: FEATURES={}, LABELS={}".format(len(testing_set[0]), len(testing_set[1])))

		#test data as batch
		testing_set = data.data_as_batch(100, "TEST", testing_set) #shape: [[features], [labels]]

		#iterate through the training set and start training the network
		network_architecture = dict(
			n_hidden_recog_1 = 500, #1st layer encoder neurons
			n_hidden_recog_2 = 500, #2nd layer encoder neurons
			n_hidden_gener_1 = 500, #1st layer decoder neurons
			n_hidden_gener_2 = 500, #2nd layer decoder neurons
			n_input=len(training_set[0][0][0]), #input image Size ie.., 784(24x24)
			n_z=20 #dimensionality of latent space
		)

		vae = train(network_architecture, training_set)

		#get the first batch data of test set
		test_x_sample = testing_set[0][0]

		test_reconstruct = vae.reconstruct(test_x_sample)

		plt.figure(figsize=(8, 12))

		for i in range(8):
			plt.subplot(8, 2, 2*i+1)
			plt.imshow(test_x_sample[i].reshape(28, 28), vmin=0, vmax=1, cmap='gray')
			plt.title("Test Input")
			plt.colorbar()
			plt.subplots(5, 2, 2*i+2)
			plt.imshow(test_reconstruct[i].reshape(28, 28), vmin=0, vmax=1, cmap='gray')
			plt.title("Reconstruction")
			plt.colorbar()

		plt.tight_layout()


		#store the class weights as a Pickle Dump
		import pickle
		with open("vae.pickle","w") as input:
			pickle.dump(vae, input, pickle.HIGHEST_PROTOCOL)


		
	else:
		print("There is no numpy records files found in the present directory")
		print("Please run **input_data.py** script before running this main Script")

if __name__ == '__main__':
	main()