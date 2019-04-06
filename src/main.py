import os
import numpy as np

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

	def data_as_batch(self, size, batch_name):

		assert len(self.d[0]) == len(self.d[1]), "Length of Input and Label records should be same!"
		self.n_samples = len(self.d[0])

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
		

def main():
	#check if numpy records were in the directory
	records = "train_dataset_dump.npy"
	if os.path.isfile(records):
		#carry the network architecture
		print("Numpy data for input records is Found!!!")
		data = Data(records)
		training = data.data_as_batch(90, "TRAIN")
		
	else:
		print("There is no numpy records files found in the present directory")
		print("Please run **input_data.py** script before running this main Script")

if __name__ == '__main__':
	main()