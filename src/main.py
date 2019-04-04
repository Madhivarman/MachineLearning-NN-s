import os
import numpy as np

class Data():
	"""Initial Declaration"""
	def __init__(self, dumpname):
		self.d = np.load(dumpname)
		self.batch_size = None
		self.n_samples = None
		self.batch_start, self.batch_ends = 0, self.batch_size

	def get_data_as_batch(self, batch_size, total_batches):

		#print(self.batch_size, self.batch_start, self.batch_ends)

		for b in range(total_batches):
			data = self.d[self.batch_start: self.batch_ends]
			self.batch_start = self.batch_ends
			self.batch_ends += self.batch_start
			yield data

	def data_as_batch(self, batch_size):
		assert len(self.d[0]) == len(self.d[1]), "Length of Input and Label records should be same!"
		self.n_samples = len(self.d[0])
		self.batch_size = batch_size

		#find total number of batches
		total_batches = int(self.n_samples/self.batch_size)

		#custom functions to yield batches
		batch_data = self.get_data_as_batch(batch_size, total_batches)
		print(next(batch_data))


def main():
	#check if numpy records were in the directory
	records = "train_dataset_dump.npy"
	if os.path.isfile(records):
		#carry the network architecture
		print("Numpy data for input records is Found!!!")
		data = Data(records)
		batch_size = 8
		data.data_as_batch(batch_size)
		
	else:
		print("There is no numpy records files found in the present directory")
		print("Please run **input_data.py** script before running this main Script")

if __name__ == '__main__':
	main()