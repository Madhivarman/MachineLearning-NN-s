#necessary libraries

import urllib.request
import collections
import math
import os
import random 
import zipfile
import datetime as dt 

import numpy as np 
import tensorflow as tf 


data_index = 0 #global variable

valid_size = 16
valid_window = 100
num_sampled = 64 #number of negative samples to sample

def download_the_dataset(filename,url,expected_bytes):

	print("Downloading the file..Wait for a seconds")
	#download the file if not present
	if not os.path.exists(filename):
		filename,_ = urllib.request.urlretrieve(url +  filename,filename)

	#status info
	status_info = os.stat(filename)

	if status_info.st_size ==  expected_bytes:
		print("File is found and verified")

	else:
		print(status_info.st_size)

		raise Exception(
			'Failed to Verify \t' + filename + '.can you get to it with  a browser?')


	return filename


def read_data(filename):

	with zipfile.ZipFile(filename) as fp:
		data = tf.compat.as_str(fp.read(fp.namelist()[0])).split()

	return data

"""

		The first step is setting a "counter" list which will store the no.of.times
		word is found within a dataset. Any words not within top 10,000 most common words
		are marked as UNK(unknown). 

		we create a dictionary=dict() which is populated by keys correspond to each unique word.
		The value assigned to each key is simply a increasing order. ie.., the most common value 
		will assign as 1 and after 10,000 words UNK.
		
		Next iterate through each word which has same length as in the dataset. Then we going to
		match if word is not in the dictionary, increase the "unk_count". If the word is in the dictionary
		then append word count in the dictionary as a index for easy table lookup

"""
def build_dataset(words,n_words):

	count = [["UNK",-1]]
	count.extend(collections.Counter(words).most_common(n_words-1))
	dictionary = dict() #create a dictionary

	for word,_ in count:
		dictionary[word] = len(dictionary)

	data = list()
	unk_count = 0

	for word in words:

		if word in dictionary:
			index = dictionary[word]
		else:
			index = 0
			unk_count += 1

		data.append(index)

	count[0][1] = unk_count
	reverse_dictionary = dict(zip(dictionary.values(),dictionary.keys()))

	return data, count, dictionary,reverse_dictionary


def generate_batch(data,batch_size,number_skip,skip_window):
	global data_index
	assert batch_size % number_skip == 0
	assert number_skip <= 2 * skip_window

	batch = np.ndarray(shape=(batch_size),dtype=np.int32)
	context = np.ndarray(shape=(batch_size,1),dtype=np.int32)
	span = 2*skip_window + 1 #[skip_window,input_window,skip_window]

	buffers = collections.deque(maxlen=span)

	for _ in range(span):
		buffers.append(data[data_index])
		data_index = (data_index + 1) % len(data)

	for i in range(batch_size // number_skip):
		target = skip_window #input word at center of the buffer
		targets_to_avoid = [skip_window]

		for j in range(number_skip):

			while target in targets_to_avoid:
				target = random.randint(0, span-1)

			targets_to_avoid.append(target)
			batch[i * number_skip + j] = buffers[skip_window] #input word
			context[i *  number_skip + j,0] = buffers[target] #context words

		buffers.append(data[data_index])
		data_index = (data_index + 1) % len(data)

	#backtrack a little bit to avoid skipping words in end of the batch
	data_index = (data_index + len(data) - span) % len(data)

	return batch,context



def run(init,graph,num_steps,data,batch_size,num_skips,skip_wndw,train_inputs,train_context,optimizer,cross_entropy,similarity,valid_examples,normalized_embeddings,reverse_dictionary):

	with tf.Session(graph=graph) as session:

		#initialize all variable before we using them
		init.run()
		print("Initialized All Variables")

		average_loss = 0 #initially

		for step in range(num_steps):

			batch_inputs,batch_context = generate_batch(
				data,batch_size,num_skips,skip_wndw)

			feed_dict = {train_inputs:batch_inputs,train_context:batch_context}

			# we perform one update step by step evaluating the optimizer op
			_,loss_val = session.run([optimizer,cross_entropy],feed_dict=feed_dict)
			average_loss += loss_val

			if step % 2000 == 0:
				if step > 0:
					average_loss /= 2000

				#average loss is an estimate of the loss over last 2000 batches
				print("Average loss at step:{s},average loss:{avg}".format(s=step,avg=average_loss))
				average_loss = 0 #reinitialize

			if step % 10000 == 0:
				sim = similarity.eval()
				for i in range(valid_size):
					valid_word = reverse_dictionary[valid_examples[i]]
					top_k = 8 #top 8 words
					nearest = (-sim[i, :]).argsort()[1:top_k+1]
					log_str = "Nearest to {original_word}:\t".format(original_word = valid_word)

					for k in range(top_k):
						close_word = reverse_dictionary[nearest[k]]
						log_str = "{before} {after}".format(before=log_str,after=close_word)

					print(log_str)

		final_embeddings = normalized_embeddings.eval()


def main():
	#download the dataset
	url_to_download = 'http://mattmahoney.net/dc/'
	datast = download_the_dataset('text8.zip',url_to_download,31344016)

	#read the data
	vocabulary = read_data(datast)
	print(vocabulary[:10]) #print sample 10 words

	"""
		Return vocabulary contains plain text of words. we need to choose top
		10,000 words to include in our embedding vector. To do so, gather all unique 
		words and index them with unique integer value. It's equal to one hot type input
		for word.

		Loop through every word in the vocabulary and assign it to the unique integer.This
		is easy for lookup table
	"""
	vocabulary_size = 10000
	data, count, dictionary, reverse_dictionary = build_dataset(vocabulary,vocabulary_size) #the length of dictionary is 10,000

	#intialize and declare parameters that are needed to train the network
	batch_size = 128
	embedding_size = 300 
	skip_wnd = 2
	num_skips = 2 #how many times to reuse an input to generate a  model

	valid_examples = np.random.choice(valid_window,valid_size,replace = False) #randomly select 16 index numbers within 100.

	graph = tf.Graph()

	with graph.as_default():

		#input data
		train_inputs = tf.placeholder(tf.int32,shape=[batch_size])
		train_context = tf.placeholder(tf.int32,shape=[batch_size,1])
		valid_dataset = tf.constant(valid_examples,dtype=tf.int32)

		#look for embeddings for inputs
		#shape(10000,300) variable with random uniform distribution between -1.0 and 1.0
		embeddings = tf.Variable(tf.random_uniform([vocabulary_size,embedding_size],-1.0,1.0)) #shape(10000,300)
		embed = tf.nn.embedding_lookup(embeddings,train_inputs)

		#softmax variables
		weights = tf.Variable(
			tf.truncated_normal([embedding_size,vocabulary_size],
				stddev=1.0/math.sqrt(embedding_size)))

		biases = tf.Variable(tf.zeros([vocabulary_size])) #(10000)
		hidden_out = tf.transpose(tf.matmul(tf.transpose(weights),tf.transpose(embed))) + biases

		#convert train_context to a one-hot format
		train_one_hot = tf.one_hot(train_context,vocabulary_size)
		cross_entropy = tf.reduce_mean(
						tf.nn.softmax_cross_entropy_with_logits(logits=hidden_out,labels=train_one_hot)
						)
		#opitmizer
		optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(cross_entropy)

		#compute the cosine similarity between minibatch examples and all embeddings.
		#L2 normalization is used

		"""
			we calculate the L2 norm of each vector using the tf.square(), tf.reduce_sum() and tf.sqrt() functions 
			to calculate the square, summation and square root of the norm, respectively
		"""

		norm = tf.sqrt(tf.reduce_mean(tf.square(embeddings), 1, keep_dims=True))
		normalized_embeddings = embeddings / norm
		valid_embeddings = tf.nn.embedding_lookup(
			normalized_embeddings,valid_dataset)

		similarity = tf.matmul(
			valid_embeddings,normalized_embeddings,transpose_b=True)

		#add variable initializer
		init = tf.global_variables_initializer()


	#paramaters
	num_steps = 50000
	softmax_start_time = dt.datetime.now()
	run(init,graph,num_steps,data,batch_size,num_skips,skip_wnd,train_inputs,train_context,optimizer,cross_entropy,similarity,valid_examples,normalized_embeddings,reverse_dictionary)
	softmax_end_time = dt.datetime.now()

	print("Softmax method took {} minutes to run 50000 iterations".format((softmax_end_time - softmax_start_time).total_seconds()))


# Whole program took 13 minutes to run 5000 iterations in GPU (GTX 1050). In CPU it would take hours to complete the iterations.
if __name__ == '__main__':
	main()
	
