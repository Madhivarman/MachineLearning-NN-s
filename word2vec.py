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

def run(graph,num_steps,data,batch_size,num_skips,skip_wndw):

	with tf.Session(graph=graph) as session:

		#initialize all variable before we using them
		init.run()
		print("Initialized All Variables")

		avg_loss = 0 #initially

		for step in range(num_steps):

			batch_inputs,batch_context = generate_batch(
				data,batch_size,num_skips,skip_wndw)

			feed_dict = {train_inputs:batch_inputs,train_context:batch_context}

			

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

	valid_size = 16
	valid_window = 100
	valid_examples = np.random.choice(valid_window,valid_size,replace = False) #randomly select 16 index numbers within 100.
	num_sampled = 64 #number of negative samples to sample

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
		weights = tf.Variable(tf.truncated_normal([vocabulary_size,embedding_size],
			stddev=1.0/math.sqrt(embedding_size))) #(10000,300)

		biases = tf.Variable(tf.zeros([vocabulary_size])) #(10000)
		hidden_out = tf.transpose(tf.matmul(embed, tf.transpose(weights))) + biases

		#convert train_context to a one-hot format
		train_one_hot = tf.one_hot(train_context,vocabulary_size)
		cross_entropy = tf.reduce_mean(
						tf.nn.softmax_cross_entropy_with_logits(logits=hidden_out,labels=train_one_hot)
						)
		#opitmizer
		optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)

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

if __name__ == '__main__':
	main()
