import tensorflow as tf 
import numpy as np 
from tensorflow.contrib.factorization import KMeans

#download the dataset
from tensorflow.examples.tutorials.mnist import input_data
mnist  = input_data.read_data_sets("/tmp/data/",one_hot=True)
full_data_x = mnist.train.images

#ignore all GPU's available, doesn't give much effect in the algorithm
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

#parameters
epochs = 50
batch_size = 64 #give batch size depends upon the Memory capacity your system does
num_clusters = 10
num_classes = 10
num_features =  784

with tf.name_scope("Input_Features"):
	x = tf.placeholder(tf.float32,shape=[None,num_features],name="Input")
	y_ = tf.placeholder(tf.float32,shape=[None,num_classes],name="Output")

#kmeans
with tf.name_scope("KMeans_Architecture"):
	Kmeans = KMeans(inputs=x,
		num_clusters=num_clusters,
		distance_metric='cosine',
		use_mini_batch=True)

#Building a graph
training_graph =  Kmeans.training_graph()

if len(training_graph) > 6:
	(all_scores, cluster_idx, scores, cluster_centers_initialized,
		cluster_center_var, init_op, train_op) = training_graph

else:
	(all_scores, cluster_idx, scores, cluster_centers_initialized,
		init_op, train_op) = training_graph

cluster_idx = cluster_idx[0]
avg_distance = tf.reduce_mean(scores)

#initialize all variables
init_vars = tf.global_variables_initializer()

#start tensorflow session
sess =  tf.Session()

#run the initializer
sess.run(init_vars, feed_dict={x:full_data_x})
sess.run(init_op,feed_dict={x: full_data_x})

#Training
for i in range(1, epochs+1):
	_, d, idx = sess.run([train_op,avg_distance, cluster_idx],
		feed_dict={x:full_data_x})

	if i%10 == 0 or i == 1:
		print("Step {}, Avg-Distance:{}".format(i,d))

#assign a lable to each centroid
#count total number of labels per centroid, using lable for each training

counts = np.zeros(shape=(k,num_classes))
for i in range(len(idx)):
	counts[idx[i]] += mnist.train.lable[i]

#assign most frequent label to centroid
lables_map = [np.argmax(c) for c in counts]
lables_map = tf.convert_to_tensor(lables_map)

#lookup: centroid_id -> label
cluster_label = tf.nn.embedding_lookup(lables_map, cluster_idx)

#compute accuracy
correct_prediction = tf.equal(cluster_label, tf.cast(tf.argmax(Y,1), tf.int32))
accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#test accuracy
test_x, test_y = mnist.test.images, mnist.test.lables
print("Test Accuracy:{}".format(sess.run(accuracy_op, feed_dict={x:test_x, y_:test_y})))