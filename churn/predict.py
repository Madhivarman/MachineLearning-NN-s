#import necessary libraries
import tensorflow as tf 
import numpy as np

#load the numpy file
#the length of the test dataset is 666
test_input_data = np.load('test_data_data.npy')
test_target_data = np.load('test_data_target.npy')

#start tensorflow Session
sess = tf.Session()
saver = tf.train.import_meta_graph('saved_model/model.ckpt.meta')
saver.restore(sess, tf.train.latest_checkpoint('./'))

#now lettes access the placeholders saved in the stored in a graph 
#create a new feed-dict to the new data
graph = tf.get_default_graph()

X = graph.get_tensor_by_name("input_variables/ Placholder:0")
pred = graph.get_tensor_by_name("model_metrics/Round:0")


print(sess.run(pred, feed_dict={X:[test_data_target]}))