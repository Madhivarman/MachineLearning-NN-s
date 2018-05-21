import matplotlib.pyplot as plt 
import numpy as np 
import cv2
import tensorflow as tf

test_data = np.load('test_data.npy')

#load the model
sess = tf.Session()
saver = tf.train.import_meta_graph('dogsvscats-0.001-2conv-basic.model.meta')
# Step-2: Now let's load the weights saved using the restore method.
saver.restore(sess, tf.train.latest_checkpoint('./'))

fig = plt.figure()

for num,data in enumerate(test_data[:12]):

	img_num = data[1]
	img_data = data[0]

	y = fig.add_subplot(3,4,num+1)
	orig = img_data
	data = img_data.reshape(50,50,1)
	model_out = model.predict([data])[0]

	if np.argmax(model_out) == 1: str_label='Dog'
	else:str_label = 'Cat'

	y.imshow(orig,cmap='gray')
	plt.title(str_label)
	y.axes.get_xaxis().set_visible(False)
	y.axes.get_yaxis().set_visible(False)


plt.show()
plt.savefig('demo.jpg')