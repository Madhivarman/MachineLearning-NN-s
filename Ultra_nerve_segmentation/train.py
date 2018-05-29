#import necessary libraries
import numpy as np 
import keras
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, Dropout,Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K

import matplotlib.pyplot as plt 
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #to eliminate 	all the warning and loaded library

#global definition
img_row = 96
img_col = 96
smooth = 1


def create_conv_layer(f,stride,activationfn,padding,prevlayer,dropout):

	conv = Conv2D(f,stride,activation=activationfn,padding=padding)(prevlayer)
	conv = Dropout(dropout)(conv)
	conv = Conv2D(f,stride,activation=activationfn,padding=padding)(conv)

	return conv

def maxpooling_fn(prevlayer):

	return MaxPooling2D(pool_size=(2,2))(prevlayer)

def concatenate_fn(f,kernal,stride,padding,src,dest):

	return concatenate([Conv2DTranspose(f,kernal,strides=stride,padding=padding)(src),dest],axis=3)

def dice_coef(y_true,y_pred):

	y_true_f = K.flatten(y_true)
	y_pred_f = K.flatten(y_pred)

	intersection = K.sum(y_true_f * y_pred_f)

	return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true,y_pred):

	return -dice_coef(y_true,y_pred)


def getNetwork():

	inputs = Input((img_row,img_col,1))

	conv1 = create_conv_layer(32,(3,3),'relu','same',inputs,0.2)
	pool1 = maxpooling_fn(conv1)

	conv2 = create_conv_layer(64,(3,3),'relu','same',pool1,0.2)
	pool2 = maxpooling_fn(conv2)


	conv3 = create_conv_layer(128,(3,3),'relu','same',pool2,0.3)
	pool3 = maxpooling_fn(conv3)

	conv4 = create_conv_layer(256,(3,3),'relu','same',pool3,0.3)
	pool4 = maxpooling_fn(conv4)

	conv5 = create_conv_layer(512,(3,3),'relu','same',pool4,0.3)
	pool5 = maxpooling_fn(conv5)

	up6 = concatenate_fn(256,(2,2),(2,2),'same',conv5,conv4)
	conv6 = create_conv_layer(256,(3,3),'relu','same',up6,0.3)

	up7 = concatenate_fn(128,(2,2),(2,2),'same',conv6,conv3)
	conv7 = create_conv_layer(128,(3,3),'relu','same',up7,0.3)

	up8 = concatenate_fn(64,(2,2),(2,2),'same',conv7,conv2)
	conv8 = create_conv_layer(64,(3,3),'relu','same',up8,0.3)

	up9 = concatenate_fn(32,(2,2),(2,2),'same',conv8,conv1)
	conv9 = create_conv_layer(32,(3,3),'relu','same',up9,0.3)

	conv10 = Conv2D(1,(1,1),activation='sigmoid')(conv9)

	model = Model(inputs=[inputs],outputs=[conv10])

	model.compile(optimizer=Adam(lr=0.00001),loss=dice_coef_loss,metrics=[dice_coef])

	return model
	

def main():
	#load necessary files
	train_data = np.load('train.npy')
	mask_train_data = np.load('train_mask.npy')
	test_data = np.load('test.npy')

	#load inputs
	training_inputs = train_data[:-1500]
	test_training_inputs = train_data[-1500:]
	#mask inputs
	training_mask_inputs = mask_train_data[:-1500]
	test_training_mask_inputs = mask_train_data[-1500:]

	#mean,std 
	training_inputs_only = training_inputs[:,0]
	mean = np.mean(training_inputs_only)
	std = np.std(training_inputs_only)

	#test_mean,test_std
	test_inputs_only = test_training_inputs[:,0]
	test_mean = np.mean(test_inputs_only)
	test_std = np.std(test_inputs_only)

	#normalizing
	training_inputs_only = np.asarray([data-mean for num,data in enumerate(training_inputs_only)]) #(4135,96,96)
	training_inputs_only = np.asarray([data/std for num,data in enumerate(training_inputs_only)],dtype=np.float32) #(4135,96,96) type:numpy.ndarray

	#test normalizing
	test_inputs_only = np.asarray([data-test_mean for num,data in enumerate(test_inputs_only)])
	test_inputs_only = np.asarray([data/test_std for num,data in enumerate(test_inputs_only)],dtype=np.float32)

	#normalizing train inputs
	training_mask_inputs_only = training_mask_inputs[:,0] #numpy.ndarray
	training_mask_inputs_only /= 255 #scale masks to (0,1) type numpy.ndarray dtype=object

	training_mask_inputs_only = np.array(list(training_mask_inputs_only),dtype = np.float)
	#normalizing test inputs
	test_training_mask_inputs_only = test_training_mask_inputs[:,0]
	test_training_mask_inputs_only /= 255 #scale masks to (0,1) type numpy.ndarray

	test_training_mask_inputs_only = np.array(list(test_training_mask_inputs_only),dtype= np.float)

	#reshape the data 
	X = np.array([i for i in training_inputs_only]).reshape(-1,img_row,img_col,1) #(-1,96,96,1)
	X_test = np.array([i for i in test_inputs_only]).reshape(-1,img_row,img_col,1) #(-1,96,96,1) (1500,96,96,1)

	#reshape the output data
	Y = np.array([i for i in training_mask_inputs_only]).reshape(-1,img_row,img_col,1) #(-1,96,96,1)
	Y_test = np.array([i for i in test_training_mask_inputs_only]).reshape(-1,img_row,img_col,1) #(-1,96,96,1)

	print('-'*30)
	print("Creating and Compiling Model Inputs")
	print('-'*30)

	model = getNetwork()

	#print(model.summary())

	model_checkpoint = ModelCheckpoint("weights_best.h5",monitor='val_loss',save_best_only = True,verbose=1)

	print('-'*30)
	print("Fitting The Model..")
	print('-'*30)

	history = model.fit(X,Y,batch_size=64,epochs=5,validation_split=0.15,callbacks = [model_checkpoint])

	print('-'*30)

	print(history.history.keys())

	plt.plot(history.history['dice_coef'])
	plt.plot(history.history['val_dice_coef'])
	plt.title("Model Dice coefficient")
	plt.ylabel('Dice coefficient')
	plt.xlabel('Epochs')

	plt.legend(['train','validation'],loc='upper left')
	plt.savefig('model_accuracy.png',bbox_inches='tight')

	return


if __name__ == '__main__':
	main()