import keras
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, Dropout
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.utils import plot_model
from keras import backend as K

K.set_image_data_format('channels_last') 

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

def getnetwork():

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

model = getnetwork()
plot_model(model, to_file='model.png')