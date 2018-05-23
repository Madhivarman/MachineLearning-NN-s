import os
from pickle import dump
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Model,load_model

"""
	Keras use pretrained model VGG16. It download weights from the internet and use those pretrained weights.
	we will remove the last layer of the model called Softmax which is used for  classification. In this problem statement we are 
	not intrested in classifying the image.
"""


#extract features from the dataset
def extract_features(datafile):
	#load model
	model = VGG16();
	model.layers.pop()
	model = Model(inputs = model.inputs,outputs=model.layers[-1].output)
	#summary the model
	print("Model Summary:\n{summary}".format(summary = model.summary()))

	features = dict()

	for name in os.listdir(datafile):
		filename = dataset +"/"+name #full path file
		image = load_img(filename,target_size=(224,224))
		image = img_to_array(image) #image to array
		image = image.reshape((1,image.shape[0],image.shape[1],image.shape[2])) #reshape the image
		image = preprocess_input(image) #preprocess the input
		feature = model.predict(image,verbose=0) #get the features
		image_id = name.split('.')[0] #get the name
		features[image_id] = feature

	return features

#dataset file
dataset = 'Flickr8k_Dataset/Flicker8k_Dataset'
features = extract_features(dataset)

print("Extracted Features:{fea}".format(fea = len(features)))

#save the pickle file
dump(features,open('features.pkl','wb'))
print("Model is saved")