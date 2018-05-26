#kaggle competition
"""
	Ultrasound Nerve Segmentation: https://www.kaggle.com/c/ultrasound-nerve-segmentation
"""
#import necessary libariry
import numpy as np 
import matplotlib.pyplot as plt 
import os
import cv2

from sklearn.model_selection import train_test_split
from random import shuffle
from tqdm import tqdm

img_size = 50

def create_label(image_name):

	name_split = image_name.split('.')[0]
	return name_split

def convert_data_into_numpy_array(file_path,data_to_convert):
	data_as_list = []
	#data_to_convert is a list
	for images in tqdm(data_to_convert):
		full_image_path = file_path + "/" + images
		#create a label
		label = create_label(images)
		#read the image
		img = cv2.imread(full_image_path,cv2.IMREAD_GRAYSCALE)
		img = cv2.resize(img,(img_size,img_size))

		data_as_list.append([np.array(img),np.array(label)])

	shuffle(data_as_list)

	filename_to_save = '{}.npy'.format(file_path)
	np.save(filename_to_save,data_as_list)

def read_image(image):

	img = cv2.imread(image,cv2.IMREAD_GRAYSCALE)
	img = cv2.resize(img,(img_size,img_size))

	return img

def prepare_the_train_dataset(filepath,train_image_list):
	#the train images has 11270, so 5635 image set we have
	train_data_as_list = []
	train_mask_data_as_list = []

	for image_name in tqdm(train_image_list):
		full_image_path = filepath + "/" + image_name
		"""
			To check if the original image and mask image occurs concurrently print the image_name.
			It appears be that the image and mask is not appeared simulteanously. so create a dataset by this method
		"""
		if "mask" in image_name:
			continue
		else:
			orig_img = create_label(image_name)
			mask = orig_img + "_mask.tif"
			mask_img = create_label(mask)

			orig_img_values = read_image(full_image_path)
			mask_img_values = read_image(filepath+"/"+mask)

			#append to the list
			train_data_as_list.append([np.array(orig_img_values),np.array(orig_img)])
			train_mask_data_as_list.append([np.array(mask_img_values),np.array(mask_img)])

	#check if the length of both train, train_mask list is same
	if len(train_data_as_list) == len(train_mask_data_as_list):
		np.save("train.npy",train_data_as_list)
		np.save("train_mask.npy",train_mask_data_as_list)

	else:
		print("Length of Original Image and Mask Image is not same. Check Once Again..!")

def main():
	#training file path
	train_file_path = 'train'
	train_images = os.listdir(train_file_path)
	#test file path
	test_file_path = 'test'
	test_images = os.listdir(test_file_path)

	print("Length of Train Images:{train}".format(train = len(train_images)))
	print("Length of Test Images:{test}".format(test = len(test_images)))

	#convert data into numpy array
	"""
		The train images has two image type: Normal, Mask_Image.
		we need to separate normal, and mask image individually and map together atlast to train a images
	"""
	prepare_the_train_dataset(train_file_path,train_images)
	#convert_data_into_numpy_array(test_file_path,test_images)


if __name__ == '__main__':
	main()