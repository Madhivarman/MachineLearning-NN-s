import os
import numpy as np 
from PIL import Image


class ReadDataset():

    def __init__(self):
        """declaring initially to store images as a array"""
        self.datadump = []
        self.categories = None
        self.labeldump = []
        self.imgsize = 28
        self.data = [[], []]

    def findTotalCategories(self, filepath):
        labels = os.listdir(filepath)
        self.categories = len(labels)


    def readImages(self, filepath, one_hot=True):

        print("One Hot Encoding is enabled for Image Tensors:{}".format(one_hot))
        print("-" * 15)

        labels = os.listdir(filepath)

        for num, l in enumerate(labels):
            print("STARTED READING IMAGE FOR LABEL:{}".format(l))
            #get the filename
            dirpath = filepath + "/" + l
            #get total number of images
            image_name = os.listdir(dirpath)

            for n, img in enumerate(image_name):
                #get the full path
                full_path = dirpath + "/" + img
                #open an image
                image_read = Image.open(full_path).convert('LA')

                #resize
                image_read = image_read.resize((self.imgsize, self.imgsize), Image.ANTIALIAS)
                #convert into numpy array
                pix = np.array(image_read) #[28, 28]
                pix = pix[:,:,0]

                #convert into records
                pix = np.reshape(pix, (np.product(self.imgsize * self.imgsize)))

                #store it into dump
                if one_hot:
                    #scale the range between 0 - 1
                    norm = (pix - np.min(pix))/np.ptp(pix)
                    self.datadump.append(norm)

                else:
                    #simply append
                    self.datadump.append(pix)

                #create an one-hot encode label
                label = [0] * self.categories
                label[num] = 1

                self.labeldump.append(np.asarray(label)) #converting type list to array

                if n % 500 == 0 and n != 0:
                    print('Finished reading images for batch={}, total Images:{} '.format(n/500, n))

            print("*" * 15)




rd = ReadDataset()

image_path = "../../dataset/train"

#finding total number of labels
total_labels = rd.findTotalCategories(image_path)
print("Number of Labels in the Dataset:{}".format(rd.categories))

#read the input images
inp_dump = rd.readImages(image_path, one_hot=True)

#convert into records
rd.data[0] = rd.datadump #input array dump
rd.data[1] = rd.labeldump #label dump

#save it as numpy dump
np.save('train_dataset_dump', rd.data)
print("Image Records are locally Saved!!!")