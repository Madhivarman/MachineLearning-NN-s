import os
import numpy as np 
from PIL import Image


class ReadDataset():

    def __init__(self):
        """declaring initially to store images as a array"""
        self.datadump = []
        self.categories = None
        self.labeldump = []
        self.imgsize = 64
        self.data = [[], []]

    def findTotalCategories(self, filepath):
        labels = os.listdir(filepath)
        self.categories = len(labels)


    def readImages(self, filepath,bs,one_hot=True):

        print("One Hot Encoding is enabled for Image Tensors:{}".format(one_hot))
        print("-" * 15)

        labels = os.listdir(filepath)

        #defining the image dimension
        sample_dim = (bs,) + (self.imgsize, self.imgsize, 3)
        sample = np.empty(sample_dim, dtype=np.float32)


        for num, l in enumerate(labels):
            print("STARTED READING IMAGE FOR LABEL:{}".format(l))
            #get the filename
            dirpath = filepath + "/" + l
            #get total number of images
            image_name = os.listdir(dirpath)

            for n, img in enumerate(image_name):
                #get the extension
                extension = img.split(".")
                if len(extension) <= 1:
                    pass
                elif (extension[1] != 'png'):
                    #print("Found some other file:{}".format(extension[0]))
                    pass
                else:
                    #get the full path
                    full_path = dirpath + "/" + img
                    #open an image
                    image_read = Image.open(full_path).convert('RGB')

                    #resize
                    image_read = image_read.resize((self.imgsize, self.imgsize), Image.ANTIALIAS)
                    #convert into numpy array
                    pix = np.asarray(image_read) #[64, 64]
                    #pix = pix[:,:,0] #get only image data(from first layer of matrix), ignore the next layer(plain card)

                    #convert into records
                    #pix = np.reshape(pix, (np.product(self.imgsize * self.imgsize)))

                    #store it into dump
                    if one_hot:
                        #scale the range between 0 - 1
                        norm = (pix / 127.5) - 1
                        #self.datadump.append(norm)
                        sample[n,...] = norm

                    else:
                        #simply append
                        #self.datadump.append(pix)
                        pass

        return sample




rd = ReadDataset()

image_path = "animeface-character-dataset/animeface-character-dataset/thumb"

#finding total number of labels
total_labels = rd.findTotalCategories(image_path)
print("Number of Labels in the Dataset:{}".format(rd.categories))

#read the input images
batch_size = 14720
inp_dump = rd.readImages(image_path,  batch_size,one_hot=True)

#convert into records
rd.data[0] = rd.datadump #input array dump
rd.data[1] = rd.labeldump #label dump

#save it as numpy dump
np.save('train_dataset_dump', rd.data)
np.save('2d-data', inp_dump)
print("Image Records are locally Saved!!!")