**Ultra Nerve Segmentation**

This Project is inspired from this Kaggle Competition[https://www.kaggle.com/c/ultrasound-nerve-segmentation]. We need to train a model
identify **Barchial Plexus** nerve in ultrasound images. To know more about Barchial Plexus, read this [https://en.wikipedia.org/wiki/Brachial_plexus].


**Requirements**
- Python 3.x
- Numpy
- Tensorflow tflayer
- Opencv
- Matplotlib

**Environment**

I trained this model on  Windows 10, Nvidia GTX 1050 GPU. 

To run this project download the dataset and clone this git repository.

> python data_preprocessing.py

Run the above file  to convert the images into trainable format.  

To train a network run the  **train.py** file

> python train.py
