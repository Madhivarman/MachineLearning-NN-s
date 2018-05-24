### Image Caption Using Neural Network

** Image dataset** : Flickr8k Image dataset

Download the dataset and to run the Program do the following

> python dataset.py

In the above file we used Keras Pre-trained VGG16() model for getting image features. In that we removed last layer called as Softmax layers which is used for classification purpose. We don't need any class types to this problem.

> python load.py

To clean the training description txt and map image and description for Training purpose. Basic Text preprocessing is done here.

Before you start Training a network, you should need to install some dependencies
-pydot
-graphviz

To install pydot

 > pip install pydot
 
To install graphviz

 > pip install graphviz

To run the Script

 > python train.py
