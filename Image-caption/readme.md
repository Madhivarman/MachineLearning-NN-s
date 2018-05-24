### Image Caption Using Neural Network

**Image dataset** : Flickr8k Image dataset

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

In this method, we going to merge two neuralnetwork as a one and produce caption. The final layer called **Decoder** which both feature extractor and sequence processor output a fixed length of the vector. They are merged together and preprocessed by **Dense layer** to make final predictions

The network architecture look like this

__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to
==================================================================================================
input_2 (InputLayer)            (None, 32)           0
__________________________________________________________________________________________________
input_1 (InputLayer)            (None, 4096)         0
__________________________________________________________________________________________________
embedding_1 (Embedding)         (None, 32, 256)      2662400     input_2[0][0]
__________________________________________________________________________________________________
dropout_1 (Dropout)             (None, 4096)         0           input_1[0][0]
__________________________________________________________________________________________________
dropout_2 (Dropout)             (None, 32, 256)      0           embedding_1[0][0]
__________________________________________________________________________________________________
dense_1 (Dense)                 (None, 256)          1048832     dropout_1[0][0]
__________________________________________________________________________________________________
lstm_1 (LSTM)                   (None, 256)          525312      dropout_2[0][0]
__________________________________________________________________________________________________
add_1 (Add)                     (None, 256)          0           dense_1[0][0]
                                                                 lstm_1[0][0]
__________________________________________________________________________________________________
dense_2 (Dense)                 (None, 256)          65792       add_1[0][0]
__________________________________________________________________________________________________
dense_3 (Dense)                 (None, 10400)        2672800     dense_2[0][0]
==================================================================================================
Total params: 6,975,136
Trainable params: 6,975,136
Non-trainable params: 0
__________________________________________________________________________________________________




**NOTE:** I trained the network with Nvidia GTX 1050 GPU, 16gigs RAM took a hour to train a model. If you train this model on CPU it could take couple hours to train the  whole network
