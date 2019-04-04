# VAEApplication
Using VAE Neural Networks to do linear interpolations from one image to another Image. This Repository will be updated frequently

# File Structure
`input_data.py` - To convert image data into numpy records to train an Variational AutoEncoder and Decoder.
</br>
`dataset` - This folder contains image dataset. In this directory, images are stored in respective directory. for example, if there are two classes Cat and Dog. The program expects to have all cat images at cat directory and Dog images at dog category.
</br>
`network.py` - VAE Architecture is defined. 
</br>
`main.py` - To train a network

## Explanation
To Know about VAE, highly recommended to read the original paper and going through this link helps to understand the whole structure [Variational AutoEncoders Overview](http://kvfrans.com/variational-autoencoders-explained/)
