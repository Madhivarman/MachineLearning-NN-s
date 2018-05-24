#import necessary libraries
import numpy as np 
from pickle import load 
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical,plot_model
from keras.models import Model 
from keras.layers import Input,Dense,LSTM,Embedding,Dropout
from keras.layers.merge import add
from keras.callbacks import ModelCheckpoint

"""
	we gonna merge RNN and image to the Fully Functional

	Photo Extractor: 16-layer VGG model pre-trained on Imagenet dataset.
Extracted features from the image is treated as input.
	
	Sequence Processor: Word Embedding layer for handling the  text input, followed by
Long Short-Term Memory(LSTM) RNN
	
	Decoder: Both the feature extractor and decoder output a fixed-length vector.These are
merged together and processed by a Dense layer to make final predictions
"""	

def load_dataset(filename):
	docs = open(filename).read() #read the file
	data = list() #create a list

	for line in docs.split('\n'):
		#skip empty lines
		if len(line) < 1:
			continue

		#get image identifier
		identifier = line.split('.')[0]
		data.append(identifier)

	return set(data) #remove duplicates

def load_description(clean_description,train):
	docs = open(clean_description).read()
	descr = dict()

	for lines in docs.split('\n'):
		#split the line by white space
		desc = lines.split()
		#split_id from description
		image_id,image_description = desc[0],desc[1:]
		#remove images not in the dataset
		if image_id in train:
			#create list
			if image_id not in descr:
				descr[image_id] = list()

			#wrap decriptions in tokens
			description = 'startseq'+' '.join(image_description)+'endseq'
			#store
			descr[image_id].append(description)

	return descr

def load_photo_features(features,dataset):
	#load all features
	all_features = load(open(features,'rb'))
	#filter features
	features = {k: all_features[k] for k in dataset}

	return features

def to_lines(descriptions):
	all_desc = list()

	for key in descriptions.keys():
		[all_desc.append(d) for d in descriptions[key]]

	return all_desc


def prepare_tokenizer(tokenize_features,dataset):

	lines = to_lines(tokenize_features)
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(lines)

	return tokenizer


def find_maximum_length(descriptions):
	lines = to_lines(descriptions)
	return max(len(d.split()) for d in lines)


def create_sequences(tokenizer,maximum_length,train_descr,train_feat,vocab_size):
	x1,x2,y = list(),list(),list()
	#walk through each identifier
	for key,desc_list in train_descr.items():
		#walk through each description
		for desc in desc_list:
			#convert text to  word
			seq = tokenizer.texts_to_sequences([desc])[0]
			#split one sequence into multiple X,y pairs
			for i in range(1,len(seq)):
				#split into input and output pair
				in_seq,out_seq = seq[:i],seq[i]
				#pad input sequence
				in_seq = pad_sequences([in_seq],maxlen=maximum_length)[0]
				#encode output sequence
				out_seq = to_categorical([out_seq],num_classes = vocab_size)[0]
				#store
				x1.append(train_feat[key][0])
				x2.append(in_seq)
				y.append(out_seq)

	return np.array(x1),np.array(x2),np.array(y)


def load_file_for_test_case(test_filename,description,tokenizer,max_length,vocab_size):

	test = load_dataset(test_filename) #train the file
	print(" The length of the train dataset:{len}".format(len=len(test)))

	test_descriptions = load_description(description,test)
	print(" Descriptions:{desc_len}".format(desc_len=len(test_descriptions)))

	#photo features
	test_features = load_photo_features('features.pkl',test)
	print(" Photo features:{feat_len}".format(feat_len = len(test_features)))

	X1test,X2test,ytest = create_sequences(tokenizer,max_length,test_descriptions,test_features,vocab_size)

	return X1test,X2test,ytest

def define_model(vocab_size,max_length):
	#feature extractor model
	inputs_1 = Input(shape=(4096,))
	fe_1  = Dropout(0.5)(inputs_1)
	fe_2 = Dense(256,activation='relu')(fe_1)

	#sequence model
	inputs_2 = Input(shape=(max_length,))
	se_1 = Embedding(vocab_size,256,mask_zero=True)(inputs_2)
	se_2 = Dropout(0.5)(se_1)
	se_3 = LSTM(256)(se_2)

	#decoder model
	decoder_1 = add([fe_2,se_3])
	decoder_2 = Dense(256,activation='relu')(decoder_1)
	outputs = Dense(vocab_size,activation='softmax')(decoder_2)

	#tie it together
	model = Model(inputs=[inputs_1,inputs_2],outputs=outputs)
	model.compile(loss='categorical_crossentropy',optimizer='adam')

	#summarize the model
	print("Model summary")
	print("---------------------------------")
	print(model.summary())
	#plot_model(model,to_file='model.png',show_shapes=True)

	return model

def main():
	#train dataset
	train_filename = 'Flickr8k_text/Flickr_8k.trainImages.txt'
	test_filename = 'Flickr8k_text/Flickr_8k.testImages.txt'

	train = load_dataset(train_filename) #train the file
	print(" The length of the train dataset:{len}".format(len=len(train)))

	train_descriptions = load_description('descriptions.txt',train)
	print(" Descriptions:{desc_len}".format(desc_len=len(train_descriptions)))

	#photo features
	train_features = load_photo_features('features.pkl',train)
	print(" Photo features:{feat_len}".format(feat_len = len(train_features)))

	#tokenizer
	tokenizer = prepare_tokenizer(train_descriptions,train)
	vocab_size = len(tokenizer.word_index) + 1
	print(" Vocabulary Size:{size}".format(size = vocab_size))

	#determine the maximum sequence length
	max_length = find_maximum_length(train_descriptions)
	print("Description Length:{}".format(max_length))

	#prepare sequences
	x1train,x2train,ytrain = create_sequences(tokenizer,max_length,train_descriptions,train_features,vocab_size)

	#load test_dataset
	print()
	print("*************************************************")
	print(" Now Preparing dataset for Test Features ")
	x1_test,x2_test,y_test = load_file_for_test_case(test_filename,'descriptions.txt',tokenizer,max_length,vocab_size)

	#fit model
	model = define_model(vocab_size,max_length)

	#define call checkpoint
	filepath = 'model-ep-{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'
	checkpoint = ModelCheckpoint(filepath,monitor='val_loss',verbose=1,save_best_only=True,mode='min')

	#fit the model
	model.fit([x1train,x2train],ytrain,epochs=20,verbose=2,callbacks=[checkpoint],validation_data=([x1_test,x2_test],y_test))

if __name__ == '__main__':
	main()

