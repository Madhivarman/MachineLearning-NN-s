# import necessary libraries
import numpy as np
import json
import urllib.request

url = "https://pythonprogramming.net/static/downloads/machine-learning-data/training_data-100k.json"
response = urllib.request.urlopen(url)

data = json.loads(response.read().decode())

xs = np.array(data["xs"])
ys = np.array(data["ys"])



#split into training and  testing
x_train = xs[:-10000]
y_train = ys[:-10000]

x_test = xs[-10000:]
y_test = ys[-10000:]



#define a model
import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout

model  = Sequential()
model.add(Dense(64,activation='relu',input_dim=6))
model.add(Dropout(0.5))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(3,activation='softmax'))

#optimizer
adam = keras.optimizers.Adam(lr=0.001)

#compile
model.compile(loss='categorical_crossentropy',
	optimizer = adam,
	metrics = ['accuracy'])

model.fit(x_train,y_train,epochs=15,batch_size=128)

score = model.evaluate(x_test,y_test,batch_size=128)

print("accuracy:{}".format(score))

#save the model
model.save('keras-pong-game-64x2-15epochs')
