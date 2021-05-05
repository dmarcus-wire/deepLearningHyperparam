# import the necessary packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# define model architecture
# 4 parameters...this is what we want to tune
# nodes in fc layer 1
# nodes in fc layer 2
# dropout in fc layer
# learning rate in fc layer

def get_mlp_model(hiddenLayerOne=784, hiddenLayerTwo=256,
	dropout=0.2, learnRate=0.01):
	# initialize a sequential model and add layer to flatten the
	# input data
	model = Sequential()
	model.add(Flatten())

	# add two stacks of FC => RELU => DROPOUT
	model.add(Dense(hiddenLayerOne, activation="relu",
		input_shape=(784,))) # no. of nodes == layer 1
	model.add(Dropout(dropout))
	model.add(Dense(hiddenLayerTwo, activation="relu")) # no. of nodes == layer 2
	model.add(Dropout(dropout))

	# add a softmax layer on top
	# 10 possible MNIST class labels
	# 10 softmax classifier distribution
	model.add(Dense(10, activation="softmax"))

	# compile the model
	# learning rate will be tuning
	# loss method crossentropy is typical for classification
	model.compile(
		optimizer=Adam(learning_rate=learnRate),
		loss="sparse_categorical_crossentropy",
		metrics=["accuracy"])

	# return compiled model
	return model