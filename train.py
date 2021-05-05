# USAGE
# python train.py

# baseline accuracy without any hyperparameter tuning
# default settings 784, 256, 0.2, .01 from mlp.py

# import tensorflow and fix the random seed for better reproducibility
import tensorflow as tf
# reproducibility, seed random initialization of the nodes in neural network
# gets it as close as possible
tf.random.set_seed(42)

# import the necessary packages
# defines model architecture
from submodules.mlp import get_mlp_model
# dataset from keras
from tensorflow.keras.datasets import mnist

# load the MNIST dataset
print("[INFO] downloading MNIST...")
# split the train test data
((trainData, trainLabels), (testData, testLabels)) = mnist.load_data()

# scale data to the range of [0, 1] from 0 to 255 original

trainData = trainData.astype("float32") / 255.0
testData = testData.astype("float32") / 255.0

# initialize our model with the default hyperparameter values
print("[INFO] initializing model...")
# the model with default parameters
model = get_mlp_model()

# train the network (i.e., no hyperparameter tuning)
print("[INFO] training model...")
# train the model with train and evalutate on test data
H = model.fit(x=trainData, y=trainLabels,
	validation_data=(testData, testLabels),
	batch_size=8,
	epochs=20)

# make predictions on the test set and evaluate it
print("[INFO] evaluating network...")
accuracy = model.evaluate(testData, testLabels)[1]
print("accuracy: {:.2f}%".format(accuracy * 100))