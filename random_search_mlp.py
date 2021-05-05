# USAGE
# python random_search_mlp.py

# import tensorflow and fix the random seed for better reproducibility
import tensorflow as tf
tf.random.set_seed(42)

# import the necessary packages
from submodules.mlp import get_mlp_model
# Keras Classifier is going to wrap the MLP model and enable scikitlearn hyperparameter tuning
# we could apply grid search, random search to find optimal values
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
# enables scikit learn to randomly search a distribution or hyperparamters
from sklearn.model_selection import RandomizedSearchCV
from tensorflow.keras.datasets import mnist

# load the MNIST dataset
print("[INFO] downloading MNIST...")
# load MNIST from disk
((trainData, trainLabels), (testData, testLabels)) = mnist.load_data()

# scale data to the range of [0, 1]
trainData = trainData.astype("float32") / 255.0
testData = testData.astype("float32") / 255.0

# wrap our model into a scikit-learn compatible classifier
print("[INFO] initializing model...")
# instantiate model build_fn is required to build and compile model
# scikitlearn will create new instance of the model using build_fn
model = KerasClassifier(build_fn=get_mlp_model, verbose=0)

# define a grid of the hyperparameter search space numer of nodes
hiddenLayerOne = [256, 512, 784]
hiddenLayerTwo = [128, 256, 512]
learnRate = [1e-2, 1e-3, 1e-4]
dropout = [0.3, 0.4, 0.5]
batchSize = [4, 8, 16, 32]
epochs = [10, 20, 30, 40]

# create a dictionary from the hyperparameter grid
# maps names to parameters they can use from model architecture definition
grid = dict(
	# maps to model set in the build_fn
	hiddenLayerOne=hiddenLayerOne, # maps to mlp.py hiddenLayerOne
	learnRate=learnRate, # maps to mlp.py learnRate
	hiddenLayerTwo=hiddenLayerTwo, # maps to mlp.py hideenayerTwo
	dropout=dropout, # maps to mlp.py dropout
	# these are reserved
	batch_size=batchSize, # maps to training parameter
	epochs=epochs # maps to training parameter
)

# initialize a random search with a 3-fold cross-validation and then
# start the hyperparameter search process
print("[INFO] performing random search...")
# randomize search, model = keras classifier, -1 is parallel jobs, 3 folds in the cross-validation
# parameter distribution grid
searcher = RandomizedSearchCV(estimator=model, n_jobs=-1, cv=3,
	param_distributions=grid, scoring="accuracy")
searchResults = searcher.fit(trainData, trainLabels)

# summarize grid search information
bestScore = searchResults.best_score_
bestParams = searchResults.best_params_
print("[INFO] best score is {:.2f} using {}".format(bestScore,
	bestParams))

# extract the best model, make predictions on our data, and show a
# classification report
print("[INFO] evaluating the best model...")
bestModel = searchResults.best_estimator_
accuracy = bestModel.score(testData, testLabels)
print("accuracy: {:.2f}%".format(accuracy * 100))