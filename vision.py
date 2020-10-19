import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV

from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout

from keras.wrappers.scikit_learn import KerasClassifier

import livelossplot
import time

plot_losses = livelossplot.PlotLossesKeras()

train_data = pd.read_csv("minst_data/train.csv")
test_data = pd.read_csv("minst_data/test.csv")

y_train = train_data["label"].values
X_train = train_data.drop("label", axis = 1).values / 255.0
X_test = test_data.values / 255.0

X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)
y_train = y_train.reshape(-1, 1)
# For RGB colors we use
# X_train = X_train.reshape(-1, 28, 28, 3)

# Display training examples
# plt.imshow(X_train[-1, :, :])

y_train = to_categorical(y_train, num_classes = 10)

# Split the training data
# random_seed = 2

# X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.1, random_state = random_seed)

# Create the generator of the model
def generate_model(convolution = False, num_filters1 = 32, num_filters2 = 32, 
	dropout_rate1 = 0.0, dropout_rate2 = 0.0, num_layers = 1, num_units = 128, dropout_rate = 0.0):

	model = Sequential()

	if convolution == True:

		model.add(Conv2D(filters = num_filters1, kernel_size = (5,5), padding = 'Same', activation ='relu', input_shape = (28,28,1)))
		model.add(Conv2D(filters = num_filters2, kernel_size = (5,5), padding = 'Same', activation ='relu'))
		model.add(MaxPool2D(pool_size = (2,2)))
		model.add(Dropout(dropout_rate1))
	
	# model.add(Flatten(input_shape = (28, 28, 1)))

	model.add(Flatten())

	while num_layers > 0:

		model.add(Dense(num_units, activation = 'relu'))
		num_layers -= 1
	
	model.add(Dropout(dropout_rate2))
	model.add(Dense(10, activation = 'softmax'))

	# Defining loss function and optimizer

	model.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])
	
	return model

# Define the model
BATCH_SIZE = 100
EPOCHS = 10

model = KerasClassifier(build_fn = generate_model, epochs = EPOCHS, batch_size = BATCH_SIZE)

# Define parameter space
param_grid = {
	"convolution": [True],
	"num_filters1": [32],
	"num_filters2": [32],
	"num_layers": [1],
	"num_units": [128],
	"dropout_rate1": [0.5],
	"dropout_rate2": [0.25]
}

grid = GridSearchCV(estimator = model, param_grid = param_grid, n_jobs = -1, cv = 3)

start = time.time()

grid_results = grid.fit(X_train, y_train)

end = time.time()

print("----------")
print("Training time: {} seconds".format(end - start))
print("Best score: {}".format(grid_results.best_score_))
print("Optimal params: {}".format(grid_results.best_params_))
print("----------")
means = grid_results.cv_results_['mean_test_score']
stds = grid_results.cv_results_['std_test_score']
params = grid_results.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
	print("%f (%f) with: %r" % (mean, stdev, param))
print("----------")


# Training the model

# nn_model.fit(X_train, y_train,
#           batch_size = BATCH_SIZE,
#           epochs = EPOCHS,
#           callbacks = [plot_losses],
#           verbose = 1,
#           validation_data = (X_val, y_val)
#           )

#score = nn_model.evaluate(X_val, y_val, verbose = 0)
#print('Test loss:', score[0])
#print('Test accuracy:', score[1])

y_test = grid.predict(X_test)

output = pd.DataFrame({"ImageId": range(1, len(y_test) + 1), "Label": y_test})
output.to_csv("minst_data/grid_predictions.csv", index = False)

