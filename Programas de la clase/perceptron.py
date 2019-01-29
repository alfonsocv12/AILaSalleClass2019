'''
Simple perceptron model, the following script was modified for the
Artificial intelligence class in Universidad La Salle Chihuahua.

This script is based in the next material:
	* https://pythonmachinelearning.pro/perceptrons-the-first-neural-networks/
	* https://medium.com/@thomascountz/19-line-line-by-line-python-perceptron-b6f113b161f3
	* https://gist.github.com/Thomascountz/77670d1fd621364bc41a7094563a7b9c

Initial Author: Thomas Countz => https://gist.github.com/Thomascountz
'''

import numpy as np

class Perceptron(object):
	def __init__(self, no_of_inputs, epochs=100, learning_rate=0.01):
		self.epochs = epochs
		self.learning_rate = learning_rate
		self.weights = np.zeros(no_of_inputs + 1)

	def predict(self, inputs):

		print("Inputs: %s" % inputs)
		print("Weights: %s", self.weights[1:])

		# Result of multiply all inputs with their respectives weights
		summation = np.dot(inputs, self.weights[1:]) + self.weights[0]

		# Activation function
		if summation > 0:
			activation = 1
		else:
			activation = 0

		# Return activation function result
		return activation

	def train(self, training_inputs, labels):
		# Train the model x epochs
		for epoch in range(self.epochs):
			print("Epoch #: %s" % epoch)
			# Pass to the model every input in its data with their respective label to evaluation
			for inputs, label in zip(training_inputs, labels):
				# Prediction result
				prediction = self.predict(inputs)

				print(prediction)

				# Model evaluation and weights updating
				self.weights[1:] += self.learning_rate * (label - prediction) * inputs

				print("New weights: %s" % self.weights[1:])
				self.weights[0] += self.learning_rate * (label - prediction)

# Data features
training_inputs = np.array([(1,1), (1,0), (0,1), (0,0)])

# Data labels
labels = np.array([1, 0, 0, 1])

# Model creation
perceptron = Perceptron(2)

# Model training
perceptron.train(training_inputs, labels)

# Data generalization
inputs = np.array([1, 1])
print(perceptron.predict(inputs))
# Expected output => 1

inputs = np.array([0, 1])
print(perceptron.predict(inputs))
# Expected output => 0
